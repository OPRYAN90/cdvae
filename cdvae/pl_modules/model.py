from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm
from diffusers.schedulers import EulerDiscreteScheduler as FlowMatchEulerDiscreteScheduler

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Called automatically by PL when loading a checkpoint. 
        We can remove or slice keys in 'checkpoint["state_dict"]' 
        to fix shape mismatches.
        """
        # If we've got logic here already, we can keep it, but let's do the mismatch fix first:
        loaded_state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()  # current (new) model's state_dict

        # Loop over loaded weights, check shape mismatch
        mismatch_keys = []
        for k in list(loaded_state_dict.keys()):
            if k not in model_state_dict:
                # Key doesn't exist in this model at all — can remove if you want
                continue
            if loaded_state_dict[k].shape != model_state_dict[k].shape:
                print(f"[on_load_checkpoint] Removing mismatched param: {k} "
                    f"loaded {loaded_state_dict[k].shape} vs. new {model_state_dict[k].shape}")
                del loaded_state_dict[k]  # remove it so it won't break strict load
                mismatch_keys.append(k)

        # Now store the pruned state_dict back into the checkpoint 
        # so that the actual load_state_dict(...) won't see these mismatch keys.
        checkpoint["state_dict"] = loaded_state_dict
        # Finally let the normal PL logic proceed
        super().on_load_checkpoint(checkpoint)

    def configure_optimizers(self):
        if self.training_phase == 'vae':
            optimizer = hydra.utils.instantiate(
                self.hparams.optim.optimizer, 
                params=self.parameters()
            )
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, 
                optimizer=optimizer
            )
        else:  # diffusion phase
            # Only optimize diffusion parameters
            params = self.noise_prediction_network.parameters()
            optimizer = hydra.utils.instantiate(
                self.hparams.diffusion_optim.optimizer, 
                params=params
            )
            scheduler = hydra.utils.instantiate(
                self.hparams.diffusion_optim.lr_scheduler, 
                optimizer=optimizer
            )
        
        # Add gradient clipping threshold
        self.grad_clip_val = 1.0
        
        # Add gradient norm tracking
        self.grad_norms = []
        
        # Add gradient clipping to the optimizer
        for param_group in optimizer.param_groups:
            param_group['clip_grad_norm'] = self.grad_clip_val
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss" if self.training_phase == 'vae' else "val_noise_loss"
        }


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # shape (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        loss = F.mse_loss(preds, batch.y)
        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        loss = F.mse_loss(preds, batch.y)
        self.scaler.match_device(preds)
        scaled_preds = self.scaler.inverse_transform(preds)
        scaled_y = self.scaler.inverse_transform(batch.y)
        mae = torch.mean(torch.abs(scaled_preds - scaled_y))

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
        }

        if self.hparams.data.prop == 'scaled_lattice':
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mard = mard(batch.angles, pred_angles)

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)
            log_dict.update({
                f'{prefix}_lengths_mae': lengths_mae,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mard': angles_mard,
                f'{prefix}_volumes_mard': volumes_mard,
            })
        return log_dict, loss


class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_phase = self.hparams.training_phase
        self.i = 0
        # Initialize all components
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)
        self.noise_prediction_network = hydra.utils.instantiate(self.hparams.noise_prediction_network)
        # self.fc_xmu = nn.Linear(self.hparams.latent_dim, 3)
        # self.fx_xvar = nn.Linear(self.hparams.latent_dim, 3)
        # self.fc_lmu = nn.Linear(self.hparams.latent_dim, 6)
        # self.fc_lvar = nn.Linear(self.hparams.latent_dim, 6)
        # self.fc_
        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6) #change
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, 256)
        self.fc_frac_coords = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers+4, 3)
        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, 1)


        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None

        if self.training_phase == "diffusion":
            # First print all parameters
            print("\nBefore freezing - trainable parameters:")
            for name, param in self.named_parameters():
                print(f"{name}: {param.requires_grad}")

            # Explicitly freeze specific components
            for name, param in self.named_parameters():
                if not name.startswith('noise_prediction_network'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # Explicitly ensure noise prediction params are trainable
            
            # Verify after freezing
            print("\nAfter freezing - trainable parameters:")
            trainable_params = 0
            for name, param in self.named_parameters():
                is_trainable = param.requires_grad
                print(f"{name}: {is_trainable}")
                if is_trainable:
                    trainable_params += param.numel()
            
            print(f"\nNumber of trainable parameters: {trainable_params}")

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch) #hidden: (batch_size, 256)
        # z_x_mu = self.fc_xmu(hidden)
        # z_x_log_var = self.fc_xvar(hidden)
        mu = self.fc_mu(hidden) #mu transformed from hidden via mlp
        log_var = self.fc_var(hidden) #log_var transformed from hidden via mlp
        z = self.reparameterize(mu, log_var) #z is the reparameterized mu and log_var for sampling
        return mu,log_var,z 

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
            frac_coords = self.predict_frac_coords(z, gt_num_atoms)
        else:
            print("teacher forcing is false WARNING--better be sampling")
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
            frac_coords = self.predict_frac_coords(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom, frac_coords
    def decode_stats_hidden(self, hidden, gt_num_atoms=None, gt_lengths=None, gt_angles=None,teacher_forcing=False):
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(hidden)
            z_l_mu, z_l_logvar, z_ang_mu, z_ang_logvar = (self.predict_lattice_adjusted(hidden, gt_num_atoms)) #do not compress
            z_a_mu, z_a_logvar = self.predict_composition(hidden, gt_num_atoms) #may be compressed but must end up being 89 total atom types
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
            z_x_mu, z_x_logvar = self.predict_frac_coords(hidden, gt_num_atoms)
        else: #do not go here
            num_atoms = self.predict_num_atoms(hidden).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(hidden, num_atoms))
            composition_per_atom = self.predict_composition(hidden, num_atoms)
        return num_atoms, z_l_mu, z_l_logvar, z_ang_mu, z_ang_logvar, z_a_mu, z_a_logvar, z_x_mu, z_x_logvar 

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing, training, phase='vae'):
        # Encode
        mu, log_var, z = self.encode(batch)
        
        # Decode stats from z
        pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, z_a, z_x = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing
        )

        if phase == 'vae':
            # VAE phase - use decoder to get final predictions
            pred_cart, pred_atom_types = self.decoder(
                z=z,
                pred_frac_coords=z_x,
                pred_atom_types=z_a,
                num_atoms=batch.num_atoms,
                lengths=pred_lengths,
                angles=pred_angles,
                batch=batch
            )

            if pred_cart is None and pred_atom_types is None:
                return self.return_zeros(batch)

            return {
                'num_atom_loss': self.num_atom_loss(pred_num_atoms, batch),
                'lattice_loss': self.lattice_loss(pred_lengths_and_angles, batch),
                'coord_loss': self.vae_coord_loss(pred_cart, batch),
                'type_loss': self.vae_type_loss(pred_atom_types, batch),
                'kld_loss': self.kld_loss(mu, log_var),
                'property_loss': self.property_loss(z, batch) if self.hparams.predict_property else 0.,
                'pred_num_atoms': pred_num_atoms,
                'pred_lengths_and_angles': pred_lengths_and_angles,
                'pred_lengths': pred_lengths,
                'pred_angles': pred_angles,
                'pred_cart': pred_cart,
                'pred_atom_types': pred_atom_types,
                'target_frac_coords': batch.frac_coords,
                'target_atom_types': batch.atom_types,
                'hidden': z,
            }

        elif phase == 'diffusion':
            batch_size = z.shape[0]
            
            # Sample timestep/noise level
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=self.device)
            weighting = torch.nn.functional.sigmoid(u)
            timesteps = weighting * 100  # Scale to [0,100]
            
            # Add noise to composition and coordinates
            sigma_a = weighting.repeat_interleave(batch.num_atoms, dim=0)[:, None]  # [sum(N_atoms), 1]
            sigma_x = weighting.repeat_interleave(batch.num_atoms, dim=0)[:, None]  # [sum(N_atoms), 1]
            
            noise_a = torch.randn_like(z_a)
            noise_x = torch.randn_like(z_x)
            
            noised_comp = sigma_a * noise_a + (1.0 - sigma_a) * z_a
            noised_coords = sigma_x * noise_x + (1.0 - sigma_x) * z_x
            
            # Predict noise
            # pred_noise_a = self.noise_predictor_a(noised_comp, timesteps)
            # pred_noise_x = self.noise_predictor_x(noised_coords, timesteps.repeat_interleave(batch.num_atoms, dim=0))
            pred_noise_a, pred_noise_x = self.noise_prediction_network( z=z,
                pred_frac_coords=noised_coords,
                pred_atom_types=noised_comp,
                num_atoms=batch.num_atoms,
                lengths=pred_lengths,
                angles=pred_angles,
                batch=batch,
                timesteps=timesteps #int
                )
            # Compute losses
            loss_a = 2.4 * F.mse_loss(pred_noise_a, noise_a)
            loss_x = F.mse_loss(pred_noise_x, noise_x)
            
            total_loss = loss_a + loss_x

            return {
                'noise_loss': total_loss,
                'noise_loss_a': loss_a,
                'noise_loss_x': loss_x,
                'pred_composition': z_a,
                'pred_coords': z_x,
                'noised_comp': noised_comp,
                'noised_coords': noised_coords,
                'pred_noise_a': pred_noise_a,
                'pred_noise_x': pred_noise_x,
                'noise_a': noise_a,
                'noise_x': noise_x,
                't': timesteps,
                'sigmas': weighting
            }

    def sample(self, num_samples, num_inference_steps=100, device="cuda"):
        """Sample new crystal structures using the rectified flow method.
        
        Args:
            num_samples: Number of crystal structures to generate
            num_atoms_per_crystal: Tensor of shape [B] specifying atoms per crystal
            num_inference_steps: Number of denoising steps
            device: Device to run sampling on
        """
        self.eval()
        
        with torch.no_grad():
            # 1. Initialize noise scheduler
            noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_inference_steps)
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            
            # 2. Initialize hidden representations and get lattice params
            hidden = torch.randn(num_samples, self.hparams.latent_dim, device=device)
            num_atoms_per_crystal, _, lengths, angles, _, _ = self.decode_stats(
                hidden, None, None, None, teacher_forcing=False
            )
            
            # 3. Initialize noised composition and coordinates
            total_atoms = num_atoms_per_crystal.sum()
            noised_composition = torch.randn(total_atoms, 256, device=device)
            noised_coords = torch.randn(total_atoms, 3, device=device)
            
            # 4. Iterative denoising
            for t in range(num_inference_steps):
                timestep = noise_scheduler.timesteps[t].to(device)
                timesteps_per_crystal = timestep.repeat(num_samples)
                
                # Predict noise using noise prediction network
                pred_noise_a, pred_noise_x = self.noise_prediction_network(
                    z=hidden,
                    pred_frac_coords=noised_coords,
                    pred_atom_types=noised_composition,
                    num_atoms=num_atoms_per_crystal,
                    lengths=lengths,
                    angles=angles,
                    batch=None,
                    timesteps=timesteps_per_crystal
                )
                
                # Denoise composition
                scheduler_output_comp = noise_scheduler.step(
                    pred_noise_a, timestep, noised_composition
                )
                noised_composition = scheduler_output_comp.prev_sample
                
                # Denoise coordinates
                scheduler_output_coords = noise_scheduler.step(
                    pred_noise_x, timestep, noised_coords
                )
                noised_coords = scheduler_output_coords.prev_sample

            # 5. Generate final structure using denoised values
            pred_cart, pred_atom_types = self.decoder(
                z=hidden,
                pred_frac_coords=noised_coords,
                pred_atom_types=noised_composition,
                num_atoms=num_atoms_per_crystal,
                lengths=lengths,
                angles=angles,
                batch=None
            )

            # 6. Post-process results
            atom_types = torch.argmax(pred_atom_types, dim=-1) + 1
            frac_coords = cart_to_frac_coords(pred_cart, lengths, angles, num_atoms_per_crystal)
            
            return {
                'atom_types': atom_types,
                'frac_coords': frac_coords,
                'lengths': lengths,
                'angles': angles,
                'num_atoms': num_atoms_per_crystal,
                'pred_cart': pred_cart,
                'pred_atom_types': pred_atom_types,
                'hidden': hidden,
                'composition_per_atom': noised_composition,
                'final_coords': noised_coords
            }
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch
        ) if self.training_phase == 'vae' else False
        
        outputs = self(batch, teacher_forcing, training=True, phase=self.training_phase)
        
        if self.training_phase == 'vae':
            log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        else:  # diffusion phase
            loss = outputs['noise_loss']
            log_dict = {
                'train_noise_loss': loss,
                'train_noise_loss_a': outputs['noise_loss_a'],
                'train_noise_loss_x': outputs['noise_loss_x']
            }
            
        # Check for NaN loss
        if torch.isnan(loss):
            print("WARNING: NaN loss detected")
            return None
            
        # Log gradients
        if self.i % 25 == 0:  # Match your existing logging frequency
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)
            log_dict['gradient_norm'] = total_norm
            
            if total_norm > 100:  # Arbitrary threshold for very large gradients
                print(f"WARNING: Large gradient norm detected: {total_norm}")
        
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))
    
    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles
    
    def predict_lattice_adjusted(self, z, num_atoms=None):
        self.lattice_scaler.match_device(z)
        scaled_preds = self.fc_lattice(z)  # (N, 6) -> (N, 12)
        z_l_mu, z_l_logvar = scaled_preds[:, :3], scaled_preds[:, 3:6]
        z_ang_mu, z_ang_logvar = scaled_preds[:, 6:9], scaled_preds[:, 9:]
        # if self.hparams.data.lattice_scale_method == 'scale_length':
        #     z_l_mu = z_l_mu * num_atoms.view(-1, 1).float()**(1/3)
        #     z_l_logvar = z_l_logvar * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return z_l_mu, z_l_logvar, z_ang_mu, z_ang_logvar

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def predict_frac_coords(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        out = self.fc_frac_coords(z_per_atom)
        return out

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)
    #predict the noise
        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def vae_coord_loss(self, pred_cart, batch):
        # Convert target coordinates
        target_cart_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        
        # Add checks for numerical stability
        if torch.isnan(pred_cart).any() or torch.isinf(pred_cart).any():
            print("WARNING: NaN or Inf detected in predicted coordinates")
            # Return zero loss but with gradient tracking
            return torch.tensor(0.0, requires_grad=True, device=pred_cart.device)
        
        # Clip extremely large predictions
        pred_cart_clipped = torch.clamp(pred_cart, min=-1000.0, max=1000.0)
        
        # Calculate relative distances to make loss more stable
        pred_dists = pred_cart_clipped.unsqueeze(1) - pred_cart_clipped.unsqueeze(0)
        target_dists = target_cart_coords.unsqueeze(1) - target_cart_coords.unsqueeze(0)
        
        # Use Huber loss instead of MSE for robustness
        loss = F.smooth_l1_loss(pred_dists, target_dists, reduction='none')
        
        # Add coordinate-wise loss with smaller weight
        direct_loss = F.smooth_l1_loss(pred_cart_clipped, target_cart_coords, reduction='none')
        
        # Combine losses with weighting
        combined_loss = 0.8 * loss.mean() + 0.2 * direct_loss.mean()
        
        # Add stability check
        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss > 1e6:
            print(f"WARNING: Unstable loss value: {combined_loss}")
            print(f"Max pred value: {pred_cart.abs().max()}")
            print(f"Max target value: {target_cart_coords.abs().max()}")
            return torch.tensor(0.0, requires_grad=True, device=pred_cart.device)
        
        return combined_loss

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def vae_type_loss(self, pred_atom_types, batch):
        target_atom_types = batch.atom_types - 1
        loss = F.cross_entropy(pred_atom_types, target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()
    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    # def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
    #     teacher_forcing = (
    #         self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
    #     outputs = self(batch, teacher_forcing, training=True)
    #     log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
    #     self.log_dict(
    #         log_dict,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )
    #     return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False, phase=self.training_phase)
        
        if self.training_phase == 'vae':
            log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        else:  # diffusion phase
            loss_a = outputs['noise_loss_a']
            loss_x = outputs['noise_loss_x']
            total_loss = loss_a + loss_x
            
            log_dict = {
                'val_noise_loss': total_loss,  # This is what the scheduler monitors
                'val_noise_loss_a': loss_a,
                'val_noise_loss_x': loss_x
            }
            loss = total_loss
        
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss
    def return_zeros(self, batch):
        device = batch.num_atoms.device
        return {
            'num_atom_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'lattice_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'coord_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'type_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'kld_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'property_loss': torch.tensor(0.0, requires_grad=True, device=device),
            'pred_num_atoms': torch.tensor(0.0, device=device),
            'pred_lengths_and_angles': torch.tensor(0.0, device=device),
            'pred_lengths': torch.tensor(0.0, device=device),
            'pred_angles': torch.tensor(0.0, device=device),
            'pred_cart': torch.tensor(0.0, device=device),
            'pred_atom_types': torch.tensor(0.0, device=device),
            'target_frac_coords': torch.tensor(0.0, device=device),
            'target_atom_types': torch.tensor(0.0, device=device),
            'hidden': torch.tensor(0.0, device=device),
        }

    def dummies(self, log_dict, prefix):
        loss = 1000
        property_loss = 1000
        num_atom_accuracy = 1000
        lengths_mard = 1000
        angles_mae = 1000
        volumes_mard = 1000
        type_accuracy = 1000
        log_dict.update({
            f'{prefix}_loss': loss,
            f'{prefix}_property_loss': property_loss,
            f'{prefix}_natom_accuracy': num_atom_accuracy,
            f'{prefix}_lengths_mard': lengths_mard,
            f'{prefix}_angles_mae': angles_mae,
            f'{prefix}_volumes_mard': volumes_mard,
            f'{prefix}_type_accuracy': type_accuracy,
        })
        return log_dict, loss
    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        # composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']
        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            # self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)
        if self.i % 25 == 0:
            print(f"Iteration {self.i}, loss: {loss}")
        self.i += 1
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            # f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            if pred_lengths_and_angles.shape == torch.Size([]):
                print("Issues with batch so skipping val")
                return self.dummies(log_dict, prefix)
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss

    def on_before_optimizer_step(self, optimizer):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
