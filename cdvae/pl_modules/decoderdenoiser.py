import torch
import torch.nn as nn
import torch.nn.functional as F

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.gemnetdenoiser import GemNetTDenoiser

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

def split_atoms(input_tensor, num_atoms, latent_dim,max_num_atoms=20):
    output_tensor = torch.randn(num_atoms.sum(), latent_dim)  # Shape: (num_atoms.sum(), latent_dim)
    # Split the output tensor into individual molecule tensors
    molecule_tensors = torch.split(output_tensor, num_atoms.tolist())

    # Pad the tensors to ensure they all have the same shape (max_num_atoms, latent_dim)
    max_num_atoms = 20
    padded_tensors = []
    masks = []

    for mol_tensor in molecule_tensors:
        num_atoms_in_molecule = mol_tensor.shape[0]
        
        # Pad the tensor
        padding = (0, 0, 0, max_num_atoms - num_atoms_in_molecule)  # Pad only the second dimension
        padded_tensor = F.pad(mol_tensor, padding, "constant", 0)
        padded_tensors.append(padded_tensor)
        
        # Create the mask
        mask = torch.zeros(max_num_atoms, dtype=float)
        mask[:num_atoms_in_molecule] = 1
        masks.append(mask)

    # Stack the padded tensors and masks to form the final batch tensors
    padded_batch_tensor = torch.stack(padded_tensors)  # Shape: (batch_size, max_num_atoms, latent_dim)
    mask_tensor = torch.stack(masks)  # Shape: (batch_size, max_num_atoms)
    return padded_batch_tensor, mask_tensor
class GemNetTDenoiserDecoder(nn.Module):
    """Denoiser with GemNetT."""

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=128,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        super(GemNetTDenoiserDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.latent_dim = latent_dim
        self.gemnet = GemNetTDenoiser(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )
    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms, lengths, angles, batch, timesteps):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # Attempt using predicted lengths and angles
        pred_z_a_noise, pred_z_x_noise = self.gemnet(
                z=z,
                frac_coords=pred_frac_coords,
                atom_types=pred_atom_types,
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
                edge_index=None,
                to_jimages=None,
                num_bonds=None,
                timesteps=timesteps
            )
        return pred_z_a_noise, pred_z_x_noise