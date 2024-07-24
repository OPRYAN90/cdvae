import torch
import torch.nn as nn
import torch.nn.functional as F

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.gemnet import GemNetT


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
class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=128,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.latent_dim = latent_dim
        self.gemnet = GemNetT(
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
        self.fc_atom = build_mlp(hidden_dim, hidden_dim, 2, MAX_ATOMIC_NUM)
        # self.fc_lengths = nn.Linear(hidden_dim, 3)
        # self.fc_angles = nn.Linear(hidden_dim, 3)
        #other way:
        # self.fc_atom = build_mlp(latent_dim, latent_dim*2, 1, MAX_ATOMIC_NUM)
        # self.fc_lengths = build_mlp(latent_dim*20, latent_dim, 1, 3)
        # self.fc_angles = build_mlp(latent_dim*20, latent_dim, 1, 3)
        # self.fc_hidden = build_mlp(hidden_dim, latent_dim*2, 1, latent_dim)
        # self.len_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)   
        # self.angles_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):
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
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coords = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        ) 
        pred_atom_types = self.fc_atom(h)
        # pred_atom_types = self.fc_atom(insider)
        # #split:
        # insider_split, mask = split_atoms(insider, num_atoms, self.latent_dim)
        # insider_view = insider_split.view(-1, self.latent_dim*20) #warning bsz
        # pred_lengths, pred_angles = self.fc_lengths(insider_view), self.fc_angles(insider_view)
        # if transformer:
        # pred_lengths, pred_angles = self.len_attention(insider, insider, insider), self.angles_attention(insider, insider, insider)
        return pred_cart_coords, pred_atom_types
