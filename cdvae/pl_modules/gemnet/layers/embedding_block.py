"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

from .base_layers import Dense

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
import torch.nn as nn
import torch
import numpy as np

def build_mlp_atom_emb(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = []
    
    # Input layer
    input_layer = nn.Linear(in_dim, hidden_dim)
    torch.nn.init.uniform_(input_layer.weight, a=-np.sqrt(3), b=np.sqrt(3))
    mods += [input_layer, nn.ReLU()]
    
    # Hidden layers
    for i in range(fc_num_layers - 1):
        hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.uniform_(hidden_layer.weight, a=-np.sqrt(3), b=np.sqrt(3))
        mods += [hidden_layer, nn.ReLU()]
    
    # Output layer
    output_layer = nn.Linear(hidden_dim, out_dim)
    torch.nn.init.uniform_(output_layer.weight, a=-np.sqrt(3), b=np.sqrt(3))
    mods.append(output_layer)
    
    return nn.Sequential(*mods)


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(64, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z): #fix atom embedding or use cartesian prediction as latent
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
        activation=None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(
            in_features, out_features, activation=activation, bias=False
        )

    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat(
            [h_s, h_t, m_rbf], dim=-1
        )  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st
