Crystal Structure Generation via VAE and Latent Space Diffusion
This repository contains the implementation of a novel approach for accelerated material discovery using a combination of Variational Autoencoder (VAE) and latent space diffusion models. The method enables efficient exploration of crystal structures while maintaining physical and chemical validity.
Key Features

Comprehensive Material Representation: Handles lattice parameters, atomic species, and coordinates while respecting crystallographic symmetries
Advanced Neural Architectures: Utilizes adapted DimeNet++ and GemNet architectures for geometric deep learning
Two-Stage Generation: Combines VAE for structure encoding with rectified flow diffusion for controlled exploration
Large-Scale Training: Trained on 1.5M thermodynamically stable materials from Materials Project and Alexandria databases
Property-Guided Generation: Supports targeted generation of materials with specific properties

Model Architecture

Encoder: Adapted DimeNet++ for processing crystal structures
Decoder: Modified GemNet architecture with time conditioning
Diffusion Process: Rectified flow formulation with Fourier time-conditioning
Property Prediction: Specialized MLPs for property-guided generation
