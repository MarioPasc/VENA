# Paths to project-related stuff

Local:
- Comments: "Two possible workstations, either RTX 3060 12GB VRAM 32 GB RAM or RTX 4060 8GB VRAM 12GB RAM" 
- MAISI-V2:
    - code: /media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR
    - VAE-GAN model: /media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt
    - Flow Matching model: /media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt
- UCSF_PDGM:
    - Source Data (.nii.gz format): /media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/source
    - H5 Data Folder: /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSFPDGM_image.h5
    - H5 Latents: /media/mpascual/MeningD2/MAISI_VAEGAN_LATENTS/UCSF_PDGM
- Documentation folder:
    - /media/mpascual/Sandisk2TB/research/vena/docs

Picasso:
- Comments: "4 exa nodes of 8xA100 40GB VRAM ; up to 128GB RAM"
- MAISI-V2:
    - VAE-GAN model: /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt
    - Flow Matching model: /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/diff_unet_3d_rflow-mr.pt
- UCSF_PDGM:
    - (Image domain, schema of 19/05/2026): /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSFPDGM_image.h5

Research Server 3:
- Comments: "2 GPUs RTX 4090 24GB VRAM ; Server with GUI Desktop, no need for SLURM jobs"
- MAISI-V2:
    - VAE-GAN model: /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt
    - Flow Matching model: /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt
- UCSF_PDGM:
    - (Image domain, schema of 19/05/2026): /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5
    - (Latent domain): /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_latents.h5
