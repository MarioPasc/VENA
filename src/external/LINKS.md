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
- BrainSegFounder (SwinUNETR feature_size=48, encoder-only SSL — for the VENA segmenter, tasks 11/17/18). SSL ONLY. Never the finetuned BraTS segmenter (it leaks labels+masks+T1ce = L1+L2+L3).
    - ⚠ LOAD-COVERAGE CAVEAT (measured 2026-07-23, S4): loading into a *standard* MONAI `SwinUNETR(feature_size=48)` via `load_bsf_encoder(model.backbone, ...)` gives Arm B UKB **125/142 = 0.880** (1-ch stem correctly skipped) but Arm A BraTS **only 126/198 = 0.636** — 72 skipped = 56 extra `swinViT.layers3.*` keys + 16 SSL task-head keys absent from our build. **This is likely a config mismatch (BSF may use `use_v2=True` and/or non-default `depths`/`num_heads`), NOT a genuinely deeper architecture** — BSF is documented as a standard Swin. **OPEN TICKET (S5, parallel): reconstruct the exact BSF SwinUNETR config from the ckpt keys so Arm A loads its full encoder** (only the 16 SSL heads should legitimately drop). See SESSIONS.md §S5 ticket + [[project_s4_segmenter_library]].
    - UKB-SSL (Arm B, LEAK-FREE HEADLINE/PRIMARY — healthy UK Biobank, no BraTS patients, no T1ce; deep encoder blocks transfer, input stem may re-init [verify at load]): /media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt
    - BraTS-SSL (Arm A, domain-matched COMPARATOR/upper-bound — 5 per-fold ckpts; NOTE it leaks BraTS-GLI patient images + T1ce exposure that OOF cannot fix, hence not the headline): /media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt
    - DO NOT USE (max leakage): /media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_{0..4}.pt
- Documentation folder:
    - /media/mpascual/Sandisk2TB/research/vena/docs

Picasso:
- Comments: "4 exa nodes of 8xA100 40GB VRAM ; up to 128GB RAM. Loginexa: separate V100-DGXS-32GB interactive node at 10.248.7.200 (vena-v100 env, sm_70)."
- MAISI-V2:
    - VAE-GAN model: /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt
    - Flow Matching model: /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/diff_unet_3d_rflow-mr.pt
- UCSF_PDGM:
    - (Image domain, schema of 19/05/2026): /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSF_PDGM/h5/UCSFPDGM_image.h5
    - (Latent domain): /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/UCSF_PDGM/h5/UCSFPDGM_latents.h5
- BrainSegFounder (SSL ONLY; see Local for arm mapping + the exclude-finetuned rule. Filenames expected to mirror Local — [verify on Picasso at first use]):
    - UKB-SSL (Arm B, leak-free headline): /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt
    - BraTS-SSL (Arm A, comparator): /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt
    - DO NOT USE (finetuned, leaks): /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/BrainSegFounder_finetuned_BraTS/

Research Server 3:
- Comments: "2 GPUs RTX 4090 24GB VRAM ; Server with GUI Desktop, no need for SLURM jobs"
- MAISI-V2:
    - VAE-GAN model: /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt
    - Flow Matching model: /media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt
- UCSF_PDGM:
    - (Image domain, schema of 19/05/2026): /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5
    - (Latent domain): /media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_latents.h5
