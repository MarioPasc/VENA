# pGAN-cGAN upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/icon-lab/pGAN-cGAN |
| Vendored at SHA | `b4ca7047cb98c5b81014912ed6f58dc0ed501c5f` |
| Vendored on | 2026-06-14 |
| Licence | MIT (with embedded BSD-2-Clause for the pix2pix/CycleGAN ancestor) — vendoring permitted |
| Citation | Dar, S. U. H. et al. "Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks." *IEEE TMI* 38(10):2375–2388, 2019. DOI: 10.1109/TMI.2019.2901750 |

## Scope of use in VENA

Only the **pGAN** branch (paired pix2pix-style) is used. The cGAN branch (CycleGAN-style)
is preserved untouched but never invoked — VENA has aligned ground-truth T1c, so unpaired
translation is not relevant.

The upstream code targets PyTorch 0.3.1 / Python 3.5.5. Minimal in-place patches were
applied to make it run on PyTorch 2.x — see `PATCHES.md`.

The upstream loader (`data/__init__.py::CreateDataset`) expects a single per-phase
`data.mat` HDF5 file with `data_x` and `data_y` arrays. VENA does NOT use this loader.
The VENA wrapper (`src/vena/competitors/pgan_cgan/dataset.py`) implements its own
`UCSFPDGMSliceDataset` and `CreateDataLoader` that mimic the same returned dict shape
(`{'A', 'B', 'A_paths', 'B_paths'}`) while reading directly from the UCSF-PDGM image H5.

## What is NOT modified

- The pGAN model architecture and losses (`models/pgan_model.py`, `models/networks.py`).
- The training-options definitions (`options/{base,train,test}_options.py`). VENA
  constructs an `argparse.Namespace`-equivalent object that matches these schemas, but
  does not invoke the upstream argparse parser.

## Reproducing the snapshot

```bash
cd src/external/pgan_cgan
git clone --depth 1 https://github.com/icon-lab/pGAN-cGAN.git upstream
cd upstream && git checkout b4ca7047cb98c5b81014912ed6f58dc0ed501c5f
rm -rf .git
# Apply patches listed in PATCHES.md in order.
```
