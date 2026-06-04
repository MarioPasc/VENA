"""Emit the per-cohort × per-rank YAMLs for the offline-aug routine.

Run once; commit the YAMLs. Re-run if the cohort list or path convention
changes. The generator is the source of truth; do not hand-edit the
emitted files.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# CV cohorts on server 3 (from routines/fm/train/configs/corpus/corpus_server3.json,
# filtered to role == "cv").
COHORTS = [
    {
        "name": "UCSF-PDGM",
        "stem": "ucsf_pdgm",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5",
    },
    {
        "name": "BraTS-GLI",
        "stem": "brats_gli",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/BRATS_GLI/PRE_OPERATIVE/h5",
    },
    {
        "name": "UPENN-GBM",
        "stem": "upenn_gbm",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/upenn_gbm/h5/UPENN-GBM_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/upenn_gbm/h5",
    },
    {
        "name": "IvyGAP",
        "stem": "ivy_gap",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/ivy_gap/h5/IvyGAP_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/ivy_gap/h5",
    },
    {
        "name": "LUMIERE",
        "stem": "lumiere",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/lumiere/h5/LUMIERE_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/lumiere/h5",
    },
    {
        "name": "REMBRANDT",
        "stem": "rembrandt",
        "source_image_h5": "/media/hddb/mario/data/GLIOMAS/rembrandt/h5/REMBRANDT_image.h5",
        "out_dir": "/media/hddb/mario/data/GLIOMAS/rembrandt/h5",
    },
]

AE_CHECKPOINT = "/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt"
AUG_PIPELINE = "routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml"
DEDUP_DECISION = "/media/hddb/mario/artifacts/preflights/cohort_dedup/LATEST/decision.json"
EQUIVARIANCE_DECISION = "/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json"
WORLD_SIZE = 2


def shard_config(cohort: dict, rank: int) -> dict:
    suffix = f"_rank{rank}"
    return {
        "cohort": cohort["name"],
        "source_image_h5": cohort["source_image_h5"],
        "autoencoder_checkpoint": AE_CHECKPOINT,
        "aug_pipeline_yaml": AUG_PIPELINE,
        "output_dir": f"/media/hddb/mario/artifacts/offline_aug/maisi/{cohort['stem']}",
        "image_aug_h5_path": f"{cohort['out_dir']}/{cohort['stem']}_image_aug{suffix}.h5",
        "latent_aug_h5_path": f"{cohort['out_dir']}/{cohort['stem']}_latents_aug{suffix}.h5",
        "modalities": ["t1pre", "t1c", "t2", "flair"],
        "variants": ["v1", "v2", "v3", "v4"],
        "device": "cuda",
        "precision_mode": "autocast",
        "autoencoder_norm_float16": True,
        "inference_mode": "auto",
        "depth_pad_base": 8,
        "percentile_lower": 0.0,
        "percentile_upper": 99.5,
        "percentile_foreground_only": True,
        "world_size": WORLD_SIZE,
        "rank": rank,
        "seed": 42,
        "overwrite": False,
        "log_level": "INFO",
        "dedup": {
            "enabled": True,
            "decisions_path": DEDUP_DECISION,
        },
        "qc": {
            "enabled": True,
            "n_patients_per_variant": 3,
            "figure_filename_template": "roundtrip_{cohort}_{variant}.png",
            "equivariance_decision_path": EQUIVARIANCE_DECISION,
            "psnr_tolerance_db": 2.0,
            "ssim_tolerance": 0.02,
        },
        "merge": {
            "enabled": False,
            "shards": [],
        },
    }


def merge_config(cohort: dict) -> dict:
    """A separate config that runs the merge step over the two ranks' shards.

    Reuses the same engine but disables the build/encode + QC and just calls
    the merge helpers. We do this by pointing image_aug_h5_path at the
    *merged* output and listing both rank shards under merge.shards. The
    engine sees those, dispatches to merge_aug_image_h5_shards +
    _merge_latent_shards, and skips the build/encode/QC blocks if rank=0
    + world_size=1 + a flag... actually the cleanest is to add a small
    bash wrapper that calls merge_aug_image_h5_shards directly. For now,
    emit a Python helper script per cohort.
    """
    return {
        "_note": "merge is run via scripts/merge_offline_aug_shards.py (one call per cohort)",
        "cohort": cohort["name"],
        "shards_image": [
            f"{cohort['out_dir']}/{cohort['stem']}_image_aug_rank0.h5",
            f"{cohort['out_dir']}/{cohort['stem']}_image_aug_rank1.h5",
        ],
        "shards_latent": [
            f"{cohort['out_dir']}/{cohort['stem']}_latents_aug_rank0.h5",
            f"{cohort['out_dir']}/{cohort['stem']}_latents_aug_rank1.h5",
        ],
        "merged_image_aug_h5": f"{cohort['out_dir']}/{cohort['stem']}_image_aug.h5",
        "merged_latents_aug_h5": f"{cohort['out_dir']}/{cohort['stem']}_latents_aug.h5",
    }


def smoke_config() -> dict:
    """Tiny smoke config for a fast end-to-end on 2 UCSF-PDGM patients."""
    cfg = shard_config(COHORTS[0], rank=0)
    cfg["output_dir"] = "/media/hddb/mario/artifacts/offline_aug/maisi/smoke"
    cfg["image_aug_h5_path"] = "/tmp/UCSFPDGM_image_aug_smoke.h5"
    cfg["latent_aug_h5_path"] = "/tmp/UCSFPDGM_latents_aug_smoke.h5"
    cfg["world_size"] = 1  # no sharding
    cfg["rank"] = 0
    cfg["overwrite"] = True
    cfg["dedup"]["enabled"] = False  # smoke
    cfg["dedup"]["decisions_path"] = None
    cfg["qc"]["n_patients_per_variant"] = 1
    cfg["limit_source_rows"] = 2  # only the first 2 non-test scans
    return cfg


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    configs_dir = here / "routines" / "offline_aug" / "maisi" / "configs"
    merges_dir = configs_dir / "merges"
    merges_dir.mkdir(parents=True, exist_ok=True)

    for cohort in COHORTS:
        for rank in range(WORLD_SIZE):
            path = configs_dir / f"{cohort['stem']}_rank{rank}.yaml"
            with path.open("w") as f:
                yaml.safe_dump(shard_config(cohort, rank), f, sort_keys=False)
            print(f"wrote {path}")
        m_path = merges_dir / f"{cohort['stem']}_merge.yaml"
        with m_path.open("w") as f:
            yaml.safe_dump(merge_config(cohort), f, sort_keys=False)
        print(f"wrote {m_path}")

    smoke_path = configs_dir / "smoke_ucsf_pdgm.yaml"
    with smoke_path.open("w") as f:
        yaml.safe_dump(smoke_config(), f, sort_keys=False)
    print(f"wrote {smoke_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
