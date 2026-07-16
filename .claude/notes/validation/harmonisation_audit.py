"""Diagnostic: does Phase-1 harmonisation corrupt already-normalised predictions?

Read-only. Measures in-brain intensity moments of `_raw` vs `_harmonised` against
the real T1c, per (method, cohort). No metrics, no writes.

Hypothesis: the latent tier (VENA/C4-C7) decodes through the MAISI VAE into the
already-normalised target space; re-applying percentile_normalise stretches
[p0,p99.5]->[0,1] and inflates bulk tissue, because the model under-saturates the
bright tail so its raw p99.5 < 1. The image tier (C0-C3) emits method-native
units and legitimately needs the mapping.
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference"
)
COHORTS = ["UCSF-PDGM", "BraTS-GLI", "IvyGAP"]
N_SCANS = 4

SELECTION_NFE = {
    "C0-Identity": 1,
    "C1-pGAN-t1pre": 1,
    "C1-pGAN-t2": 1,
    "C1-pGAN-flair": 1,
    "C2-ResViT": 1,
    "C3-SynDiff-t1pre": 4,
    "C3-SynDiff-t2": 4,
    "C3-SynDiff-flair": 4,
    "C4-3D-DiT": 5,
    "C5-T1C-RFlow": 5,
    "C6-3D-LDDPM": 1000,
    "C7-3D-Latent-Pix2Pix": 1,
    "VENA-S1-v3a": 5,
    "VENA-S1-v3b": 5,
    "VENA-S1-v3b-rw": 5,
    "VENA-S3-LPL-b2c": 5,
}
LATENT_TIER = {
    "C4-3D-DiT",
    "C5-T1C-RFlow",
    "C6-3D-LDDPM",
    "C7-3D-Latent-Pix2Pix",
    "VENA-S1-v3a",
    "VENA-S1-v3b",
    "VENA-S1-v3b-rw",
    "VENA-S3-LPL-b2c",
}


def _s(arr):
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def main() -> int:
    # EXCLUDE stale smoke shards: only picasso_shard_* / picasso_ped_* are production.
    shard_dirs = sorted(
        d
        for d in glob.glob(os.path.join(ROOT, "*"))
        if os.path.isdir(os.path.join(d, "predictions"))
        and os.path.basename(d).startswith(("picasso_shard_", "picasso_ped_"))
    )
    print(f"production shards: {[os.path.basename(d) for d in shard_dirs]}\n")

    hdr = (
        f"{'method':22s} {'cohort':12s} {'tier':7s} "
        f"{'raw_mean':>9s} {'raw_p995':>9s} {'harm_mean':>9s} "
        f"{'real_mean':>9s} {'MAE_raw':>8s} {'MAE_harm':>9s} {'verdict':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for cohort in COHORTS:
        ref_cache = {}
        for shard in shard_dirs:
            rp = os.path.join(shard, "references", f"{cohort}.h5")
            if os.path.exists(rp):
                with h5py.File(rp, "r") as fr:
                    ids = _s(fr["metadata/scan_id"][:])
                    for k in range(min(N_SCANS, len(ids))):
                        ref_cache[ids[k]] = (
                            fr["reference/t1c_real_harmonised"][k],
                            fr["masks/brain"][k] > 0,
                        )
                break
        if not ref_cache:
            continue

        for shard in shard_dirs:
            for mdir in sorted(glob.glob(os.path.join(shard, "predictions", "*"))):
                method = os.path.basename(mdir)
                nfe = SELECTION_NFE.get(method)
                if nfe is None:
                    continue
                p = os.path.join(mdir, cohort, f"nfe_{nfe:03d}.h5")
                if not os.path.exists(p):
                    continue
                rows = []
                with h5py.File(p, "r") as fp:
                    pids = _s(fp["metadata/scan_id"][:])
                    for sid, (real, brain) in ref_cache.items():
                        if sid not in pids:
                            continue
                        i = pids.index(sid)
                        raw = fp["predictions/t1c_synthetic_raw"][i][brain]
                        har = fp["predictions/t1c_synthetic_harmonised"][i][brain]
                        rl = real[brain]
                        rows.append(
                            (
                                raw.mean(),
                                np.percentile(raw, 99.5),
                                har.mean(),
                                rl.mean(),
                                np.abs(raw - rl).mean(),
                                np.abs(har - rl).mean(),
                            )
                        )
                if not rows:
                    continue
                a = np.array(rows, dtype=np.float64).mean(axis=0)
                tier = "latent" if method in LATENT_TIER else "image"
                # Verdict: is harmonisation making this method WORSE?
                verdict = (
                    "HARM WORSE" if a[5] > a[4] * 1.15 else ("ok" if a[5] <= a[4] else "harm~raw")
                )
                print(
                    f"{method:22s} {cohort:12s} {tier:7s} "
                    f"{a[0]:9.4f} {a[1]:9.4f} {a[2]:9.4f} {a[3]:9.4f} "
                    f"{a[4]:8.4f} {a[5]:9.4f} {verdict:>10s}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
