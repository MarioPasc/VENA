"""CSV / artifact loaders for the post-training plotting routine.

Pure functions that take a run directory and return tidy pandas DataFrames /
dictionaries. Never invoke matplotlib; never write to disk. The plotting
modules consume the structures defined here.

The schemas are:

* `metrics/train_epoch.csv` — one row per epoch. Columns:
    epoch, step, n_steps,
    <loss>_mean, <loss>_std for loss in {cfm, contrastive, reconstruction, total},
    samples_per_sec_{mean,std},
    grad_norm_{cn,trunk}_{pre,post}clip_{mean,std},
    t_mean_{mean,std}, gpu_mem_peak_mb_{mean,std},
    cfm_cohort_<C>_{mean,std} for each cohort in the run.

* `exhaustive_val/epoch_NNN/metrics.csv` — one row per (patient_id × nfe).
    cohort, epoch, patient_id, nfe,
    psnr_db, ssim,
    psnr_db_wt, ssim_wt,                 # whole-tumour region
    psnr_db_bg, ssim_bg,                 # complement of WT (includes outside-brain)
    psnr_db_nwt, ssim_nwt,               # healthy brain: foreground AND NOT(wt)
    latent_mse, latent_l1, latent_cosine, gen_sec, decode_sec.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "active_grad_norm_series",
    "detect_active_cohorts",
    "detect_active_losses",
    "discover_exhaustive_val",
    "load_train_epoch_csv",
]

_EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")

#: Loss components the post-training plot ever decomposes. `reconstruction`
#: is present in the CSV header for forward compatibility but excluded by
#: design (user decision 2026-06-09).
LOSS_COMPONENTS: tuple[str, ...] = ("cfm", "contrastive")


def load_train_epoch_csv(run_dir: Path) -> pd.DataFrame:
    """Load `<run_dir>/metrics/train_epoch.csv` into a DataFrame.

    Parameters
    ----------
    run_dir : Path
        Training run directory (`experiments/<run_id>/` or equivalent).

    Returns
    -------
    pandas.DataFrame
        One row per epoch, sorted by `epoch` ascending.

    Raises
    ------
    FileNotFoundError
        The CSV does not exist under `run_dir/metrics/`.
    """
    import pandas as pd

    csv_path = Path(run_dir) / "metrics" / "train_epoch.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train_epoch.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def discover_exhaustive_val(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Return `{epoch_index: per-epoch metrics DataFrame}`.

    Scans `<run_dir>/exhaustive_val/epoch_NNN/metrics.csv`. Missing or empty
    CSVs are skipped silently.

    Parameters
    ----------
    run_dir : Path

    Returns
    -------
    dict[int, pandas.DataFrame]
        Keys are integer epoch indices in ascending order. Empty dict if
        the `exhaustive_val/` directory does not exist.
    """
    import pandas as pd

    base = Path(run_dir) / "exhaustive_val"
    if not base.exists():
        return {}

    found: dict[int, pd.DataFrame] = {}
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        m = _EPOCH_DIR_RE.match(child.name)
        if m is None:
            continue
        epoch = int(m.group(1))
        csv_path = child / "metrics.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        # Stamp the epoch onto each row when missing — older CSVs may omit.
        if "epoch" not in df.columns:
            df = df.assign(epoch=epoch)
        found[epoch] = df
    return {k: found[k] for k in sorted(found)}


def detect_active_losses(df: pd.DataFrame) -> list[str]:
    """Return the loss components that actually carry signal in the run.

    A component is "active" if its `<name>_mean` column exists and contains
    at least one finite, non-zero value. The order matches `LOSS_COMPONENTS`.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_train_epoch_csv`.
    """
    import numpy as np

    active: list[str] = []
    for name in LOSS_COMPONENTS:
        col = f"{name}_mean"
        if col not in df.columns:
            continue
        values = df[col].to_numpy()
        if np.any(np.isfinite(values) & (values != 0.0)):
            active.append(name)
    return active


def detect_active_cohorts(df: pd.DataFrame, *, loss: str = "cfm") -> list[str]:
    """List the cohorts whose per-cohort loss column carries signal.

    Looks for columns named `<loss>_cohort_<cohort>_mean` and keeps those
    with at least one finite non-zero value.

    Parameters
    ----------
    df : pandas.DataFrame
    loss : str
        Loss component prefix (default `"cfm"`).

    Returns
    -------
    list[str]
        Cohort names in CSV column order.
    """
    import numpy as np

    prefix = f"{loss}_cohort_"
    suffix = "_mean"
    cohorts: list[str] = []
    for col in df.columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        cohort = col[len(prefix) : -len(suffix)]
        values = df[col].to_numpy()
        if np.any(np.isfinite(values) & (values != 0.0)):
            cohorts.append(cohort)
    return cohorts


def active_grad_norm_series(df: pd.DataFrame) -> list[str]:
    """Return the postclip grad-norm branches present and non-NaN.

    Always includes `"cn"` (ControlNet); includes `"trunk"` only if the
    `grad_norm_trunk_postclip_mean` column has at least one finite value
    (i.e. the run had a trainable trunk).
    """
    import numpy as np

    branches: list[str] = []
    if "grad_norm_cn_postclip_mean" in df.columns:
        branches.append("cn")
    trunk_col = "grad_norm_trunk_postclip_mean"
    if trunk_col in df.columns:
        values = df[trunk_col].to_numpy()
        if np.any(np.isfinite(values)):
            branches.append("trunk")
    return branches
