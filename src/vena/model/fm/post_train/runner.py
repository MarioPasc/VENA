"""End-to-end orchestrator for the post-training plotting routine.

`PostTrainRunner(run_dir).run()` produces `<run_dir>/plots/` populated with
the three figures described in `plan/context-we-have-finished-async-piglet.md`.
Per-plot failures are caught and logged at WARNING; the runner returns the
list of paths it successfully wrote.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["PostTrainRunner", "render_post_train_plots"]

_FIG_LOSS_TOTAL = "loss_total_grad"
_FIG_LOSS_COHORT = "loss_per_cohort_grad"
_FIG_PARETO = "pareto_psnr_ssim"


class PostTrainRunner:
    """Render the post-training plot bundle for a finished run directory.

    Parameters
    ----------
    run_dir : Path
        Training run directory (must contain `metrics/train_epoch.csv`).
    formats : tuple[str, ...]
        Output formats; one file per format per plot. Defaults to PNG only.
    """

    def __init__(self, run_dir: Path, *, formats: tuple[str, ...] = ("png",)) -> None:
        self._configure_matplotlib_backend()
        from vena.model.fm.post_train.plotting_styles import apply_plot_settings

        apply_plot_settings()

        self.run_dir = Path(run_dir)
        self.formats = tuple(formats)
        self.plots_dir = self.run_dir / "plots"

    @staticmethod
    def _configure_matplotlib_backend() -> None:
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)

    def run(self) -> list[Path]:
        """Render every plot. Returns the list of files written.

        Per-plot exceptions are caught and logged; the run continues.
        """
        from vena.model.fm.post_train.loaders import (
            discover_exhaustive_val,
            load_train_epoch_csv,
        )
        from vena.model.fm.post_train.plot_loss_grad import (
            plot_per_cohort_grad,
            plot_total_grad,
        )
        from vena.model.fm.post_train.plot_pareto import plot_pareto

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        try:
            df = load_train_epoch_csv(self.run_dir)
        except FileNotFoundError as exc:
            logger.warning("post-train: %s; skipping loss/grad plots", exc)
            df = None

        if df is not None:
            written.extend(self._render_for_each_format(_FIG_LOSS_TOTAL, plot_total_grad, df))
            written.extend(self._render_for_each_format(_FIG_LOSS_COHORT, plot_per_cohort_grad, df))

        exhaustive = discover_exhaustive_val(self.run_dir)
        if exhaustive:
            written.extend(self._render_for_each_format(_FIG_PARETO, plot_pareto, exhaustive))
        else:
            logger.warning(
                "post-train: no exhaustive_val/epoch_NNN/ data under %s; skipping Pareto plot",
                self.run_dir,
            )

        logger.info("post-train: wrote %d plot file(s) under %s", len(written), self.plots_dir)
        return written

    def _render_for_each_format(self, stem: str, fn, *args) -> list[Path]:
        out: list[Path] = []
        for fmt in self.formats:
            path = self.plots_dir / f"{stem}.{fmt}"
            try:
                out.append(Path(fn(*args, path)))
            except Exception as exc:
                logger.warning(
                    "post-train: %s.%s failed: %s",
                    stem,
                    fmt,
                    exc,
                    exc_info=True,
                )
        return out


def render_post_train_plots(run_dir: Path, *, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Functional shortcut: build a `PostTrainRunner` and run it once."""
    return PostTrainRunner(run_dir, formats=formats).run()
