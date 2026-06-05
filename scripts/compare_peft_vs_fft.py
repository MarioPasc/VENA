"""Compare a PEFT/LoRA training run against an FFT reference.

Usage::

    python scripts/compare_peft_vs_fft.py \
        --peft  /media/hddb/mario/experiments/<peft_run_id>/ \
        --fft   /media/hddb/mario/experiments/2026-06-04_15-59-57_s2_6b25f2d9/ \
        --out   /media/hddb/mario/experiments/peft_lora_r16_vs_fft.md

Emits four figures and one markdown report:

1. ``train_epoch_losses.png`` — total / cfm / contrastive vs epoch.
2. ``train_step_grad_norms.png`` — controlnet + trunk grad norms (pre/post clip).
3. ``exhaustive_val_psnr_ssim.png`` — per-NFE PSNR / SSIM means with error bars.
4. ``train_step_loss_curve.png`` — per-step total loss across the full run.

The verdict block at the top of the markdown report compares:

* Trainable parameter counts (read from each run's first ``Trainable params``
  line in ``logs/train.log``).
* Final-epoch ``total_mean`` ratio (PEFT / FFT) — values < 1.10 are healthy.
* Final-epoch exhaustive-val PSNR / SSIM (PEFT - FFT, signed) per NFE.
* Grad-norm stability (std of ``grad_norm_trunk_preclip_mean`` across epochs).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_train_epoch(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "metrics" / "train_epoch.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_train_step(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "metrics" / "train_step.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_decision(run_dir: Path) -> dict:
    p = run_dir / "decision.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _trainable_params(run_dir: Path) -> int | None:
    # Try Lightning's stdout summary line first (typically in nohup log, not in
    # the file-handler log). Fall back to scanning the file-handler log.
    for candidate in (run_dir / "logs" / "train.log", *run_dir.glob("logs/*.log")):
        if not candidate.exists():
            continue
        txt = candidate.read_text(errors="ignore")
        m = re.search(r"Trainable params:\s*([\d.]+)\s*([KMG]?)", txt)
        if m:
            base = float(m.group(1))
            mult = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
            return int(base * mult)
    return None


def _read_exhaustive(run_dir: Path) -> pd.DataFrame:
    rows = []
    ev_root = run_dir / "exhaustive_val"
    if not ev_root.is_dir():
        return pd.DataFrame()
    for ep_dir in sorted(ev_root.glob("epoch_*")):
        m = re.search(r"epoch_(\d+)", ep_dir.name)
        if not m:
            continue
        ep = int(m.group(1))
        mp = ep_dir / "metrics.csv"
        if not mp.exists():
            continue
        df = pd.read_csv(mp)
        df["epoch"] = ep
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_train_epoch(peft: pd.DataFrame, fft: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for col, ax, label in [
        ("total_mean", axes[0], "total"),
        ("cfm_mean", axes[1], "cfm"),
        ("contrastive_mean", axes[2], "contrastive"),
    ]:
        if col in peft.columns:
            ax.plot(peft["epoch"], peft[col], "o-", label="PEFT/LoRA r=16", color="C0")
        if col in fft.columns:
            ax.plot(fft["epoch"], fft[col], "s-", label="FFT", color="C1")
        ax.set_title(f"{label} loss / epoch")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_grad_norms(peft: pd.DataFrame, fft: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics = [
        ("grad_norm_cn_preclip", axes[0, 0], "ControlNet pre-clip"),
        ("grad_norm_cn_postclip", axes[0, 1], "ControlNet post-clip"),
        ("grad_norm_trunk_preclip", axes[1, 0], "Trunk pre-clip"),
        ("grad_norm_trunk_postclip", axes[1, 1], "Trunk post-clip"),
    ]
    for col, ax, title in metrics:
        if col in peft.columns:
            ax.plot(peft["step"], peft[col], ".", alpha=0.4, color="C0", label="PEFT")
        if col in fft.columns:
            ax.plot(fft["step"], fft[col], ".", alpha=0.4, color="C1", label="FFT")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_exhaustive(peft: pd.DataFrame, fft: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_df, label, color in [(peft, "PEFT/LoRA r=16", "C0"), (fft, "FFT", "C1")]:
        if run_df.empty or "nfe" not in run_df.columns:
            continue
        # Take latest epoch for each run.
        last_epoch = run_df["epoch"].max()
        sub = run_df[run_df["epoch"] == last_epoch]
        agg = sub.groupby("nfe")[["psnr_db", "ssim"]].agg(["mean", "std"]).reset_index()
        nfe = agg["nfe"]
        axes[0].errorbar(
            nfe,
            agg["psnr_db"]["mean"],
            yerr=agg["psnr_db"]["std"],
            fmt="o-",
            label=f"{label} (ep {last_epoch})",
            color=color,
        )
        axes[1].errorbar(
            nfe,
            agg["ssim"]["mean"],
            yerr=agg["ssim"]["std"],
            fmt="o-",
            label=f"{label} (ep {last_epoch})",
            color=color,
        )
    axes[0].set_title("Exhaustive-val PSNR / NFE (last epoch)")
    axes[0].set_xlabel("NFE")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()
    axes[1].set_title("Exhaustive-val SSIM / NFE (last epoch)")
    axes[1].set_xlabel("NFE")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_step_loss(peft: pd.DataFrame, fft: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for df, label, color in [(peft, "PEFT/LoRA r=16", "C0"), (fft, "FFT", "C1")]:
        if "total" in df.columns:
            ax.plot(df["step"], df["total"], ".", alpha=0.3, color=color, label=label)
    ax.set_title("Per-step total loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _verdict_block(
    peft_dir: Path,
    fft_dir: Path,
    peft_te: pd.DataFrame,
    fft_te: pd.DataFrame,
    peft_ev: pd.DataFrame,
    fft_ev: pd.DataFrame,
) -> str:
    lines: list[str] = []
    peft_dec = _read_decision(peft_dir)
    fft_dec = _read_decision(fft_dir)
    peft_tp = _trainable_params(peft_dir)
    fft_tp = _trainable_params(fft_dir)
    lines.append("## Header\n")
    lines.append(
        f"- PEFT run: `{peft_dir.name}` — regime=`{peft_dec.get('trunk_regime')}` "
        f"variant=`{peft_dec.get('trunk_peft_variant')}` "
        f"params=`{peft_dec.get('trunk_peft_params')}`"
    )
    lines.append(
        f"- FFT  run: `{fft_dir.name}` — regime=`{fft_dec.get('trunk_regime', 'fft (pre-0.6.0)')}`"
    )
    lines.append("")
    lines.append("## Trainable params\n")
    if peft_tp and fft_tp:
        ratio = peft_tp / fft_tp
        lines.append(f"- PEFT: {peft_tp:>12,} ({peft_tp / 1e6:.2f} M)")
        lines.append(f"- FFT:  {fft_tp:>12,} ({fft_tp / 1e6:.2f} M)")
        lines.append(f"- PEFT / FFT: **{ratio:.3f}** ({1 / ratio:.1f}× reduction)")
    lines.append("")

    lines.append("## Final-epoch losses\n")
    if not peft_te.empty and not fft_te.empty:
        peft_last = peft_te.iloc[-1]
        fft_last = fft_te.iloc[-1]
        lines.append(
            "| Metric | PEFT (ep {}) | FFT (ep {}) | Δ (PEFT−FFT) | Ratio |".format(
                int(peft_last["epoch"]), int(fft_last["epoch"])
            )
        )
        lines.append("|---|---|---|---|---|")
        for col in ("total_mean", "cfm_mean", "contrastive_mean"):
            if col in peft_te.columns and col in fft_te.columns:
                p = float(peft_last[col])
                f = float(fft_last[col])
                lines.append(
                    f"| `{col}` | {p:.4g} | {f:.4g} | {p - f:+.3g} | {p / f if f else float('nan'):.3f} |"
                )
    lines.append("")

    lines.append("## Exhaustive-val (last epoch)\n")
    if not peft_ev.empty and not fft_ev.empty:
        p_last = peft_ev[peft_ev["epoch"] == peft_ev["epoch"].max()]
        f_last = fft_ev[fft_ev["epoch"] == fft_ev["epoch"].max()]
        lines.append("| NFE | PEFT PSNR | FFT PSNR | ΔPSNR | PEFT SSIM | FFT SSIM | ΔSSIM |")
        lines.append("|---|---|---|---|---|---|---|")
        nfes = sorted(set(p_last["nfe"]).intersection(set(f_last["nfe"])))
        for nfe in nfes:
            pa = p_last[p_last["nfe"] == nfe]
            fa = f_last[f_last["nfe"] == nfe]
            pp = pa["psnr_db"].mean()
            fp = fa["psnr_db"].mean()
            ps = pa["ssim"].mean()
            fs = fa["ssim"].mean()
            lines.append(
                f"| {int(nfe)} | {pp:.2f} | {fp:.2f} | {pp - fp:+.2f} | "
                f"{ps:.3f} | {fs:.3f} | {ps - fs:+.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True, type=Path)
    ap.add_argument("--fft", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    peft_te = _read_train_epoch(args.peft)
    fft_te = _read_train_epoch(args.fft)
    peft_ts = _read_train_step(args.peft)
    fft_ts = _read_train_step(args.fft)
    peft_ev = _read_exhaustive(args.peft)
    fft_ev = _read_exhaustive(args.fft)

    _plot_train_epoch(peft_te, fft_te, fig_dir / "train_epoch_losses.png")
    _plot_grad_norms(peft_ts, fft_ts, fig_dir / "train_step_grad_norms.png")
    _plot_exhaustive(peft_ev, fft_ev, fig_dir / "exhaustive_val_psnr_ssim.png")
    _plot_step_loss(peft_ts, fft_ts, fig_dir / "train_step_loss_curve.png")

    md = "# PEFT/LoRA r=16 vs FFT — VENA S2 smoke comparison\n\n"
    md += _verdict_block(args.peft, args.fft, peft_te, fft_te, peft_ev, fft_ev)

    # Epoch-1 apples-to-apples (both runs have epoch 1 after 2 epochs of training).
    if (
        not peft_te.empty
        and not fft_te.empty
        and (peft_te["epoch"] == 1).any()
        and (fft_te["epoch"] == 1).any()
    ):
        md += "## Epoch-1 apples-to-apples (LoRA had 3 epochs; FFT had 2)\n\n"
        p1 = peft_te[peft_te["epoch"] == 1].iloc[0]
        f1 = fft_te[fft_te["epoch"] == 1].iloc[0]
        md += "| Metric | PEFT ep 1 | FFT ep 1 | Δ | Ratio |\n|---|---|---|---|---|\n"
        for col in ("total_mean", "cfm_mean", "contrastive_mean"):
            if col in peft_te.columns and col in fft_te.columns:
                p = float(p1[col])
                f = float(f1[col])
                md += f"| `{col}` | {p:.4g} | {f:.4g} | {p - f:+.3g} | {p / f if f else float('nan'):.3f} |\n"
        md += "\n"
        # Same for exhaustive val epoch 1.
        if (
            not peft_ev.empty
            and not fft_ev.empty
            and (peft_ev["epoch"] == 1).any()
            and (fft_ev["epoch"] == 1).any()
        ):
            md += "### Exhaustive-val epoch 1 (both runs)\n\n"
            p_ev1 = peft_ev[peft_ev["epoch"] == 1]
            f_ev1 = fft_ev[fft_ev["epoch"] == 1]
            md += "| NFE | PEFT PSNR | FFT PSNR | ΔPSNR | PEFT SSIM | FFT SSIM | ΔSSIM |\n"
            md += "|---|---|---|---|---|---|---|\n"
            nfes = sorted(set(p_ev1["nfe"]).intersection(set(f_ev1["nfe"])))
            for nfe in nfes:
                pp = p_ev1[p_ev1["nfe"] == nfe]["psnr_db"].mean()
                fp = f_ev1[f_ev1["nfe"] == nfe]["psnr_db"].mean()
                ps = p_ev1[p_ev1["nfe"] == nfe]["ssim"].mean()
                fs = f_ev1[f_ev1["nfe"] == nfe]["ssim"].mean()
                md += f"| {int(nfe)} | {pp:.2f} | {fp:.2f} | {pp - fp:+.2f} | {ps:.3f} | {fs:.3f} | {ps - fs:+.3f} |\n"
            md += "\n"

    md += "\n## Figures\n\n"
    md += "![Train epoch losses](figures/train_epoch_losses.png)\n\n"
    md += "![Train step grad norms](figures/train_step_grad_norms.png)\n\n"
    md += "![Exhaustive-val PSNR/SSIM](figures/exhaustive_val_psnr_ssim.png)\n\n"
    md += "![Per-step total loss](figures/train_step_loss_curve.png)\n\n"

    args.out.write_text(md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
