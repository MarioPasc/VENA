"""Pydantic config schema for the latent perceptual loss (LPL).

Bridges the decoder-feature perceptual stage described in
``.claude/notes/changes/decoder_perceptual_loss_s3.md`` (§2.4 + §2.6) to a
single round-trippable YAML / dict block that both the S3 training engine
and the ``decoder_lpl_profile`` preflight read against. The values
in the production config (``A``, ``w_l``, ``t_min``, ``outlier_k``,
``alpha``, …) are *measured*, not defaulted — the preflight emits a
``decision.json`` v1.0 that pins them on the actual MAISI decoder rather
than porting Berrada-2025's SD-VAE defaults that the §4.7c N=4 pilot
showed are wrong-signed on MAISI.

The defaults in this file are conservative starting values for unit
tests and dev smokes; production runs override every knob from the
preflight's ``decision.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LplConfig(BaseModel):
    """Recipe for the latent perceptual loss.

    Fields
    ------
    A : list[int]
        Decoder block indices to read features from. The preflight's
        error-concentration analysis picks the top two; default ``[2, 5]``
        per the design-doc §3.5 + §4.7c provisional finding.
    w_l : dict[int, float]
        Per-block depth weight. Keys must be a subset of ``A``. The
        Berrada-2025 inverse-upscale rule (``1 / scale``) is *wrong-signed*
        on MAISI per the N=4 pilot — defaults here follow the measured
        magnitude curve (``w_2=1.0``, ``w_5=2.0``) and the preflight pins
        the final values.
    t_min : float
        High-SNR gate. The LPL term is zero for ``t <= t_min`` (Berrada
        2025 §3.3). The preflight finds the knee in the
        ``x̂_1`` reliability curve and overrides this default.
    lambda_img : float
        Outer coupling weight in the composite loss. Consumed by the
        future S3 ``CompositeLoss`` wiring; not used inside ``LplLoss``.
    alpha : dict[str, float]
        Per-region weight in the §2.6 region-weighted variant. Keys
        ``"wt"`` and ``"notwt"``; production default ``(2, 3)`` per the
        §4.7c provisional finding.
    p : dict[str, int]
        Per-region exponent of the feature-distance accumulation. The
        L^p_r generalisation of the squared error. Production default
        ``p=2`` everywhere; the ``(p_wt=1, p_notwt=3)`` variant is run
        only after the stability sweep.
    outlier_k : dict[int, float]
        Per-block ``k * MAD`` outlier-mask threshold (Berrada 2025).
        Defaults to ``5.0`` for every requested block; the preflight
        widens it on heavy-tailed blocks.
    soft_region : bool
        If True, weight each voxel by ``soft_wt`` (the clip-summed
        ``tumor_latent``) rather than thresholded ``m_wt``. The
        engineering-cost-free §2.6 sweep variant.
    grad_checkpoint_segments : int
        0 → no checkpointing. ≥2 → ``checkpoint_sequential`` segments on
        the partial decoder. Default 0; the §4.6 K=5 prototype uses 2.
    compute_placement : Literal["a", "b"]
        ``"a"`` — in-process on the training GPU (default; only mode
        supported in PR-1). ``"b"`` — cross-device Variant B per design
        §3.6 (deferred).
    region_set : list[str]
        Per-region keys. Defaults to ``["wt", "notwt"]``. The deferred
        3-region variant (``["necrotic", "edema", "enhancing", "notwt"]``)
        is accepted at config-parse time so the YAML can declare it but
        only the 2-region variant is currently consumed by ``LplLoss``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    A: list[int] = Field(default_factory=lambda: [2, 5])
    w_l: dict[int, float] = Field(default_factory=lambda: {2: 1.0, 5: 2.0})
    t_min: float = 0.7
    lambda_img: float = 1.0
    alpha: dict[str, float] = Field(default_factory=lambda: {"wt": 2.0, "notwt": 3.0})
    p: dict[str, int] = Field(default_factory=lambda: {"wt": 2, "notwt": 2})
    outlier_k: dict[int, float] = Field(default_factory=lambda: {2: 5.0, 5: 5.0})
    soft_region: bool = False
    grad_checkpoint_segments: int = 0
    compute_placement: Literal["a", "b"] = "a"
    region_set: list[str] = Field(default_factory=lambda: ["wt", "notwt"])

    @model_validator(mode="after")
    def _check_keys(self) -> LplConfig:
        a_set = set(self.A)
        if not a_set:
            raise ValueError("A must be non-empty")
        if min(a_set) < 0:
            raise ValueError("A entries must be non-negative block indices")
        # w_l / outlier_k must key exactly on A.
        if set(self.w_l) != a_set:
            raise ValueError(f"w_l keys {sorted(self.w_l)} must equal A {sorted(a_set)}")
        if set(self.outlier_k) != a_set:
            raise ValueError(
                f"outlier_k keys {sorted(self.outlier_k)} must equal A {sorted(a_set)}"
            )
        # alpha / p must key on region_set.
        rs = set(self.region_set)
        if set(self.alpha) != rs:
            raise ValueError(f"alpha keys {sorted(self.alpha)} must equal region_set {sorted(rs)}")
        if set(self.p) != rs:
            raise ValueError(f"p keys {sorted(self.p)} must equal region_set {sorted(rs)}")
        for rname, pv in self.p.items():
            if pv not in (1, 2, 3):
                raise ValueError(f"p[{rname!r}] must be 1, 2, or 3; got {pv}")
        # t_min in [0, 1).
        if not (0.0 <= self.t_min < 1.0):
            raise ValueError(f"t_min must be in [0, 1); got {self.t_min}")
        if self.grad_checkpoint_segments < 0 or self.grad_checkpoint_segments == 1:
            raise ValueError(
                "grad_checkpoint_segments must be 0 (off) or >= 2; got "
                f"{self.grad_checkpoint_segments}"
            )
        if self.compute_placement == "b":
            # PR-1 ships Variant A only; the cross-device path lands in a
            # follow-up. Fail fast instead of silently running Variant A.
            raise ValueError(
                "compute_placement='b' (cross-device) is not implemented in"
                " this PR; use 'a' (in-process). See"
                " .claude/notes/changes/decoder_perceptual_loss_s3.md §3.6."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> LplConfig:
        """Load an :class:`LplConfig` from a YAML file.

        The YAML maps top-level keys directly to the schema fields. Dict
        fields (``w_l``, ``alpha``, ``outlier_k``) accept str keys which
        Pydantic coerces; for ``w_l`` / ``outlier_k`` we pre-cast str keys
        to int so the validator can compare against ``A``.
        """
        text = Path(path).read_text()
        data: dict[str, Any] = yaml.safe_load(text) or {}
        # YAML int keys round-trip fine for ints, but writing
        # ``w_l: { 2: 1.0 }`` in YAML survives as ``{2: 1.0}`` while
        # ``w_l: {"2": 1.0}`` survives as ``{"2": 1.0}``. Normalise the
        # latter to ints so the validator sees uniform keys.
        for k in ("w_l", "outlier_k"):
            if k in data and data[k] is not None:
                data[k] = {int(kk): float(vv) for kk, vv in data[k].items()}
        return cls.model_validate(data)

    @classmethod
    def from_decision(
        cls,
        decision_path: Path | str,
        *,
        lambda_img: float = 1.0,
        soft_region: bool | None = None,
        grad_checkpoint_segments: int | None = None,
    ) -> LplConfig:
        """Build the loss recipe from a ``decoder_lpl_profile`` decision.json.

        The decision contract pins ``A``, ``w_l``, ``t_min``, ``outlier_k``,
        and the global ``region_recipe``. The S3 train YAML supplies the
        outer ``lambda_img`` coupling weight (the decision file does not
        commit a value — that is an experiment-design knob) and may
        override ``soft_region`` / ``grad_checkpoint_segments`` for
        deployment-specific tuning.

        Per-cohort α overrides in ``region_recipe.per_cohort_overrides``
        are NOT consumed here — the production training loop applies the
        global recipe. Per-cohort plumbing is a follow-up; for now, the
        global α=(2.0, 3.0) is the contract.
        """
        from vena.preflight.decoder_lpl_profile.decision import (
            assert_decoder_lpl_decision_valid,
        )

        decision = assert_decoder_lpl_decision_valid(Path(decision_path))
        recipe = decision.region_recipe
        return cls(
            A=list(decision.A_recommended),
            w_l={int(k): float(v) for k, v in decision.w_l.items()},
            t_min=float(decision.t_min),
            lambda_img=float(lambda_img),
            alpha={"wt": float(recipe.alpha_wt), "notwt": float(recipe.alpha_notwt)},
            p={"wt": 2, "notwt": 2},  # production default; sweep axis
            outlier_k={int(k): float(v) for k, v in decision.outlier_k.items()},
            soft_region=(
                bool(soft_region) if soft_region is not None else bool(recipe.soft_region)
            ),
            grad_checkpoint_segments=(
                int(grad_checkpoint_segments) if grad_checkpoint_segments is not None else 0
            ),
            compute_placement="a",
            region_set=["wt", "notwt"],
        )
