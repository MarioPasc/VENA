"""Competitor benchmark routines.

Each subdirectory is one competitor model (e.g. ``pgan_cgan/``). Routines follow
the VENA preflight-pattern conventions: one positional YAML arg, frozen Pydantic
config with ``from_yaml``, ``Engine.run() -> Path``, decision.json on completion.

The competitor's training receives ONLY VENA-normalised data and applies no
augmentation. See ``src/vena/competitors/<name>/`` for the wrapper library and
``.claude/skills/integrate-competitor/SKILL.md`` for the integration recipe.
"""

from __future__ import annotations
