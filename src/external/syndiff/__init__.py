"""Vendored snapshot of the official SynDiff repository.

The actual code lives under ``upstream/``. This file exists so the namespace
is importable as a package marker. The runner under
``src/vena/competitors/syndiff/`` extends ``sys.path`` to point at
``upstream/`` so that ``backbones.*`` and ``utils.*`` resolve to the
vendored modules.
"""
