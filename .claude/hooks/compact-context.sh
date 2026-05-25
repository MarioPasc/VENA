#!/usr/bin/env bash
# Printed into the session when a compaction starts.
# Gives the assistant a fast project-state snapshot so it does not have to
# re-explore the tree after compacting.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

printf '=== VENA session resume ===\n'
printf 'pwd:  %s\n' "$PWD"
printf 'date: %s\n\n' "$(date -Iseconds)"

printf '%s\n' '--- git ---'
git status --short --branch 2>/dev/null | head -n 20 || printf '%s\n' '(not a git repo)'
printf '\n%s\n' 'recent commits:'
git log --oneline -n 5 2>/dev/null || printf '%s\n' '(no commits)'

printf '\n%s\n' '--- top-level layout ---'
ls -1 --color=never 2>/dev/null | head -n 30

printf '\n%s\n' '--- preflight decisions written so far ---'
if [[ -d artifacts ]]; then
    find artifacts -type f -name 'decision.json' 2>/dev/null | head -n 10
else
    printf '%s\n' '(no artifacts/ directory)'
fi

printf '\n%s\n' '--- key reference paths ---'
printf '%s\n' '  proposal:   /media/mpascual/Sandisk2TB/research/vena/docs/proposal.md'
printf '%s\n' '  literature: /media/mpascual/Sandisk2TB/research/vena/docs/literature.md'
printf '%s\n' '  external:   src/external/LINKS.md'
printf '%s\n' '  rules:      .claude/rules/'
printf '%s\n' '  env:        ~/.conda/envs/vena/bin/python'
