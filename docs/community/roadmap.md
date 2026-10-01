# Roadmap

AgentTune doesn't keep a separate roadmap document. The items below are the known gaps
and next steps that are already visible in the codebase and [Changelog](changelog.md),
collected here in one place rather than scattered across module comments. If you want to
work on one of these, check [Contributing](contributing.md) first, and search
[open issues](https://github.com/Lexsi-Labs/AgentTune/issues): someone may already be on it.

## Known gaps

- **A generated per-module API reference isn't wired to the docs site yet.**
  No `mkdocstrings` page is in this nav yet, so nothing renders module docs from
  docstrings. In the meantime, [Python API](../user-guide/agentic-spine.md)
  is a hand-written inventory of every standalone-usable class/function across the package,
  and [`src/agenttune/agentic/README.md`](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/README.md)
  is the de-facto reference for the spine.
- **The full user guide isn't complete here yet.** This site is
  intentionally spine-focused; broader upstream docs are future work. See the note at the
  top of [`mkdocs.yml`](https://github.com/Lexsi-Labs/AgentTune/blob/main/mkdocs.yml).

See [Known Issues](known-issues.md) for the full list of things a code-level audit found
broken, orphaned, or silently degraded. Most are narrow (one function, one edge case), a
few are the significant gaps above.

## Recently shipped

See the [Changelog](changelog.md) for the full history. Most recently, `SelfHealLoop` can
now be gated by the retrain/gate/deploy stage (`RetrainingTrigger` / `DeploymentGate`)
instead of training unconditionally on every detected failure.
