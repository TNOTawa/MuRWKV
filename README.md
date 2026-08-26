# Agent Harness Starter

This starter separates Agent guidance into four layers:

```text
AGENTS.md                 Always-loaded map
    ↓ pointers
docs/agents/*.md          On-demand decision / quality / harness guidance
    ↓ promotion
tests / CI / linters      Machine-enforced constraints
    ↓ hard boundary
sandbox / permissions     Capability enforcement
```

## Why

A giant instruction file can crowd out task-relevant context and accumulate stale rules. The goal here is to keep the always-loaded file small while preserving strict engineering discipline through progressive disclosure and mechanical verification.

## Files

- `AGENTS.md` — copy to a new repository root and fill the project map.
- `docs/agents/DECISIONS.md` — targeted clarification and `/grill-me`-compatible decision protocol.
- `docs/agents/QUALITY.md` — multi-oracle QA and test-strength guidance.
- `docs/agents/HARNESS.md` — where constraints should live: root, nested file, Skill, CI, or permissions.
- `docs/agents/RESEARCH.md` — evidence and caveats; intended mainly for maintainers.
- `templates/AGENTS.module.md` — minimal nested/subproject delta template.

## Recommended adoption order

1. Copy `AGENTS.md`; fill only real project facts.
2. Point `Full QA command` at one real command.
3. Add machine checks only for invariants the project genuinely needs.
4. Add a nested `AGENTS.md` only when a subtree truly differs.
5. Use `/grill-me` or the decision protocol only when material ambiguity exists or you explicitly want a design stress-test.
6. Periodically delete instructions that became redundant, stale, or mechanically enforced.

The starter is intentionally not a complete CI system: language-specific QA belongs to the project, not a universal prompt.
