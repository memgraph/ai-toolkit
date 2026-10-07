# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`<package>/docs/adr/`** — read ADRs that touch the area you're about to work in, plus root `docs/adr/` for system-wide decisions once it exists.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Multi-context uv workspace. `CONTEXT-MAP.md` at the root lists every context.

```
/
├── CONTEXT-MAP.md
├── unstructured2graph/
│   ├── CONTEXT.md
│   └── docs/adr/
└── context-graph/
    ├── agent-context-graph/{CONTEXT.md, docs/adr/}
    ├── actions-graph/CONTEXT.md
    ├── skills-graph/CONTEXT.md
    ├── sessions-graph/CONTEXT.md
    ├── memory-graph/CONTEXT.md   ← redirect only: old name for sessions-graph
    └── eval/CONTEXT.md
```

ADRs are scoped to a package (`<package>/docs/adr/`). There's no root `docs/adr/` yet; system-wide decisions would go there.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
