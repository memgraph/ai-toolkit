# Skills Graph

Stores reusable agent skills, records how sessions use them. Data can later support skill evolution + recommendation.

## Language

**Skill**:
Reusable agent capability conforming to the Agent Skills specification.
_Avoid_: Internal skill concept

**Skill Definition**:
Authored Agent Skills directory + required `SKILL.md` file describing a skill.
_Avoid_: Skill node

**Skill Node**:
Persisted graph rep of a **Skill** in Skills Graph.
_Avoid_: SKILL.md

**Skill Usage**:
Recorded fact that an agent session used a skill during its work.
_Avoid_: Skill reference, skill mention

**Skill Surfacing**:
Skill appearing in list/search output. Weaker evidence than reading its `SKILL.md`. Current graph stores both on `USED_SKILL`; access source recorded in relationship's `actions` list.
_Avoid_: Skill usage

**Skills Graph**:
Graph component preserving reusable agent skills, tracking session use, supporting future evolution + recommendation.
_Avoid_: Skill registry, skill storage

**Procedure** (designed, not implemented; [#262](https://github.com/memgraph/ai-toolkit/issues/262), [#272](https://github.com/memgraph/ai-toolkit/issues/272)):
Planned node for reusable patterns mined from Action history/touched files. Sequence mining: repeated tool calls, no LLM. Content mining: LLM finds repeated file patterns. Kept separate from Skills — generated patterns may be noisy, don't conform to Agent Skills spec. No human approval gate planned.
_Avoid_: Mined Skill, Skill candidate (implies sharing Skill's namespace/guarantees — deliberately doesn't)

## Relationships

- **Skills Graph** belongs to broader **Context Graph** family.
- **Skill** follows Agent Skills specification: https://agentskills.io/specification
- **Skill Definition** can be represented as a **Skill Node**.
- **Skill Usage** links Session or subagent **Agent Node** to a **Skill** via `USED_SKILL`. `SkillGraphConnector` uses event's `agent_name` to pick container. Same relationship also used for **Skill Surfacing**; `actions` records whether evidence = list/search result or direct use.
- **Skills Graph** consumes relevant **Event Protocol** events via `SkillGraphConnector`.
- **Skills Graph** treats activity as skill usage when it invokes a skill (a skill tool, Claude Code's `Skill` tool) or reads a local `SKILL.md` file.
- Claude Code `Skill` tool call creates **Skill Usage** keyed on `tool_input.skill`; `plugin:name` reduces to `name`. The `SKILL.md` Claude Code would load (project `.claude/skills`, `~/.claude/skills`, `~/.agents/skills`, plugin cache) populates the Skill Node when found.
- Reading a local `SKILL.md` creates **Skill Usage**, only when the call itself reads it: a file-read tool's argument, or a read program's (`cat`, `sed`, `head`, ...) argument in a shell command. Every such file in one command counts. Readable file -> its metadata populates Skill Node; else fallback metadata used.
- Not **Skill Usage**: staging, diffing, searching or writing a `SKILL.md`; a path in a subagent prompt or other metadata; unexpanded shell paths (`$D/SKILL.md`, globs); any other Markdown with `name`/`description` frontmatter (Claude Code memory files have that shape). Injected skill content counts only with Claude Code's `Base directory for this skill:` marker.
- Skill in search/list output creates **Skill Surfacing** evidence on same `USED_SKILL` relationship, distinct `actions` value.
- **Procedure** and **Skill Node** never share identity or relationship types. Future promotion from Procedure to curated Skill must be explicit.

## Example dialogue

> **Dev:** "Is Skills Graph just where we store skill markdown?"
> **Domain expert:** "No. Storage is the first capability; the graph exists so skills can later evolve and be recommended based on usage."

> **Dev:** "Can we define our own skill format?"
> **Domain expert:** "No. A Skill follows the Agent Skills specification; Skills Graph decides how that skill is represented and related in the graph."

> **Dev:** "Codex read `/skills/memgraph-console/SKILL.md` with `sed`. Is that usage?"
> **Domain expert:** "Yes. Connector treats local `SKILL.md` path as usage. Reading `README.md` or mentioning `memgraph-console` isn't enough."

## Flagged ambiguities

- "storage" under-describes component. Resolved: **Skills Graph** = preservation + usage tracking + basis for evolution/recommendation.
- "skill" shouldn't be local invention. Resolved: **Skill** = Agent Skills spec skill; **Skill Node** = graph rep.
- "usage" broader than activation. Resolved: **Skill Usage** = umbrella fact; `USED_SKILL` = current relationship name.
- "surfaced" vs "used": distinct evidence concepts. Both create `USED_SKILL`; `actions` values distinguish source.
- Should auto-mined pattern become a Skill directly? Resolved: no. Generated patterns may be wrong, don't guarantee Agent Skills conformance — stay Procedures until explicitly curated.
