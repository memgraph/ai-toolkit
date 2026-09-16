# Skills Graph

Skills Graph stores reusable agent skills and records how sessions use them. This
data can later support skill evolution and recommendation.

## Language

**Skill**:
A reusable agent capability that conforms to the Agent Skills specification.
_Avoid_: Internal skill concept

**Skill Definition**:
The authored Agent Skills directory and required `SKILL.md` file that describe a skill.
_Avoid_: Skill node

**Skill Node**:
The persisted graph representation of a **Skill** in Skills Graph.
_Avoid_: SKILL.md

**Skill Usage**:
A recorded fact that an agent session used a skill during its work.
_Avoid_: Skill reference, skill mention

**Skill Surfacing**:
A skill appearing in list or search output. This is weaker evidence than reading
its `SKILL.md`. The current graph stores both on `USED_SKILL` and records the
access source in the relationship's `actions` list.
_Avoid_: Skill usage

**Skills Graph**:
The graph component that preserves reusable agent skills, tracks their use by agent sessions, and supports future skill evolution and recommendation.
_Avoid_: Skill registry, skill storage

**Procedure** (designed, not yet implemented; see
[#262](https://github.com/memgraph/ai-toolkit/issues/262) and
[#272](https://github.com/memgraph/ai-toolkit/issues/272)):
A planned node for reusable patterns mined from Action history or touched files.
Sequence mining will find repeated tool calls without an LLM; content mining will
use an LLM to find repeated file patterns. Procedures stay separate from Skills
because generated patterns may be noisy and do not conform to the Agent Skills
specification. No human approval gate is planned.
_Avoid_: Mined Skill, Skill candidate (implies it shares Skill's namespace/guarantees; it deliberately doesn't)

## Relationships

- **Skills Graph** belongs to the broader **Context Graph** family.
- A **Skill** follows the Agent Skills specification: https://agentskills.io/specification
- A **Skill Definition** can be represented as a **Skill Node**.
- **Skill Usage** links a Session or subagent **Agent Node** to a **Skill** with
  `USED_SKILL`. `SkillGraphConnector` uses the event's `agent_name` to choose the
  container ([#275](https://github.com/memgraph/ai-toolkit/issues/275), ticket
  #280). Current persistence uses this same relationship for **Skill Surfacing**;
  `actions` records whether evidence came from a list/search result or direct use.
- **Skills Graph** consumes relevant **Event Protocol** events through `SkillGraphConnector`.
- **Skills Graph** interprets agent activity as skill usage when the activity refers to a known skill or a local `SKILL.md` file.
- Reading a local path ending in `SKILL.md` creates **Skill Usage**. If the file is
  readable, its metadata populates the Skill Node; otherwise fallback metadata is
  used. Reading other Markdown or mentioning a skill in prose does not count.
- A skill in search/list output creates **Skill Surfacing** evidence on the same
  `USED_SKILL` relationship, with a distinct `actions` value.
- A **Procedure** and **Skill Node** never share identity or relationship types.
  Future promotion from Procedure to curated Skill must be explicit.

## Example dialogue

> **Dev:** "Is Skills Graph just where we store skill markdown?"
> **Domain expert:** "No. Storage is the first capability; the graph exists so skills can later evolve and be recommended based on usage."

> **Dev:** "Can we define our own skill format?"
> **Domain expert:** "No. A Skill follows the Agent Skills specification; Skills Graph decides how that skill is represented and related in the graph."

> **Dev:** "Codex read `/skills/memgraph-console/SKILL.md` with `sed`. Is that usage?"
> **Domain expert:** "Yes. The connector treats a local `SKILL.md` path as usage. Reading `README.md` or mentioning `memgraph-console` is not enough."

## Flagged ambiguities

- "storage" under-describes the component. Resolved: **Skills Graph** includes preservation, usage tracking, and the basis for evolution and recommendation.
- "skill" should not be a local invention. Resolved: **Skill** means an Agent Skills specification skill; **Skill Node** means the graph representation.
- "usage" is broader than activation. Resolved: use **Skill Usage** for the umbrella fact and `USED_SKILL` as the current relationship name.
- "surfaced" and "used" are distinct evidence concepts. Both create
  `USED_SKILL`; the relationship's `actions` values distinguish their source.
- Whether an auto-mined pattern should become a Skill directly. Resolved: no.
  Generated patterns may be wrong and do not guarantee Agent Skills conformance,
  so they remain Procedures until explicitly curated.
