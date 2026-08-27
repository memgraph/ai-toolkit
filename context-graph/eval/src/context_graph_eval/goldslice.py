"""The gold slice: recall questions answered from a real harness session.

Tier 1 injects fixture text straight into the graph, which is fast and cheap but
never exercises the capture layer. The gold slice runs a genuine Claude Code
session with hooks live, so what gets scored is what the pipeline actually
records -- the only place `skills-graph` and subagent nesting get any eval
coverage at all (#308 found no benchmark covers either).

**One question to start** (#307). Each is a real, billed session, so the slice
grows by *carrier* -- where the fact physically lives in the graph -- rather
than by round number.

The first carrier is a fact that exists **only inside a subagent**. Top-level
recall is already covered by Tier 1's fixtures, whereas the nested carrier has a
demonstrated silent-failure mode: #281 found ``get_session_actions()`` does a
single-hop ``HAS_ACTION`` match, so once subagent activity moved under
``(:Agent)`` (#278), reconciliation would stop seeing it -- no entities, no
Episode mention, and no error. That was caught once by reading code. As a
gold-slice question it is caught automatically, forever.
"""

from typing import TYPE_CHECKING

from deepeval.dataset import Golden

from .convert.longmemeval import DEFAULT_REVISION

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from actions_graph import ActionsGraph

#: The fact under test: the first 12 characters of the pinned corpus revision.
#:
#: Chosen because it must be **unguessable**. A fact a model can answer from
#: priors ("this package is MIT licensed") tests nothing -- recall only means
#: something when the only way to know is to have read it. A hex string cannot
#: be guessed, and it is genuinely present in this repo, so no scratch fixture
#: has to exist or be maintained.
#:
#: Coupling worth knowing: bumping ``DEFAULT_REVISION`` changes this answer. A
#: test asserts the two agree, so a bump fails loudly here rather than showing
#: up later as a mysterious recall regression.
GOLD_SLICE_FACT = DEFAULT_REVISION[:12]

#: The top-level prompt that plants the fact.
#:
#: Three constraints it has to satisfy at once:
#:
#: 1. It must NOT contain the fact. If it did, the fact would land in a
#:    top-level Action as well, and recall would pass even with nesting
#:    completely broken -- the question would silently stop testing anything.
#: 2. It must force genuine delegation. `test-graph-model` established the
#:    technique: restrict the session to the Task tool so the model has no other
#:    way to accomplish anything.
#: 3. It must contain no literal ``SKILL.md`` path. ``SkillGraphConnector``
#:    scans metadata for anything resembling a resolvable skill path and records
#:    a skill read that never happened -- the open false-positive at #293.
GOLD_SLICE_PROMPT = (
    "Use the Task tool to launch exactly one subagent (subagent_type: Explore). "
    "Have that subagent open context-graph/eval/src/context_graph_eval/convert/longmemeval.py, "
    "find the constant naming the pinned upstream dataset revision, and report its value "
    "verbatim along with a one-line note on what it pins. "
    "Do not open the file yourself -- delegate the entire task to that one subagent."
)

GOLD_SLICE_QUESTION = "Which upstream dataset revision does the eval corpus pin, and what does pinning it protect?"


def gold_slice_goldens() -> list[Golden]:
    """The gold-slice corpus. One question, for now."""
    return [
        Golden(
            input=GOLD_SLICE_QUESTION,
            expected_output=GOLD_SLICE_FACT,
            # What must be recallable. Deliberately just the fact: the
            # explanatory half of the question is there to make it a natural
            # thing to ask, not something the answer key grades.
            context=[f"The pinned upstream dataset revision is {GOLD_SLICE_FACT}."],
            name="goldslice-nested-revision",
            source_file="gold-slice",
            additional_metadata={
                # Tier 2: authored, not adopted, and scored apart from Tier 1
                # so an organizational-recall regression cannot hide behind a
                # personal-memory gain (#303).
                "tier": 2,
                "carrier": "subagent",
                "abstention": False,
                "question_type": "nested-recall",
                # Carried on the Golden itself rather than a joined fixture
                # file, so a question cannot outlive its planting script (#307).
                "session_prompt": GOLD_SLICE_PROMPT,
            },
        )
    ]


def evidence_is_planted(graph: "ActionsGraph", fact: str = GOLD_SLICE_FACT) -> bool:
    """Whether the gold-slice fact is anywhere in the graph at all.

    Checked before a gold-slice question is scored. Its fixture is planted by a
    real harness session, not injected like Tier 1's -- so running the question
    without having driven that session scores a fact the graph never contained,
    and reports a guaranteed zero as though it were a recall failure. A
    fixture that was never laid down is not a measurement.
    """
    rows = graph._db.query(
        "MATCH (a:Action) WHERE a.properties CONTAINS $fact RETURN count(a) AS n",
        {"fact": fact},
    )
    return bool(rows and rows[0]["n"] > 0)


def evidence_is_nested(graph: "ActionsGraph", session_id: str, fact: str) -> bool:
    """Whether ``fact`` appears in an Action owned by an Agent, not the session.

    Checked **before** trusting a recall result. If the model declines to
    delegate, the fact lands in a top-level Action, recall succeeds trivially,
    and the question silently stops testing nesting -- a false pass that looks
    exactly like a real one.
    """
    rows = graph._db.query(
        """
        MATCH (:Session {session_id: $sid})-[:HAS_AGENT]->(:Agent)-[:HAS_ACTION]->(a:Action)
        WHERE a.properties CONTAINS $fact
        RETURN count(a) AS nested
        """,
        {"sid": session_id, "fact": fact},
    )
    return bool(rows and rows[0]["nested"] > 0)
