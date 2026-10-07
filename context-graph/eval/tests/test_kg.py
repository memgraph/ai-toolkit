"""The kg eval's pieces that need no model: building sessions, comparing runs."""

from context_graph_eval.convert.beam import BeamChat
from context_graph_eval.kg import KgRun, compare, documents


def _chat():
    batch = {
        "time_anchor": "March-15-2024",
        "turns": [
            [
                {"role": "user", "content": "I deploy with Kubernetes."},
                {"role": "assistant", "content": "Kubernetes is a solid choice."},
            ],
            [{"role": "user", "content": "I deploy with Kubernetes."}],  # a repeat, dropped as reconcile drops it
        ],
    }
    return BeamChat(size="100K", chat_id=1, batches=[batch], probing_questions={})


def test_a_session_is_built_as_reconcile_hands_it_to_extraction():
    ((session_id, document),) = documents([_chat()])

    assert session_id == "beam-100K-1--b1"
    assert document.user_id == "beam-100K-1"
    assert [(document.text[s.start : s.end], s.role) for s in document.segments] == [
        ("I deploy with Kubernetes.", "user"),
        ("Kubernetes is a solid choice.", "assistant"),
    ]


def _run(mentions, relations, **metrics):
    return KgRun(
        meta={"chats": [1], "windows_requested": 10},
        metrics=metrics,
        probes={"MoMA visit": True},
        fingerprints={"mentions": mentions, "relations": relations},
    )


def test_compare_reports_metrics_side_by_side_and_what_was_kept():
    baseline = _run(["a", "b", "c", "d"], ["x", "y"], coverage=0.9, catch_all_share=0.5)
    run = _run(["a", "b", "c", "e", "f"], ["x"], coverage=0.8, catch_all_share=0.4)

    lines = compare(run, baseline)

    assert any(line.startswith("coverage") and line.split()[1:] == ["0.9", "0.8"] for line in lines)
    assert "mentions kept from baseline: 75% (3/4), new: 2" in lines
    assert "relations kept from baseline: 50% (1/2), new: 0" in lines
    assert "probe MoMA visit: baseline True, this run True" in lines
    assert not any(line.startswith("WARNING") for line in lines)


def test_compare_warns_when_the_sample_differs():
    baseline = _run([], [])
    run = KgRun(meta={"chats": [1, 2], "windows_requested": 10}, metrics={}, fingerprints={})

    assert compare(run, baseline)[-1].startswith("WARNING")
