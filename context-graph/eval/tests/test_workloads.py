"""Tests for the workload registry: what each workload asks, and where its graph lives."""

from collections import Counter

import pytest
from context_graph_eval.corpus import read_corpus
from context_graph_eval.workloads import WORKLOADS, shards, workload


def test_every_workload_has_its_own_instance_and_port():
    """Two workloads sharing an instance would rebuild over each other's graph."""
    assert len({w.instance for w in WORKLOADS.values()}) == len(WORKLOADS)
    assert len({w.port for w in WORKLOADS.values()}) == len(WORKLOADS)


def test_each_benchmark_has_an_official_and_a_dev_workload():
    kinds = Counter((w.benchmark, w.kind) for w in WORKLOADS.values())

    for benchmark in ("longmemeval-s", "beam-100k"):
        assert kinds[(benchmark, "official")] == 1
        assert kinds[(benchmark, "dev")] == 1


def test_official_workloads_are_uncapped_and_whole():
    lme, beam = workload("lme-s"), workload("beam-100k")

    assert lme.max_sessions_per_question is None
    assert lme.corpus_path is not None and len(read_corpus(lme.corpus_path)) == 500
    assert beam.chats == tuple(range(1, 21))


def test_the_lme_dev_workload_is_a_capped_stratified_subset_of_the_official_questions():
    dev, official = workload("lme-s-dev"), workload("lme-s")
    assert dev.corpus_path is not None and official.corpus_path is not None

    questions = read_corpus(dev.corpus_path)

    assert len(questions) == 50
    assert {q.name for q in questions} <= {q.name for q in read_corpus(official.corpus_path)}
    assert len({(q.additional_metadata or {})["question_type"] for q in questions}) == 6
    assert dev.max_sessions_per_question == 10


def test_the_beam_dev_workload_is_four_of_the_official_chats():
    assert set(workload("beam-100k-dev").chats) < set(workload("beam-100k").chats)
    assert len(workload("beam-100k-dev").chats) == 4


def test_an_unknown_workload_is_refused():
    with pytest.raises(ValueError, match="unknown workload"):
        workload("lme-m")


def test_official_lme_is_asked_in_25_disjoint_shards_of_20_the_same_way_every_time():
    lme = workload("lme-s")
    assert lme.corpus_path is not None
    questions = read_corpus(lme.corpus_path)

    parts = shards(lme, questions)

    assert [len(part) for part in parts] == [20] * 25
    names = [q.name or "" for part in parts for q in part]
    assert sorted(names) == sorted(q.name or "" for q in questions)
    assert [q.name for q in shards(lme, questions)[0]] == [q.name for q in parts[0]]


def test_an_unsharded_workload_is_one_graph():
    dev = workload("lme-s-dev")
    assert dev.corpus_path is not None
    questions = read_corpus(dev.corpus_path)

    assert shards(dev, questions) == [questions]
