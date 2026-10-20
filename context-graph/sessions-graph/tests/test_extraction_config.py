"""ExtractionConfig: the config file's [extraction] as GLiNER2Backend arguments."""

import pytest
from sessions_graph.extraction import ExtractionConfig
from sessions_graph.recall import QUERY_INSTRUCTION, query_instruction


def test_unset_keeps_every_backend_default():
    assert ExtractionConfig().backend_arguments() == {}


def test_config_file_strings_become_backend_arguments():
    config = ExtractionConfig.from_mapping(
        {"model": "my/checkpoint", "entity_threshold": "0.4", "chunk_size": "256", "chunk_overlap": "0", "other": "x"}
    )

    assert config.backend_arguments() == {
        "model_name": "my/checkpoint",
        "entity_confidence_threshold": 0.4,
        "chunk_size": 256,
        "chunk_overlap": 0,
    }


@pytest.mark.parametrize(
    "values",
    [{"entity_threshold": "1.5"}, {"relation_threshold": "-0.1"}, {"chunk_size": "0"}, {"chunk_overlap": "-1"}],
)
def test_out_of_range_values_are_rejected(values):
    with pytest.raises(ValueError, match="extraction"):
        ExtractionConfig.from_mapping(values)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("BAAI/bge-small-en-v1.5", QUERY_INSTRUCTION),
        ("BAAI/bge-large-en-v1.5", QUERY_INSTRUCTION),
        ("BAAI/bge-m3", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_only_bge_english_models_get_its_query_instruction(model, expected):
    assert query_instruction(model) == expected
