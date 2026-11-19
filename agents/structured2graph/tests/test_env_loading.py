"""The CLI defaults come from a .env file in the working directory."""

from structured2graph.main import parse_cli_args
from structured2graph.utils.environment import load_environment


def test_cli_defaults_read_dotenv_from_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SQL2MG_MODE=incremental\nSQL2MG_STRATEGY=llm\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SQL2MG_MODE", raising=False)
    monkeypatch.delenv("SQL2MG_STRATEGY", raising=False)

    load_environment()
    args = parse_cli_args([])

    assert args.mode == "incremental"
    assert args.strategy == "llm"
