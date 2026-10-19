"""A stand-in for the `claude` and `codex` CLIs' `plugin` commands, so tests never touch a real agent's config.

Installed on PATH under both names by the ``fake_agents`` fixture, which passes the name it was called by
as the first argument. It keeps its marketplaces and plugins in
the JSON file named by ``FAKE_AGENT_STATE`` and answers in each CLI's own ``--json`` shapes. Setting
``FAKE_AGENT_FAIL`` to ``install`` makes installing fail, as a network error would.
"""

import json
import os
import pathlib
import sys

MARKETPLACE = "context-graph-plugins"


def main() -> int:
    agent, args = sys.argv[1], sys.argv[2:]
    state_path = pathlib.Path(os.environ["FAKE_AGENT_STATE"])
    everyone = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = everyone.setdefault(agent, {"marketplaces": [], "plugins": [], "calls": []})
    state["calls"].append(args)
    code = _answer(agent, args, state)
    state_path.write_text(json.dumps(everyone))
    return code


def _answer(agent: str, args: list[str], state: dict) -> int:
    match args:
        case ["plugin", "list", "--json"]:
            if agent == "claude":
                print(json.dumps([{"id": plugin} for plugin in state["plugins"]]))
            else:
                print(json.dumps({"installed": [{"pluginId": plugin} for plugin in state["plugins"]], "available": []}))
        case ["plugin", "marketplace", "list", "--json"]:
            entries = [{"name": name} for name in state["marketplaces"]]
            print(json.dumps(entries if agent == "claude" else {"marketplaces": entries}))
        case ["plugin", "marketplace", "add", "memgraph/ai-toolkit", "--sparse", _]:
            state["marketplaces"].append(MARKETPLACE)
            print(f"Added marketplace {MARKETPLACE}")
        case ["plugin", "marketplace", "remove", name]:
            state["marketplaces"].remove(name)
        case ["plugin", "install", plugin, "-y"] | ["plugin", "add", plugin]:
            if os.environ.get("FAKE_AGENT_FAIL") == "install":
                print("network unreachable", file=sys.stderr)
                return 1
            if plugin.split("@")[1] not in state["marketplaces"]:
                print(f"no marketplace for {plugin}", file=sys.stderr)
                return 1
            if plugin not in state["plugins"]:
                state["plugins"].append(plugin)
            print(f"Installed {plugin}")
        case ["plugin", "marketplace", "update" | "upgrade", name] if name in state["marketplaces"]:
            print(f"Updated marketplace {name}")
        case ["plugin", "update", plugin] if plugin in state["plugins"]:
            print(f"Updated {plugin}")
        case ["plugin", "uninstall", plugin] | ["plugin", "remove", plugin]:
            state["plugins"].remove(plugin)
        case _:
            print(f"fake {agent}: unexpected {args}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
