#!/usr/bin/env bash
# Register a Claude Code marketplace named like the Memgraph marketplace, its Context Graph plugin served from
# a copy of this checkout.
#
# The published marketplace fetches the plugin from GitHub's main branch, so a PR's plugin changes would
# go untested. Setup installs from a marketplace already registered under the Memgraph marketplace's name,
# so registering this one first makes setup install the plugin under review. A marketplace already
# registered under that name is removed first, the published one included, since uninstall keeps it.
#
# Usage: ./context-graph/scripts/install-e2e/local-marketplace.sh <directory>

set -euo pipefail

root="$(cd "$(dirname "$0")/../../.." && pwd)"
market="${1:?usage: local-marketplace.sh <directory>}/context-graph-marketplace"
rm -rf "$market"
mkdir -p "$market/.claude-plugin"
cp -R "$root/context-graph/plugins/memgraph-context-graph-claude" "$market/plugin"
python3 - "$root/.claude-plugin/marketplace.json" "$market/.claude-plugin/marketplace.json" <<'PY'
import json
import sys

marketplace = json.load(open(sys.argv[1]))
marketplace["name"] = "memgraph"
marketplace["plugins"][0]["source"] = "./plugin"
json.dump(marketplace, open(sys.argv[2], "w"), indent=2)
PY
claude plugin marketplace remove memgraph >/dev/null 2>&1 || true
claude plugin marketplace add "$market"
