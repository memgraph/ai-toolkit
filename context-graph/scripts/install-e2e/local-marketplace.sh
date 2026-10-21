#!/usr/bin/env bash
# Register a copy of this checkout's Claude Code marketplace, its plugin served from the copy.
#
# The published marketplace fetches the plugin from GitHub's main branch, so a PR's plugin changes would
# go untested. Setup installs from a marketplace already registered under our name, so registering
# this one first makes setup install the plugin under review. Run it again after uninstall, which
# removes the marketplace.
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
marketplace["plugins"][0]["source"] = "./plugin"
json.dump(marketplace, open(sys.argv[2], "w"), indent=2)
PY
claude plugin marketplace add "$market"
