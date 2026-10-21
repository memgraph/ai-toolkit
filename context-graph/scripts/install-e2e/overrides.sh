#!/usr/bin/env bash
# Write a uv overrides file that resolves every Context Graph package to this checkout, and print its path.
#
# install.sh and setup's own package step install from PyPI; with UV_OVERRIDE set to this file, both
# install the code under review instead, so the install matrix tests a PR rather than the last release.
# An override replaces the whole requirement, so each line carries the extras install.sh asks for.
#
# Usage: export UV_OVERRIDE="$(./context-graph/scripts/install-e2e/overrides.sh "$RUNNER_TEMP")"

set -euo pipefail

root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="${1:?usage: overrides.sh <directory>}/context-graph-overrides.txt"
{
  echo "memgraph-context-graph[mcp] @ file://$root/context-graph/memgraph-context-graph"
  echo "skills-graph[memgraph-context-graph] @ file://$root/context-graph/skills-graph"
  echo "actions-graph[memgraph-context-graph] @ file://$root/context-graph/actions-graph"
  echo "sessions-graph[memgraph-context-graph] @ file://$root/context-graph/sessions-graph"
  echo "context-graph-viewer @ file://$root/context-graph/viewer"
  echo "memgraph-toolbox @ file://$root/memgraph-toolbox"
  echo "hygm @ file://$root/hygm"
  echo "unstructured2graph @ file://$root/unstructured2graph"
} >"$out"
echo "$out"
