# GQLAlchemy: first baseline comparison

The ordinary-context candidate passed all three final automated checks. Graph
recall worked in a fresh thread, but this case does not demonstrate a practical
advantage over persistent code and tests. Do not scale to hundreds of issues or
claim more correct issue resolutions from this result.

## Scope and fairness

One pinned issue (#375) was implemented, followed by a genuinely fresh-thread
assessment of batching it with #246 and #319. Each arm used GPT-6 Luna, medium
reasoning, 120-second session limits, 14 counted tool starts, and the same frozen
resource data. Native Codex memory was disabled. Code and tests persisted in all
arms; investigation notes were allowed in two, and graph recall was available in
one. No additional summaries or ontology derivation ran.

To conserve weekly allowance, the earlier transport-valid graph pair was reused
and four new sessions completed the missing baselines. Matching model settings,
resource/evaluator hashes, fresh thread IDs, and prompts normalized for workspace
paths and intended memory instructions were checked programmatically. This was
not a randomized, counterbalanced concurrent experiment. Earlier infrastructure
failures remain in the cumulative ledger and are excluded by transport failure,
not by candidate quality. Poor candidates were not rerun to obtain nicer results.

## Quality results

| Check | Ordinary context | Notes allowed | Graph recall |
|---|---|---|---|
| Independent multigraph acceptance | Pass | Pass | Pass |
| Default edge-property compatibility | Pass | Pass | **Fail** |
| Translator suite | 18 pass | **17 pass, 1 fail** | 18 pass |
| All automated quality gates | **Pass** | Fail | Fail |
| Human blinded review | Pending | Pending | Pending |
| Accepted upstream resolutions | 0 | 0 | 0 |

The ordinary implementation preserves the prior simple-graph property replacement
path while adding opt-in multigraph handling. The notes-allowed submission's new
test confuses stored `id` properties with internal node IDs. The graph submission
uses `DiGraph.add_edges_from`, retaining properties from multiple relationships in
the default collapsed edge; the same compatibility probe passes at baseline.
These first-session differences cannot establish a memory effect: every arm
started without prior learned experience and the sample is only one candidate.

The default-property probe was introduced after the earlier graph submission and
applied identically to every final candidate without revealing its specifics to
the new agents. It is a post-submission robustness check, not part of the original
frozen issue-specific acceptance test. No candidate was edited by the evaluator.

## Reuse and planning results

All three fresh sessions recommended keeping the broad API/mapping and filtered
retrieval changes separate from #375. Each compared two approaches, discussed
compatibility and performed a local validation. Ordinary context recovered earlier
work from the changed code/tests and reran a regression. Graph recall recovered
the prior collapse experiment with matching source-session provenance and tested
that the query builder emits both parallel edges.

**No arm wrote `INVESTIGATION.md`.** The notes-allowed arm reported that it found
none. Consequently this run compares graph recall with repository artifacts,
but does not test active file-note recall. The graph was useful as an accessible
record of experiments; the assessments do not show that this changed the chosen
batch or enabled a decision the ordinary arm could not make. Revalidating a claim
is not automatically a wasted repeated investigation.

Graph extraction completed before the fresh graph session with no summaries and
five nonconformant relations reported. Recall success does not prove extracted
fact accuracy or establish that structured graph facts, rather than stored session
turns, caused the useful retrieval.

## Resource results

| Metric, two sessions per arm | Ordinary context | Notes allowed | Graph recall |
|---|---:|---:|---:|
| Implementation session seconds | 57.67 | 44.10 | 57.78 |
| Fresh assessment seconds | 35.82 | 37.03 | 28.21 |
| Total agent seconds | 93.49 | 81.13 | 85.99 |
| Counted tool starts | 17 | 13 | 16 |
| Reported input tokens | 729,434 | 579,790 | 701,950 |
| Cached input tokens | 638,464 | 503,296 | 631,552 |
| Noncached input tokens | 90,970 | 76,494 | 70,398 |
| Output tokens | 6,718 | 4,999 | 5,183 |
| CI runs | 0 | 0 | 0 |
| Dollar cost | Unknown | Unknown | Unknown |

The graph fresh session was 7.61 seconds faster than ordinary context. However,
capture-to-readiness file timestamps show approximately **103.21 seconds** of
serial graph processing between sessions. Adding that interval to agent time
gives about **189.20 seconds**, excluding other cold setup and common observation
capture overhead. It is an approximate interval, not a profiler measurement.
The small downstream time saving did not offset memory processing in this case.
Timings are descriptive and cannot establish a statistically reliable speedup.

Reported input tokens accumulate across requests and include caching; they are
not the size of one context window. Subscription charges cannot be calculated
from API token prices. No cost-per-accepted-resolution ratio is defined when zero
resolutions have been accepted. Identical resource caching was supplied to every
arm, so no GitHub-call saving can be attributed to the learned-memory treatment.

The baseline completion used four new bounded sessions; all experiments now total
18. Account-level weekly usage was 20% before and 21% afterward; five-hour usage
was 14% before initial inspection and 18% afterward. These readings include other
work and are not precise experiment attribution. Each new launch verified ordinary
usage availability and headroom, with stop gates at 25% weekly and 50% short-window
usage. No paid credit purchase or allowance reset was requested. Launches are
disabled again, copied authentication removed and new containers stopped.

## Practical judgment

For this small repository task, code and tests already preserve most information
needed in the next session. Context Graph provided real retrieval, but added
processing and integrity concerns without a demonstrated improvement in decisions
or accepted fixes. File notes and ordinary repository artifacts remain serious
baselines; the evidence does not justify replacing them or claiming graph memory
is always more efficient.

A further evaluation only makes sense if it tests information that repository
artifacts do not retain: failed hypotheses, commands and environment discoveries
across several intervening tasks, ideally with independently verified downstream
decisions. It must compare against actively maintained file notes and charge
capture, processing and retrieval overhead. That is a hypothesis to test, not a
benefit demonstrated here. More repetitions would be needed to generalize beyond
this case; none were run to avoid consuming the weekly allowance.

## Evidence

The comparison summary, identical-check outputs, candidate diffs and assessment
reports are under `runs/20261009T131006Z/comparison/`. Source graph sessions are
under `runs/20261009T124943Z/`. `summary.json` records prompt/hash checks, resource
measurements, candidate failures and methodological limitations. Raw sessions and
review packets remain local and are excluded from Git. Human blinded acceptance
has not been completed; the ordinary candidate is test-qualified, not a claimed
merged fix. Filesystem read isolation was cooperative throughout the compared
runs, not enforced against adversarial agents.
