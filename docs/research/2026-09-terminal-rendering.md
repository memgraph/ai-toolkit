# Terminal rendering: which TUI framework, and how to show a graph in a terminal?

Research findings for [#382](https://github.com/memgraph/ai-toolkit/issues/382), part of [Map: Context-graph visualisation & observability](https://github.com/memgraph/ai-toolkit/issues/374).

Investigated 2026-09-29. Claims are cited to primary sources: official docs, source repos, release pages, PyPI metadata. Install sizes, dependency trees, and a few behaviours were **measured locally**. Those are marked *(measured)*, and the throwaway scripts that produced them are described in [Reproduction](#reproduction). Anything I could not verify is marked **unverified**.

Constraints already settled by [#380](https://github.com/memgraph/ai-toolkit/issues/380): the viewer is a read-only, standalone terminal app in its own workspace package. It's installed with `uv tool install agent-context-graph --with <viewer>` and opens on one global dashboard. It polls Memgraph every 2–5 s with no push channel, and it has **no Memgraph Lab fallback**, so the knowledge graph view has to work in the terminal alone.

## TL;DR recommendation

| Decision | Recommendation | Main evidence |
|---|---|---|
| **Framework** | **Textual** (which brings Rich with it). Poll with `set_interval`, and run each Memgraph read in an `exclusive` worker. | It's the only candidate with a table, tree, tabs, sparkline, log **and** headless snapshot testing all in first-party code. It adds ~11 MB to an ~27 MB `agent-context-graph` install *(measured)*. It uses DEC 2026 synchronized output against flicker. MIT licence. Python ≥3.9, and it runs on Python 3.10 *(measured)*. |
| **Knowledge graph view** | **Focused-entity neighbour tree plus a provenance pane**, navigated by walking the graph with a breadcrumb trail. **No node-link drawing.** | The tools that read well at terminal resolution (`cargo tree`, `pipdeptree`, k9s xray, `networkx.write_network_text`) all flatten the graph into a tree and mark repeated nodes. The Python node-link renderers drop edge labels and draw misleading lines at 11 nodes *(measured)*. |
| **Session anatomy** | **Tree-table with Gantt bars**: nested actions and subagents on the left, a time-scaled bar per row on the right, and a details pane. Derived Episode/Chunks/entities go in a section below. | otel-tui's span timeline is this layout, working in a shipped terminal tool ([screenshot](https://github.com/ymtdzzz/otel-tui/blob/main/docs/spans.png)). Textual has no tree-table widget, so we'd build it: a `Tree` or `DataTable` row with a bar drawn in Rich. |
| **Navigation model** | Dashboard panes you jump between with number keys, `enter` to drill down, `esc` to go back up a breadcrumb stack, `/` to filter, `?` for help. | This is the common core of k9s, lazygit, htop and Harlequin (see [Q4](#q4-prior-art-for-observability-tuis)). It maps directly onto Textual's screen stack (`push_screen`/`pop_screen`). |

The one real risk is **Textual's maintenance runway**. Textualize the company wound down in May 2025, and the maintainer said he will keep maintaining Textual and Rich ([announcement](https://textual.textualize.io/blog/2025/05/07/the-future-of-textualize/)). Releases kept coming through 2026 (v8.2.8 on 2026-06-30), but commits have slowed sharply since July 2026 (see [Maintenance](#maintenance-activity)). Mitigations: pin the major version, keep queries apart from renderers (already required by #380/#381), and note that urwid is the only other maintained candidate with a comparable widget set.

## Q1. Frameworks

### Comparison

Sizes are the `site-packages` footprint of a fresh Python 3.12 venv containing only that package and its dependencies *(measured)*. For context, `agent-context-graph` 0.2.0 on its own is **~27.0 MB**, mostly `neo4j` and `numpy`. Adding Textual to that same venv brings it to **~38.3 MB**, so Textual costs about **+11.3 MB** *(measured)*.

| | Textual 8.2.8 | Rich 15.0.0 (alone) | prompt_toolkit 3.0.53 | urwid 4.1.7 | PyTermGUI 7.8.1 | asciimatics 1.15.0 | npyscreen 5.0.4 |
|---|---|---|---|---|---|---|---|
| Installed size *(measured)* | 11.3 MB | 7.1 MB | 4.8 MB | 4.8 MB | 3.9 MB | 25.1 MB (pulls Pillow) | 0.5 MB |
| Runtime deps *(measured)* | rich, pygments, markdown-it-py (+linkify, mdit-py-plugins), platformdirs, typing-extensions | pygments, markdown-it-py, mdurl | wcwidth | wcwidth, typing-extensions | wcwidth, typing-extensions | pillow, pyfiglet, wcwidth | none |
| `requires_python` ([PyPI](https://pypi.org/)) | `>=3.9,<4.0` | `>=3.9.0` | `>=3.10` | `>=3.9.0` | `>=3.8` | `>=3.8` | `>=3.7` |
| Licence (PyPI/GitHub) | MIT | MIT | BSD-3-Clause | **LGPL-2.1** | MIT | Apache-2.0 | BSD-3-Clause (GitHub: NOASSERTION) |
| Latest release (PyPI) | 2026-06-30 | 2026-04-12 | 2026-07-26 | 2026-09-23 | 2026-09-10 (**final**) | **2023-10-25** | 2026-07-04 |
| Periodic refresh | `set_interval` timer + async workers | `Live(refresh_per_second=…)` | `Application(refresh_interval=…)` | `MainLoop.set_alarm_in` | unverified | unverified | unverified |
| Anti-flicker | Synchronized output (DEC 2026) + compositor partial updates | unverified (no mention in `Live` docs) | `min_redraw_interval` throttle | unverified | unverified | unverified | unverified |
| Keyboard input / focus | Yes | **No**. Rich renders output; its README points to Textual for UIs | Yes | Yes | Yes | Yes | Yes |
| Table | `DataTable` (keyed rows, `update_cell`, sort) | `Table` (static) | **none** | **none** | unverified | unverified | grid |
| Tree | `Tree` (expand events, rich labels) | `Tree` (static) | **none** | `TreeWidget`/`TreeListBox` | unverified | unverified | tree |
| Tabs | `Tabs`, `TabbedContent`, `ContentSwitcher` | no | no | no | unverified | unverified | no |
| Sparkline / chart | `Sparkline` (+ `textual-plotext`) | no | `ProgressBar` only | `BarGraph` | unverified | unverified | no |
| Log view | `Log`, `RichLog` | `Console.log` | `TextArea` | `ListBox` | unverified | unverified | pager |
| Headless tests | `run_test()` + `Pilot`, SVG snapshots via `pytest-textual-snapshot` | `Console(record=True)` export (unverified for our use) | `create_pipe_input` + `DummyOutput`; docs advise *against* testing rendered output | unverified | unverified | unverified | unverified |
| Windows | Yes, "Windows Terminal runs Textual apps beautifully" | Yes | Yes (`win32`, `windows10` outputs) | Windows 10+ only, with some features missing | unverified | Yes (Win 7–10) | unverified |

### Textual

- **Install and platforms.** Textual needs Python 3.9 or later and runs on "Linux, macOS, Windows and probably any OS where Python also runs". Its docs say "the new Windows Terminal runs Textual apps beautifully" and that macOS Terminal.app "is limited to 256 colors" ([Getting started](https://textual.textualize.io/getting_started/)). The Textual FAQ adds that Terminal.app may not render TUIs "very well, particularly when it comes to box characters" ([FAQ](https://textual.textualize.io/FAQ/)). The `textual/drivers/` package ships `linux_driver.py`, `windows_driver.py` and `headless_driver.py` *(measured, v8.2.8 wheel)*. I installed it on CPython 3.10.16 and ran a headless app there *(measured)*.
- **Size.** Pygments is 5.1 MB of Textual's 11.3 MB, textual itself is 3.3 MB, and rich is 1.4 MB *(measured)*. The `syntax` extra (tree-sitter grammars) is optional, and we don't need it ([Textual METADATA](https://pypi.org/project/textual/)).
- **Periodic refresh.** `set_interval(interval, callback=None, *, name=None, repeat=0, pause=False)` means "call a function at periodic intervals" ([MessagePump API](https://textual.textualize.io/api/message_pump/#textual.message_pump.MessagePump.set_interval)). In the source, the timer **awaits the callback inline** and defaults to `skip=True`: "enable skipping of scheduled events that couldn't be sent in time" ([`timer.py`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/timer.py)). So a slow query can't make ticks pile up. Two consequences for design, both seen in a probe *(measured)*:
  1. An exception raised in a timer callback goes to `app._handle_exception`. In my probe, a single `CellDoesNotExist` **ended the app**. A transient Memgraph error in a bare timer callback would crash the viewer, so the poll must catch its own errors.
  2. A blocking (sync) driver call inside the callback would freeze input. The [Workers guide](https://textual.textualize.io/guide/workers/) covers this: `@work(exclusive=True)` makes each call start a worker, and "the `exclusive` flag tells Textual to cancel all previous workers before starting the new one". Use `thread=True` for blocking APIs, and in that case "avoid calling methods on your UI directly", using `call_from_thread()` instead. `memgraph-toolbox` already has an `AsyncMemgraph` alongside the sync `Memgraph` ([`memgraph_toolbox/api/memgraph.py`](https://github.com/memgraph/ai-toolkit/blob/main/memgraph-toolbox/src/memgraph_toolbox/api/memgraph.py)), so async workers are an option.
- **Flicker.** Textual wraps each update in DEC mode 2026 synchronized output (`SYNC_START = "\x1b[?2026h"`). It only does this after the terminal reports support, and not in inline mode (`_on_terminal_supports_synchronized_output` / `_begin_update` in [`app.py`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/app.py); [`_ansi_sequences.py`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/_ansi_sequences.py)). Its compositor also does partial updates: "if you click a button and it changes color, the compositor can update just the region occupied by the button" ([Algorithms for high performance terminal apps](https://textual.textualize.io/blog/2024/12/12/algorithms-for-high-performance-terminal-apps/)). `DataTable.update_cell()` changes one cell in place, with rows keyed stably across sorts ([DataTable](https://textual.textualize.io/widgets/data_table/)). A poll can therefore patch values instead of rebuilding tables. I have **not** checked flicker by eye in each terminal (Terminal.app, iTerm2, Windows Terminal). #383 should do that.
- **Widgets we need** ([widget gallery](https://textual.textualize.io/widget_gallery/)): `DataTable`, `Tree`, `Tabs`/`TabbedContent`/`ContentSwitcher`, `Sparkline` (since 0.27.0, with reactive `data` for live updates ([Sparkline](https://textual.textualize.io/widgets/sparkline/))), `Log`/`RichLog` (`max_lines`, `auto_scroll` ([RichLog](https://textual.textualize.io/widgets/rich_log/))), `Collapsible`, `Header`/`Footer` and `Toast`. `Tree` emits `NodeExpanded`/`NodeCollapsed`/`NodeSelected`, takes Rich text labels, and carries typed data per node ([Tree](https://textual.textualize.io/widgets/tree/)). That is enough to fetch neighbours lazily when a node is expanded. **There's no tree-table or Gantt widget**, and a search of the Textual issue tracker for "tree table" found nothing relevant *(measured, `gh search issues`)*. Charts come from `textual-plotext`, which pins `plotext>=5.2.8,<6.0.0` *(measured, METADATA)* while plotext is now on 6.1.0, and it last released in 2024-11 ([PyPI](https://pypi.org/project/textual-plotext/)). Avoid it unless we need real charts.
- **Navigation primitives.** `push_screen` "puts a screen on top of the stack and makes that screen active", `pop_screen` removes it, and *modes* keep "multiple independent screen stacks" ([Screens](https://textual.textualize.io/guide/screens/)). A breadcrumb trail is just the screen stack.
- **Testing.** `run_test()` "will run the app in headless mode". It returns a `Pilot` for `press()`/`click()`/`pause()`. `pytest-textual-snapshot` records SVG screenshots and compares them on later runs, with `--snapshot-update` to accept changes ([Testing guide](https://textual.textualize.io/guide/testing/)). I drove a probe headless (a timer patching a `DataTable` cell, a `Tree`, key presses) and exported a screenshot with `export_screenshot()` *(measured)*. That fits the repo's "real Memgraph, not mocks" policy: a test can seed the test Memgraph, run the app headless, and snapshot what it shows.
- **Licence:** MIT ([PyPI](https://pypi.org/project/textual/)).
- **Prior art on Textual:** Harlequin, a SQL IDE, pins `textual==8.2.8` and needs Python ≥3.10 ([pyproject](https://github.com/tconbeer/harlequin/blob/main/pyproject.toml)). It also uses `textual-fastdatatable`, "a performance-focused reimplementation of Textual's DataTable widget, with a pluggable data storage backend" ([repo](https://github.com/tconbeer/textual-fastdatatable)). That's worth knowing if a table grows large.

### Rich (alone)

Rich renders output. It has `Table`, `Tree`, `Progress` ("flicker-free"), `Live`, `Columns` and more. Its README points elsewhere for interactive UIs: "See also Rich's sister project, Textual, which you can use to build sophisticated User Interfaces in the terminal" ([README](https://github.com/Textualize/rich)). `Live` refreshes 4 times a second by default, takes `auto_refresh=False`, and has `screen=True` for the alternate screen ([Live docs](https://rich.readthedocs.io/en/stable/live.html)). The docs don't cover keyboard input. **Verdict:** Rich alone can't drill down or navigate, which views 2 and 3 need. It comes in with Textual anyway and is the right tool for drawing custom cells such as Gantt bars.

### prompt_toolkit

It can "create complex full screen terminal applications" out of containers and controls ([Full screen apps](https://python-prompt-toolkit.readthedocs.io/en/master/pages/full_screen_apps.html)). `Application(refresh_interval=…)` "automatically invalidate[s] the UI every so many seconds", and `min_redraw_interval` throttles redraws ([reference](https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html)). It's the lightest option with input handling (4.8 MB, wcwidth only *(measured)*). But `prompt_toolkit.widgets` has no table, tree, tabs or sparkline, only `TextArea`, `Button`, `Frame`, `Dialog`, `RadioList`, `Checkbox`, `ProgressBar`, toolbars and menus *(measured, source)*. Its testing docs say "we don't want to test the bytes that are written to sys.stdout" ([Unit testing](https://python-prompt-toolkit.readthedocs.io/en/master/pages/advanced_topics/unit_testing.html)), so there's no snapshot story. **Verdict:** we'd build every widget ourselves to save ~6.5 MB.

### urwid

Mature and very active: 100+ commits in the last 90 days *(measured, GitHub API)*, released 2026-09-23 ([PyPI](https://pypi.org/project/urwid/)). It supports CPython 3.9–3.15 and has several event loops (asyncio, trio, Twisted, …) ([README](https://github.com/urwid/urwid)). `set_alarm_in` callbacks automatically trigger `draw_screen()` ([Main loop manual](https://urwid.org/manual/mainloop.html)). It has `TreeWidget`/`TreeListBox` ([`treetools.py`](https://github.com/urwid/urwid/blob/master/urwid/widget/treetools.py)) and `BarGraph`, but **no table or tabs widget** *(measured, source)*. "Windows support is limited to the Windows 10+" and some features are missing there ([README](https://github.com/urwid/urwid)). The licence is **LGPL-2.1**. That's fine as a dynamically imported dependency, but more friction than MIT. **Verdict:** the credible fallback if Textual stalls.

### Others

- **PyTermGUI**: **archived on 2026-09-10**. "PyTermGUI has reached its final release and is no longer under active development" ([repo](https://github.com/bczsalba/pytermgui)). Excluded.
- **asciimatics**: last release 2023-10-25 ([PyPI](https://pypi.org/project/asciimatics/)), and it pulls in Pillow (25 MB total *(measured)*). Excluded.
- **npyscreen**: tiny (0.5 MB) and released again in 2026 ([PyPI](https://pypi.org/project/npyscreen/)), but curses-based, with no async story found (unverified) and a 501-star community *(measured, GitHub API)*. Excluded.

### Maintenance activity

Measured with the GitHub API on 2026-09-29 (commits on the default branch, last 90 days):

| Repo | Last push | Commits in last 90 d | Stars |
|---|---|---|---|
| [Textualize/textual](https://github.com/Textualize/textual) | 2026-07-11 | 2 | 37.4k |
| [Textualize/rich](https://github.com/Textualize/rich) | 2026-06-23 | 0 | 57.4k |
| [prompt-toolkit/python-prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) | 2026-07-26 | 5 | 10.6k |
| [urwid/urwid](https://github.com/urwid/urwid) | 2026-09-29 | 100+ | 3.0k |
| [tconbeer/harlequin](https://github.com/tconbeer/harlequin) (Textual app) | 2026-09-28 | 100+ | 6.4k |

Textual commits by month in 2026: 9 (Apr), 70 (May), 19 (Jun), 2 (Jul), none since *(measured)*. Releases: v8.2.1 through v8.2.8 between 2026-03-29 and 2026-06-30 ([releases](https://github.com/Textualize/textual/releases)). The maintainer's stated position is "Textual will live on as an Open Source project … I will be maintaining Textual and Rich as I have always done", and "Textual is mature and battle-tested" ([The future of Textualize](https://textual.textualize.io/blog/2025/05/07/the-future-of-textualize/)). Read it as mature and slowing down, not abandoned. A downstream project (Harlequin) pinning the latest version and shipping weekly is some evidence that people depend on it successfully.

## Q2. Showing a graph in a terminal

### Node-link drawing: what's available, and how it reads

| Tool | What it is | Findings |
|---|---|---|
| [Graph::Easy](https://metacpan.org/pod/Graph::Easy) (Perl) | Grid "Manhattan" layout to ASCII / box-art | Its own docs admit "in complex graphs, non-optimal layout part might appear", and that a second-stage optimizer "is not yet implemented". It's also Perl, so it's not something we can install. |
| [grandalf](https://pypi.org/project/grandalf/) | Sugiyama/force layout, **no rendering**. "It's up to you to actually draw things." | GPLv2 / EPLv1. Last release 2023-01-10. This is what [DVC's `dagascii.py`](https://github.com/iterative/dvc/blob/main/dvc/dagascii.py) builds on, and LangChain's `draw_ascii` is "adapted from" DVC's code ([`graph_ascii.py`](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/graph_ascii.py)). Both are used for small **DAGs** (pipelines, chains), not dense, labelled, cyclic knowledge graphs. |
| [phart](https://github.com/scottvr/phart) | Pure-Python ASCII renderer on NetworkX | Current 2.1.0 needs **Python ≥3.14** ([PyPI](https://pypi.org/project/phart/)), so it's ruled out for ≥3.10. I rendered an 11-node, 11-edge entity sample with 1.1.4 *(measured)*. It came out ~150 columns wide, with **no relation labels**. A shared horizontal rail made unrelated entities look connected (it drew `[Bolt]←──[Cypher]←──[Docker]──…` although none of those edges exist). |
| [otel-tui topology](https://github.com/ymtdzzz/otel-tui/blob/main/docs/topology.png) (Go) | Boxes and orthogonal edges with count labels | Readable for ~8 services with numeric edge labels. It's labelled "(beta)" in its own UI, and several edges share vertical segments, so which label belongs to which edge gets unclear. |

**Conclusion (evidence-backed opinion):** terminal node-link drawing only works for small, mostly acyclic graphs with short or numeric edge labels. Our case has typed relation labels (`USES`, `WORKS_ON`, …), cycles, and hub entities with many neighbours. The measured phart output shows the failure mode directly: labels dropped and adjacency invented by line routing. Leave it out.

### Flattened representations that read well

- **Dependency trees with de-dup markers.** In `cargo tree`, "packages marked with `(*)` are 'de-duplicated'" because their subtree was already printed, `-i/--invert` shows reverse dependencies, and `--depth` limits the tree ([cargo tree docs](https://doc.rust-lang.org/cargo/commands/cargo-tree.html)). `pipdeptree` has a tree view, `--reverse --packages`, and cycle detection ([repo](https://github.com/tox-dev/pipdeptree)). Both take a graph and render a **tree from a chosen root**. Repeated nodes become a marker instead of a new branch, and direction flips with a flag.
- **`networkx.write_network_text`**: "a depth-first traversal of the graph", writing "a line for each unique node". "Non-tree edges are written to the right of each node, and connection to a non-tree edge is indicated with an ellipsis." It takes `sources` and `max_depth` ([docs](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.text.write_network_text.html)). On the same 11-node sample *(measured)*:

  ```
  ╙── Memgraph ╾ LightRAG, GLiNER2, Ante
      ├─╼ Cypher ╾ actions-graph
      ├─╼ sessions-graph
      │   └─╼  ...
      ├─╼ actions-graph
      │   └─╼  ...
      ├─╼ Docker
      ├─╼ Bolt
      └─╼ text search
  ```
  Here incoming edges are inlined after `╾` and already-seen nodes collapse to `...`. It's already better than any node-link output, but it **still has no relation types**, and we'd need those. The idea transfers; the library doesn't need to (it would add networkx, 8.4 MB *(measured)*).
- **k9s xray**: `:xray RESOURCE [NAMESPACE]` "Launch XRay view" for po/svc/dp/rs/sts/ds ([k9s README](https://github.com/derailed/k9s)). It shows ownership relationships as a collapsible tree (Deployment → ReplicaSet → Pod → Container). The per-kind tree builders live in [`internal/xray`](https://github.com/derailed/k9s/tree/master/internal/xray) (`dp.go`, `rs.go`, `pod.go`, `container.go`, `tree_node.go`). Beyond the README table and source, it's lightly documented: an issue asked for [xray documentation](https://github.com/derailed/k9s/issues/2737), now closed.
- **`git log --graph`**: "Draw a text-based graphical representation of the commit history on the left hand side of the output. This may cause extra lines to be printed in between commits", and it implies `--topo-order` ([git-log docs](https://git-scm.com/docs/git-log)). This lane-based layout works because a commit graph is a DAG with a total order and low branching. A knowledge graph has neither.
- **Terminal graph-DB clients render tables, not graphs.** `mgconsole`'s `--output_format` "can be csv, tabular or cypherl" ([mgconsole `main.cpp`](https://github.com/memgraph/mgconsole/blob/master/src/main.cpp)). I found **no established terminal graph-database browser**. GitHub repo searches for "neo4j tui", "graph tui" (Python) and "knowledge graph terminal" returned nothing relevant (≤3 stars) *(measured, 2026-09-29)*, and the graph browsers on the web are GUIs (e.g. [Neo4j Browser](https://github.com/neo4j/neo4j-browser)). Treat this as "none found", not proven absence.

### Recommendation for "an entity's neighbourhood with provenance back to the source session"

Use a **focused-entity ego view as a lazily expanding tree, grouped by relation type and direction**, next to a **provenance pane** for the highlighted item. Walk the graph by *refocusing* (`enter` on a neighbour makes it the new root) and keep the path in a **breadcrumb** (`esc`/`backspace` pops it).

Why this shape, from the evidence above:

1. **Tree from a chosen root** is how every well-read tool handles a graph: cargo tree, pipdeptree, k9s xray, `write_network_text`.
2. **Group by relation type and direction** (`→ USES (3)`, `← WORKS_ON (1)`), which puts back the edge labels that the tree tools drop. Direction is `cargo tree -i` / `pipdeptree --reverse` folded into one view instead of a flag.
3. **De-dup marker** (`↺` or `(*)`, as in cargo tree) when a neighbour is already on the breadcrumb path. That keeps cycles finite without hiding them, as the `...` does in `write_network_text`.
4. **Lazy expansion on `NodeExpanded`** fits Textual's `Tree` events and keeps each poll to a one-hop query. Hubs are capped (top-N by mention count, then "+ 37 more…").
5. **Provenance as its own pane, not more tree levels.** The provenance chain (`entity -[:MENTIONED_IN]-> Chunk <-[:HAS_CHUNK]- Action|Memory … Session`) comes from [`unstructured2graph/CONTEXT.md`](https://github.com/memgraph/ai-toolkit/blob/main/unstructured2graph/CONTEXT.md) and [`sessions-graph/CONTEXT.md`](https://github.com/memgraph/ai-toolkit/blob/main/context-graph/sessions-graph/CONTEXT.md). That same CONTEXT.md notes that when a session splits into several Chunks, "provenance [is] then session-level, not exact per source". So the pane should show **sessions and episodes first**, with chunk text as evidence, and it shouldn't claim per-action precision it doesn't have.
6. Jumping from a provenance row into **Session anatomy** is the same drill-down as `enter` in k9s, which ties the knowledge graph view back to the timeline view.

## Q3. Timelines in a terminal (session anatomy)

- **otel-tui (Go, tview/tcell)** is the closest prior art. Its "Trace Timeline" is a **tree-table**: an indented, foldable span-name column with durations, a **time-scaled bar per row** against a `0 · 1ms · 2ms …` axis, a **Details** tree pane on the right, and a logs pane below filtered by span. Footer hints: "Enter: Toggle folding the child spans", "Right/Left: widen/narrow span name column" ([spans screenshot](https://github.com/ymtdzzz/otel-tui/blob/main/docs/spans.png), [README](https://github.com/ymtdzzz/otel-tui); `rivo/tview` and `gdamore/tcell` listed in [go.mod](https://github.com/ymtdzzz/otel-tui/blob/main/go.mod)). The repo is active (pushed 2026-09-27, Apache-2.0 *(measured)*). Nested subagents map directly onto nested spans.
- **tokio-console** is a "top-for-tasks": a task list, task details with a poll-time histogram, and a resources view ([README](https://github.com/tokio-rs/console)). It's list → detail rather than a waterfall. It fits the capture health view better than the session view.
- **htop / btop tree mode**: htop's F5/`t` shows processes by parent/child ([htop(1)](https://www.man7.org/linux/man-pages/man1/htop.1.html)), and btop has a process tree too ([README](https://github.com/aristocratos/btop)). These show nesting as indentation inside a sortable table with metric columns. That's the tree-table half without the time axis.
- **In Textual** there's no tree-table or Gantt widget (see Q1). Two options for building it:
  (a) a `Tree` whose node labels are Rich `Text` with a padded name column plus a bar made of block characters (`▏▎▍▌▋▊▉█`) scaled to the session span, or
  (b) a `DataTable` in row-cursor mode, with indentation done by hand in the name column and a bar column rendered as a Rich renderable.
  (a) gives folding for free through `Tree` expand/collapse. (b) gives columns and sorting. The #383 prototype should try (a) first, because folding subagents is the main interaction.
- Precedent for rendering Gantt bars in Rich: `Progress` bars are exactly this kind of fixed-width block bar ([Rich README](https://github.com/Textualize/rich)). There's no ready-made Gantt renderable (unverified beyond the README list).

## Q4. Prior art for observability TUIs

| Tool | Layout | Navigation | Refresh |
|---|---|---|---|
| **k9s** ([README](https://github.com/derailed/k9s)) | One resource table at a time, with a header and **crumbs** at the bottom (`ctrl-g` toggles them, `crumbsless` config) | `:` command mode to switch resource (`:pod`, `:xray dp`), `/` regex filter, `enter` drill down, `esc` "go up/back to the previous view. If you have crumbs on, this will go to the previous one", `-` last command ("like `cd -`"), `[`/`]` history, `?` help | `refreshRate: 2` (seconds) in config |
| **lazygit** ([keybindings](https://github.com/jesseduffield/lazygit/blob/master/docs/keybindings/Keybindings_en.md), [config](https://github.com/jesseduffield/lazygit/blob/master/docs/Config.md)) | Fixed side panels (status/files/branches/commits/stash) plus a main view | `1`–`5` jump to a panel (`jumpToBlock`), `0` focuses the main view, `tab`/`h`/`l` next/previous panel, `[`/`]` tabs inside a panel, `/` search, `enter` drill in, `?` keybindings menu, `+`/`_` normal/half/fullscreen | `refresher.refreshInterval: 10` s, auto-refresh in the background |
| **htop** ([man page](https://www.man7.org/linux/man-pages/man1/htop.1.html)) | Meters on top, process table below | F3 search, F4 incremental filter, F5 tree, F6 sort field, F2 setup | `-d` delay in tenths of a second, clamped to 0.1–10 s |
| **btop** ([README](https://github.com/aristocratos/btop)) | Boxes (cpu/mem/net/proc) in presets | Number keys show/hide boxes, up to 9 layout presets, mouse support | `update_ms`, "recommended 2000 ms or above" |
| **Harlequin** (Textual; [usage docs source](https://github.com/tconbeer/harlequin-web/blob/main/src/docs/getting-started/usage.md)) | Catalog tree on the left, editor top-right, results bottom-right | F-keys focus panes (F2 editor, F6 catalog), `enter`/`space` expand tree nodes, F10 fullscreen the focused pane, `ctrl+b` hide the sidebar | n/a (query-driven) |

What they have in common, and what to copy: **one landing screen of panes you can jump to with a key**; **`enter` to drill down, `esc` to back out, with a visible breadcrumb**; **`/` to filter the focused list**; **`?` to list keybindings** (Textual's `Footer` shows bindings); **a way to fullscreen one pane**; **a 2–10 s refresh interval**, which matches #380's 2–5 s. k9s's single-table-plus-crumbs model suits drill-down (dashboard → session → entity). lazygit's numbered panels suit the dashboard landing screen.

## Sketches

### Knowledge graph view (focused entity, walk with breadcrumbs)

```
 Context Graph ▸ Knowledge ▸ Memgraph ▸ sessions-graph ▸ reconcile_session          ⟳ 3s  ● live
┌ Neighbours of  reconcile_session  (function) ─────────┐┌ Provenance ───────────────────────────────┐
│ ▾ → CALLS (3)                                          ││ reconcile_session —DEFINED_IN→ sessions-… │
│     create_episode                    12 mentions      ││                                           │
│     from_texts            (unstructured2graph)  8      ││ Sessions (4)                   last seen  │
│     MERGE_session                                3     ││ ▸ 7c1e… "fix Episode upsert"   2026-09-28 │
│ ▾ ← DEFINES (1)                                        ││ ▸ a90b… "reconcile queue"      2026-09-24 │
│     sessions-graph  ↺ on path                          ││   4f02… "eval baseline"        2026-09-19 │
│ ▸ → WRITES (2)                                         ││   e5d7… "GLiNER2 backend"      2026-09-11 │
│ ▸ ← MENTIONED_BY (14)                                  ││                                           │
│ ▾ → RELATED_TO (41)                                    ││ Evidence (chunk, session 7c1e…)           │
│     Episode                              22            ││ "…reconcile_session() now MERGEs the      │
│     Chunk                                19            ││  HAS_EPISODE edge so a re-run updates…"   │
│     … + 39 more  (enter to page)                       ││                                           │
│                                                        ││ provenance is session-level when a        │
│                                                        ││ session produced >1 chunk                 │
└────────────────────────────────────────────────────────┘└───────────────────────────────────────────┘
 enter focus  ←/→ collapse/expand  tab provenance  s open session  esc back  / filter  ? keys  q quit
```

### Session anatomy view (tree-table with Gantt bars)

```
 Context Graph ▸ Sessions ▸ 7c1e…  "fix Episode upsert"   claude-code · ~/repos/ai-toolkit   ⟳ 3s
┌ Timeline ─────────────────────────────────── 0 ───── 5m ───── 10m ───── 15m ───── 20m ─┐┌ Details ───────────┐
│ ▾ Session 7c1e…                     22m14s ██████████████████████████████████████████ ││ Action  a3…        │
│   ├ Message user                        0s ▏                                           ││ kind  ToolCall     │
│   ├ ToolCall Read sessions.py        0.4s  ▏                                           ││ tool  Bash         │
│   ├ ▾ Agent Explore  (subagent)      4m02s  ███████▋                                   ││ at    09:14:22     │
│   │   ├ ToolCall Grep               0.2s   ▏                                           ││ dur   2.1s         │
│   │   ├ ToolCall Read                0.1s     ▏                                        ││ ok    ✓            │
│   │   └ Message assistant               –         ▏                                    ││                    │
│   ├ ToolCall Edit reconcile.py       0.3s             ▏                                ││ input              │
│   ├ ▸ Agent general-purpose (3)      9m40s             ████████████████▌               ││  pytest -q tests/  │
│ » ├ ToolCall Bash pytest              2.1s                               ▎             ││                    │
│   └ Message assistant                  –                                         ▏     ││                    │
├ Derived ───────────────────────────────────────────────────────────────────────────────┤│                    │
│ Episode  ✓ summarized 09:40   "Fixed reconcile_session so re-runs update the Episode…" ││                    │
│ Chunks 3 · Entities 17 (+5 new) · Relations 24   ▸ reconcile_session  ▸ Episode  ▸ …   ││                    │
└────────────────────────────────────────────────────────────────────────────────────────┘└────────────────────┘
 enter fold/unfold  ↑/↓ move  e entities  g graph from entity  f fullscreen  esc back  ? keys
```

## Open items for #383 (prototype)

- Check flicker by eye while polling every 2–5 s in Terminal.app, iTerm2/Ghostty, a Linux terminal and Windows Terminal. The synchronized-output code path depends on the terminal reporting mode 2026 support.
- Choose Tree-with-bars (a) or DataTable (b) for the timeline, over a real dogfood session with nested subagents.
- Time a one-hop neighbour query grouped by relation type on a hub entity, to set the top-N cap.
- Poll errors must never reach Textual's default exception handler, which ends the app *(measured)*. Show a stale/error badge in the header instead.

## Reproduction

Throwaway, run in `/tmp/tuisize`, not committed:

- **Sizes:** for each package, `uv venv -p 3.12 <dir> && uv pip install -p <dir>/bin/python <pkg>`, then `du -sk <dir>/lib/python3.12/site-packages` and `uv pip list`. Marginal Textual cost: install `agent-context-graph` (PyPI 0.2.0), measure, add `textual`, measure again.
- **Metadata:** `https://pypi.org/pypi/<pkg>/json` (`requires_python`, licence, upload times). GitHub `repos/<r>` and `repos/<r>/commits?since=<90 days ago>` through `gh api`.
- **Graph rendering:** an 11-edge sample entity graph rendered with `networkx.write_network_text` (networkx 3.7) and `phart.ASCIIRenderer` (phart 1.1.4, the last version installable on Python <3.14).
- **Textual probe (CPython 3.10.16, Textual 8.2.8):** `set_interval(0.05)` patching one `DataTable` cell by key, plus a `Tree`, driven by `run_test(size=(60, 14))` with `pilot.press(...)`. The cell advanced 14 times in about 0.5 s, and `export_screenshot()` returned an 11.7 KB SVG. An earlier version of the probe raised inside the timer callback, and that exception ended the app.
