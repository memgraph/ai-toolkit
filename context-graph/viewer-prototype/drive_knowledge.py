"""PROTOTYPE — drives the knowledge-graph screens headless and saves SVG screenshots.

Run:  uv run --no-project --python 3.12 --with textual --with neo4j context-graph/viewer-prototype/drive_knowledge.py
"""

import asyncio
import sys
from pathlib import Path

from prototype_knowledge import EntityList
from prototype_viewer import Viewer

# Optional session id: start from that session's entities (e.g. a GLiNER2-reconciled one) instead of the global list.
SESSION = sys.argv[1] if len(sys.argv) > 1 else None
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "k"
OUT = Path(__file__).parent / "screenshots"


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    app = Viewer()
    async with app.run_test(size=(170, 46)) as pilot:

        async def shot(name: str, wait: float = 1.5) -> None:
            await pilot.pause(wait)
            app.save_screenshot(f"{PREFIX}{name[1:]}.svg", path=str(OUT))

        await pilot.pause(2.0)
        await pilot.press("a", "a")  # every real session is unattributed (#383 finding 1)
        await pilot.pause(2.0)
        if SESSION:
            app.push_screen(EntityList(SESSION))
        else:
            await pilot.press("k")
        await shot("k1_entities_global", 2.5)
        await pilot.press("enter")
        await shot("k2_nb_A_by_relation", 2.5)
        await pilot.press("down", "down", "down")
        await shot("k3_nb_A_evidence")
        await pilot.press("v")
        await shot("k4_nb_B_by_type")
        await pilot.press("v")
        await shot("k5_nb_C_table")
        await pilot.press("down", "down", "enter")
        await shot("k6_refocused_breadcrumb", 2.5)
        await pilot.press("escape", "escape", "escape")
        await pilot.press("slash", *"onto", "enter")
        await shot("k7_filter_onto")
        print("error:", app.error)


asyncio.run(main())
