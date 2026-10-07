"""PROTOTYPE — drives prototype_viewer.py headless against the configured Memgraph and saves SVG screenshots.

Run:  uv run --no-project --python 3.12 --with textual --with neo4j context-graph/viewer-prototype/drive_headless.py
"""

import asyncio
from pathlib import Path

from prototype_viewer import SessionScreen, Viewer

RECONCILED = "bde97559-11e2-4e63-a419-5ada48055fc0"

OUT = Path(__file__).parent / "screenshots"


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    app = Viewer()
    async with app.run_test(size=(160, 42)) as pilot:

        async def shot(name: str, wait: float = 1.5) -> None:
            await pilot.pause(wait)
            app.save_screenshot(f"{name}.svg", path=str(OUT))

        await shot("1_dash_A_user", 2.5)
        await pilot.press("a", "a")
        await shot("2_dash_A_all")
        await pilot.press("v")
        await shot("3_dash_B_all")
        await pilot.press("enter")
        await shot("4_session_A_live", 2.5)
        await pilot.press("v")
        await shot("5_session_B_live")
        await pilot.press("escape")
        app.push_screen(SessionScreen(RECONCILED))
        await shot("6_session_B_reconciled", 2.5)
        await pilot.press("down", "down", "down")
        await pilot.press("v")
        await shot("7_session_A_reconciled")
        print("error:", app.error)


asyncio.run(main())
