"""Сохранить пять полноразмерных скриншотов локального Streamlit-интерфейса."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATH = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
OUTPUT_DIR = PROJECT_ROOT / "results" / "rk2_week5" / "screenshots"
BASE_URL = "http://127.0.0.1:8501"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not EDGE_PATH.exists():
        raise FileNotFoundError(f"Microsoft Edge не найден: {EDGE_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE_PATH),
            headless=True,
            args=["--disable-gpu", "--hide-scrollbars"],
        )
        for sample_id in range(1, 6):
            context = browser.new_context(
                viewport={"width": 1440, "height": 1450},
                device_scale_factor=1,
                color_scheme="light",
            )
            page = context.new_page()
            page.goto(
                f"{BASE_URL}/?sample={sample_id}&autorun=1",
                wait_until="domcontentloaded",
                timeout=120_000,
            )
            page.get_by_text("Предсказанный нормативный акт").wait_for(
                state="visible", timeout=120_000
            )
            page.locator('[data-testid="probability-bars"]').wait_for(
                state="visible", timeout=120_000
            )
            page.wait_for_function(
                """() => Array.from(document.querySelectorAll('[data-testid="stMetricValue"]'))
                    .filter((node) => node.textContent.trim()).length >= 3""",
                timeout=120_000,
            )
            page.wait_for_timeout(750)
            if page.locator("[data-testid='stException']").count():
                raise RuntimeError(f"Streamlit показал исключение в тесте {sample_id}")
            output_path = OUTPUT_DIR / f"test_{sample_id}.png"
            page.screenshot(path=str(output_path), full_page=False)
            print(f"Тест {sample_id}: {output_path}")
            context.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
