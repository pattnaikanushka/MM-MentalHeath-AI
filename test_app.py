import os
import sys
import time
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def run_tests():
    errors = []
    log("Launching Playwright Chromium browser for Calibrated & rPPG Dashboard E2E testing...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1600, 'height': 1200})
        page = context.new_page()

        page.on("pageerror", lambda err: errors.append(f"Page Error: {err.text}"))
        page.on("console", lambda msg: errors.append(f"Console {msg.type}: {msg.text}") if msg.type == "error" else None)

        try:
            log("Navigating to http://localhost:8501...")
            page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
            time.sleep(4)

            # Test Export Clinical Report Button
            log("Testing Export Clinical Report button...")
            export_btn = page.query_selector("button:has-text('Export Clinical Report')")
            if export_btn:
                export_btn.click()
                time.sleep(2)

            page.screenshot(path=os.path.join(ARTIFACT_DIR, "calibrated_rppg_dashboard.png"), full_page=True)
            log("Saved calibrated_rppg_dashboard.png")

        except Exception as e:
            errors.append(f"Execution Error: {str(e)}")

        browser.close()

    log("\n--- Calibrated & rPPG Verification Summary ---")
    if len(errors) == 0:
        log("SUCCESS: 0 errors encountered across Calibrated & rPPG Dashboard workflow!")
    else:
        log(f"Reported {len(errors)} potential errors/warnings:")
        for err in errors:
            log(f" - {err}")

if __name__ == "__main__":
    run_tests()
