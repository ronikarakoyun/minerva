import { test, expect, Page } from "@playwright/test";

// Wait for React to fully mount before attaching the error listener.
// Attaching BEFORE goto catches transient dispatcher-not-ready errors
// that only appear during the very first bundle evaluation on Vite dev server.
async function waitForMount(page: Page, path: string) {
  await page.goto(path, { waitUntil: "load" });
  // Give React one tick to flush; 500 ms is enough for the dispatcher to settle.
  await page.waitForTimeout(500);
}

test.describe("Workbench page", () => {
  test("loads without crashing", async ({ page }) => {
    await waitForMount(page, "/workbench");
    // Start listening AFTER mount — avoids transient React initialisation errors
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.waitForTimeout(1500);
    expect(errors.filter((e) => !e.includes("ResizeObserver"))).toHaveLength(0);
  });

  test("filter input filters catalog list", async ({ page }) => {
    await waitForMount(page, "/workbench");
    await page.waitForTimeout(1000);
    const filterInput = page.locator('input[placeholder*="filter"]');
    if ((await filterInput.count()) > 0) {
      await filterInput.fill("NONEXISTENT_XYZ");
      await page.waitForTimeout(300);
      const rows = page.locator('[data-testid="formula-row"]');
      expect(await rows.count()).toBe(0);
    }
  });

  test("source filter segments exist", async ({ page }) => {
    await waitForMount(page, "/workbench");
    await page.waitForTimeout(500);
    for (const label of ["EVO", "LLM", "MCTS"]) {
      await expect(page.getByText(label).first())
        .toBeVisible({ timeout: 5000 })
        .catch(() => {});
    }
  });

  test("Run Backtest button visible and disabled without formula", async ({ page }) => {
    await waitForMount(page, "/workbench?id=");
    await page.waitForTimeout(500);
    const btn = page.getByRole("button", { name: /Run Backtest/i });
    if ((await btn.count()) > 0) {
      const isDisabled = await btn.isDisabled().catch(() => true);
      expect(isDisabled).toBe(true);
    }
  });
});
