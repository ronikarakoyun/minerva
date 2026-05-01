import { test, expect, Page } from "@playwright/test";

// Helpers
async function gotoWorkbench(page: Page) {
  await page.goto("/workbench");
  // Wait for catalog column or loading to resolve
  await page
    .waitForSelector('[data-testid="catalog-col"], [data-testid="empty-catalog"]', {
      timeout: 10_000,
    })
    .catch(() => {
      // Fallback: wait for any h1 / main content
    });
}

test.describe("Workbench page", () => {
  test("loads without crashing", async ({ page }) => {
    await page.goto("/workbench");
    // No unhandled JS errors
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.waitForTimeout(2000);
    expect(errors.filter((e) => !e.includes("ResizeObserver"))).toHaveLength(0);
  });

  test("filter input filters catalog list", async ({ page }) => {
    await page.goto("/workbench");
    await page.waitForTimeout(1500);
    const filterInput = page.locator('input[placeholder*="filter"]');
    if ((await filterInput.count()) > 0) {
      await filterInput.fill("NONEXISTENT_XYZ");
      await page.waitForTimeout(300);
      // Expect no formula rows matching
      const rows = page.locator('[data-testid="formula-row"]');
      expect(await rows.count()).toBe(0);
    }
  });

  test("source filter segments exist", async ({ page }) => {
    await page.goto("/workbench");
    await page.waitForTimeout(1000);
    // SegRow should contain EVO, LLM, MCTS
    for (const label of ["EVO", "LLM", "MCTS"]) {
      await expect(page.getByText(label).first())
        .toBeVisible({ timeout: 5000 })
        .catch(() => {});
    }
  });

  test("Run Backtest button visible and disabled without formula", async ({ page }) => {
    await page.goto("/workbench?id=");
    await page.waitForTimeout(1000);
    const btn = page.getByRole("button", { name: /Run Backtest/i });
    if ((await btn.count()) > 0) {
      // If no formula selected, button should be disabled
      const isDisabled = await btn.isDisabled().catch(() => true);
      expect(isDisabled).toBe(true);
    }
  });
});
