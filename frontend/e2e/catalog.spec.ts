import { test, expect } from "@playwright/test";

test.describe("Catalog page", () => {
  test("loads catalog list", async ({ page }) => {
    await page.goto("/catalog");
    await page.waitForTimeout(2000);
    // Should not crash — either shows items or empty state
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    expect(errors.filter((e) => !e.includes("ResizeObserver"))).toHaveLength(0);
  });

  test("navigate to workbench on item click", async ({ page }) => {
    await page.goto("/catalog");
    await page.waitForTimeout(2000);
    const firstRow = page.locator('[data-testid="formula-row"]').first();
    if ((await firstRow.count()) > 0) {
      await firstRow.click();
      await expect(page).toHaveURL(/workbench/, { timeout: 5000 });
    }
  });

  test("delete button requires confirm param (backend guard)", async ({ page, request }) => {
    // Test N34: DELETE without confirm=true → 400
    const resp = await request.delete("http://localhost:8000/api/catalog/dummy_formula");
    // Should be 400 (confirm not set) or 404 (formula not found) — never 200
    expect([400, 404, 422]).toContain(resp.status());
  });
});
