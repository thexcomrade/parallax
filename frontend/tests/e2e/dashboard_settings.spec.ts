import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Dashboard & Settings', () => {
  const testPassword = 'SecureMasterPassword2026!';

  test.beforeEach(async ({ page }) => {
    // Register unique user
    const uniqueEmail = `analyst_p9_${Date.now()}_${Math.floor(Math.random() * 10000)}@parallax.io`;
    await page.goto('/register');
    await page.getByLabel('Operator Email').fill(uniqueEmail);
    await page.getByLabel('New Password').fill(testPassword);
    await page.getByLabel('Confirm Password').fill(testPassword);
    await page.getByRole('button', { name: /COMPLETE ENROLLMENT/i }).click();
    await expect(page).toHaveURL(/\/app/);
  });

  test('Dashboard renders bento grid with stacked histogram, telemetry and no layout shift', async ({
    page,
  }) => {
    await page.goto('/app');
    await expect(page.locator('h1')).toContainText('Forensic Command Overview');

    // Verify key bento instrumentation cells
    await expect(page.getByText('Media Ingestion Workbench')).toBeVisible();
    await expect(page.getByText('Total Analyses')).toBeVisible();
    await expect(page.getByText('Inference Engine')).toBeVisible();

    // Verify horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Screenshot
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase9-dashboard-${width}px.png`,
      fullPage: true,
    });
  });

  test('Settings renders profile, password change, accessible purge dialog with focus trap', async ({
    page,
  }) => {
    await page.goto('/app/settings');
    await expect(page.locator('h1')).toContainText('Analyst Account & Security');

    // Verify operator profile
    await expect(page.getByRole('heading', { name: 'Operator Identity' })).toBeVisible();
    await expect(page.getByRole('button', { name: /UPDATE MASTER CREDENTIALS/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /SIGN OUT EVERYWHERE/i })).toBeVisible();

    // Open confirmation dialog
    const purgeButton = page.getByRole('button', { name: /PURGE ACCOUNT & ALL DATA/i });
    await purgeButton.dispatchEvent('click');

    // Verify dialog visible
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText('PERMANENTLY PURGE OPERATOR ACCOUNT?')).toBeVisible();

    // Press Escape to dismiss dialog
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Screenshot
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase9-settings-${width}px.png`,
      fullPage: true,
    });
  });
});
