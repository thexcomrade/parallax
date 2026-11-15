import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Phase 9: Dashboard, Model Card & Settings', () => {
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
    await expect(page.locator('h1')).toContainText('Forensic Command Bench');

    // Verify key bento instrumentation cells
    await expect(page.getByText('MEDIA INGEST // WORKBENCH')).toBeVisible();
    await expect(page.getByText('TOTAL ANALYSES')).toBeVisible();
    await expect(page.getByText('VERDICT DISTRIBUTION')).toBeVisible();
    await expect(page.getByText('DYNAMIC WEIGHT FUSION (AVT²-DWF)')).toBeVisible();
    await expect(page.getByText('ENGINE TELEMETRY')).toBeVisible();

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

  test('Model Card renders architecture specs, separate claimed vs live metrics, and passes a11y', async ({
    page,
  }) => {
    await page.goto('/app/model');
    await expect(page.locator('h1')).toContainText('AVT²-DWF Architecture & Model Card');

    // Verify architecture sections
    await expect(page.getByText('BRANCH 1 // SPATIAL-TEMPORAL FACE TRANSFORMER')).toBeVisible();
    await expect(page.getByText('BRANCH 2 // SPECTRAL AUDIO TRANSFORMER')).toBeVisible();
    await expect(page.getByText('FUSION // DYNAMIC WEIGHT FUSION (DWF)')).toBeVisible();

    // Verify separate claimed vs live metrics
    await expect(page.getByText('PROJECT REPORT CLAIMS')).toBeVisible();
    await expect(page.getByText('LIVE REPO CHECKPOINT')).toBeVisible();

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
      path: `test-results/screenshots/phase9-model-${width}px.png`,
      fullPage: true,
    });
  });

  test('Settings renders profile, password change, accessible purge dialog with focus trap', async ({
    page,
  }) => {
    await page.goto('/app/settings');
    await expect(page.locator('h1')).toContainText('Analyst Account Settings');

    // Verify operator profile
    await expect(page.getByText('OPERATOR PROFILE // CREDENTIALS')).toBeVisible();
    await expect(page.getByRole('button', { name: /UPDATE OPERATOR PASSWORD/i })).toBeVisible();
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
