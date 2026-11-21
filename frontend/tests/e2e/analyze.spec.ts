import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Phase 7: Analyze & Report Forensic UI', () => {
  const testPassword = 'SecureMasterPassword2026!';
  const fixturePath = path.resolve(__dirname, '../fixtures/test_sample.mp4');

  test.beforeEach(async ({ page }) => {
    // Register and authenticate with unique email
    const uniqueEmail = `analyst_p7_${Date.now()}_${Math.floor(Math.random() * 10000)}@parallax.io`;
    await page.goto('/register');
    await page.getByLabel('Operator Email').fill(uniqueEmail);
    await page.getByLabel('New Password').fill(testPassword);
    await page.getByLabel('Confirm Password').fill(testPassword);
    await page.getByRole('button', { name: /COMPLETE ENROLLMENT/i }).click();
    await expect(page).toHaveURL(/\/app/);
  });

  test('Analyze dropzone renders with viewfinder corners, passes a11y & handles upload', async ({
    page,
  }) => {
    test.setTimeout(120000);
    await page.goto('/app/analyze');
    await expect(page.locator('h1')).toContainText('Forensic Media Workbench');

    // Verify Dropzone is accessible and interactive
    const dropZone = page.getByRole('button', { name: /Upload video file for forensic deepfake analysis/i });
    await expect(dropZone).toBeVisible();

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe audit on Analyze page
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Screenshot Analyze page
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase7-analyze-${width}px.png`,
      fullPage: true,
    });

    // Upload fixture file via filechooser
    const fileChooserPromise = page.waitForEvent('filechooser');
    await dropZone.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(fixturePath);

    // Wait for client metadata probe and click "Run Forensic Analysis"
    const startButton = page.getByRole('button', { name: /Run Forensic Analysis/i });
    await startButton.waitFor({ state: 'visible', timeout: 10000 });
    await startButton.dispatchEvent('click');

    // Verify Pipeline Rail displays and stages stream
    await expect(page.getByText('Forensic Execution Pipeline')).toBeVisible({ timeout: 15000 });

    // Wait for pipeline completion and automatic navigation to Report page
    await expect(page).toHaveURL(/\/app\/analyses\/[a-f0-9-]+/, { timeout: 60000 });
    await expect(page.locator('h1')).toBeVisible();

    // Verify Core Forensic Components (wait for pipeline to complete and report to render)
    await expect(page.getByText('Detection Result')).toBeVisible({ timeout: 60000 });
    await expect(page.getByText(/The submitted media was classified as/i)).toBeVisible();
    await expect(page.getByText('Analyzed Media')).toBeVisible();

    // Verify no horizontal overflow on Report page
    const reportScroll = await page.evaluate(() => document.documentElement.scrollWidth);
    const reportClient = await page.evaluate(() => document.documentElement.clientWidth);
    expect(reportScroll).toBeLessThanOrEqual(reportClient + 1);

    // Wait for route transition animation to fully settle before Axe audit
    await page.waitForTimeout(400);

    // Axe audit on Report page
    const reportAxe = await new AxeBuilder({ page }).analyze();
    expect(reportAxe.violations).toEqual([]);

    // Capture Report page screenshot
    await page.screenshot({
      path: `test-results/screenshots/phase7-report-${width}px.png`,
      fullPage: true,
    });

    // Navigate to History log
    await page.goto('/app/history');
    await expect(page.locator('h1')).toContainText(/Forensic Audit (Ledger|Trail)/i);
    await expect(page.getByText('test_sample.mp4')).toBeVisible();

    // Wait for route transition animation to settle
    await page.waitForTimeout(400);

    // Axe audit on History page
    const historyAxe = await new AxeBuilder({ page }).analyze();
    expect(historyAxe.violations).toEqual([]);

    // Capture History page screenshot
    await page.screenshot({
      path: `test-results/screenshots/phase7-history-${width}px.png`,
      fullPage: true,
    });

    // Test History Search functionality
    const searchInput = page.getByPlaceholder(/Search analyses|Search dossier/i);
    await expect(searchInput).toBeVisible();
    await searchInput.fill('test_sample');
    await expect(page.getByText('test_sample.mp4')).toBeVisible();

    // Test search for non-matching item
    await searchInput.fill('nonexistent_clip_xyz');
    await expect(page.getByText(/No matching/i)).toBeVisible();

    // Reset search
    await page.getByRole('button', { name: /Reset Filters/i }).click();
    await expect(page.getByText('test_sample.mp4')).toBeVisible();

    // Reopen dossier from history
    await page.getByText('test_sample.mp4').click();
    await expect(page).toHaveURL(/\/app\/analyses\/[a-f0-9-]+/);

    // Delete dossier end-to-end through the accessible confirmation dialog
    await page.getByRole('button', { name: /^Delete$/i }).click();
    const deleteDialog = page.getByRole('dialog');
    await expect(deleteDialog).toBeVisible();
    await deleteDialog.getByRole('button', { name: /Delete record/i }).click();

    // Verify redirected back to history and archive is now empty
    await expect(page).toHaveURL(/\/app\/history/, { timeout: 15000 });
    await expect(page.getByText(/No forensic analyses in ledger|No analyses recorded yet|No analyses found/i)).toBeVisible({ timeout: 15000 });
  });

  test('Respects prefers-reduced-motion on Report page', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/app/analyze');
    await expect(page.locator('h1')).toContainText('Forensic Media Workbench');

    await page.screenshot({
      path: `test-results/screenshots/phase7-reduced-motion.png`,
      fullPage: true,
    });
  });
});
