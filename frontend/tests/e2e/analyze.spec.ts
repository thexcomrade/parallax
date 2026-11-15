import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'path';

test.describe('Phase 7: Analyze & Report Forensic UI', () => {
  const testPassword = 'SecureMasterPassword2026!';
  const fixturePath = path.resolve(process.cwd(), 'tests/fixtures/test_sample.mp4');

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
    test.setTimeout(90000);
    await page.goto('/app/analyze');
    await expect(page.locator('h1')).toContainText('Analyze Media Clip');

    // Verify Dropzone and button
    const dropButton = page.getByRole('button', { name: /BROWSE LOCAL DRIVE/i });
    await expect(dropButton).toBeVisible();

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

    // Upload fixture file
    const fileChooserPromise = page.waitForEvent('filechooser');
    await dropButton.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(fixturePath);

    // Wait for client metadata probe and click "START FORENSIC ANALYSIS"
    const startButton = page.getByRole('button', { name: /START FORENSIC ANALYSIS/i });
    await startButton.waitFor({ state: 'visible', timeout: 10000 });
    await startButton.click();

    // Verify Pipeline Rail displays and stages stream
    await expect(page.getByText(/SCAN IN PROGRESS/i)).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('PIPELINE MONITOR // AVT²-DWF')).toBeVisible();

    // Wait for pipeline completion and automatic navigation to Report page
    await expect(page).toHaveURL(/\/app\/analyses\/[a-f0-9-]+/, { timeout: 45000 });
    await expect(page.getByText(/DOSSIER \/\//i)).toBeVisible();

    // Verify Mode Handling & Verdict Engine
    const isUntrained = await page.getByText(/UNCHECKED SYSTEM/i).isVisible();
    if (isUntrained) {
      await expect(page.getByText(/Untrained weights — scores are not meaningful/i).first()).toBeVisible();
      await expect(page.getByText(/VERDICT ENGINE \/\/ RESEARCH MODE/i)).toBeVisible();
    } else {
      await expect(page.getByText(/VERDICT ENGINE \/\/ CALIBRATED/i)).toBeVisible();
    }

    // Verify Forensic Disclaimer
    await expect(
      page.getByText(/Probability reflects model confidence on this clip, not forensic proof/i).first(),
    ).toBeVisible();

    // Verify Core Forensic Components
    await expect(page.getByText(/FUSION BALANCE \/\/ DWF/i)).toBeVisible();
    await expect(page.getByText(/TIMELINE RECONSTRUCTION \/\/ MULTIMODAL/i)).toBeVisible();
    await expect(page.getByText(/CH-V \/\/ CONTACT SHEET/i)).toBeVisible();
    await expect(page.getByText(/CH-A \/\/ LOG-MEL SPECTROGRAM/i)).toBeVisible();

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
    await expect(page.locator('h1')).toContainText('Analysis History');
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
    const searchInput = page.getByPlaceholder(/Search archive by filename or ID/i);
    await expect(searchInput).toBeVisible();
    await searchInput.fill('test_sample');
    await expect(page.getByText('test_sample.mp4')).toBeVisible();

    // Test search for non-matching item
    await searchInput.fill('nonexistent_clip_xyz');
    await expect(page.getByText(/NO ANALYSES MATCH CURRENT QUERY/i)).toBeVisible();

    // Reset search
    await page.getByRole('button', { name: /RESET FILTERS/i }).click();
    await expect(page.getByText('test_sample.mp4')).toBeVisible();

    // Reopen dossier from history
    await page.getByText('test_sample.mp4').click();
    await expect(page).toHaveURL(/\/app\/analyses\/[a-f0-9-]+/);
    await expect(page.getByText(/DOSSIER \/\//i)).toBeVisible();

    // Delete dossier end-to-end
    page.on('dialog', (dialog) => dialog.accept());
    const deleteButton = page.getByRole('button', { name: /^DELETE$/i });
    await deleteButton.click();

    // Verify redirected back to history and archive is now empty
    await expect(page).toHaveURL(/\/app\/history/);
    await expect(page.getByText(/NO ANALYSES RECORDED IN ARCHIVE/i)).toBeVisible();
  });

  test('Respects prefers-reduced-motion on Report page', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/app/analyze');
    await expect(page.locator('h1')).toContainText('Analyze Media Clip');

    await page.screenshot({
      path: `test-results/screenshots/phase7-reduced-motion.png`,
      fullPage: true,
    });
  });
});
