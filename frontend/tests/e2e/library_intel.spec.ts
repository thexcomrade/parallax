import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Phase 8: Library & Intel Public APIs', () => {
  const testPassword = 'SecureMasterPassword2026!';

  test.beforeEach(async ({ page }) => {
    // Register and authenticate with unique email
    const uniqueEmail = `analyst_p8_${Date.now()}_${Math.floor(Math.random() * 10000)}@parallax.io`;
    await page.goto('/register');
    await page.getByLabel('Operator Email').fill(uniqueEmail);
    await page.getByLabel('New Password').fill(testPassword);
    await page.getByLabel('Confirm Password').fill(testPassword);
    await page.getByRole('button', { name: /COMPLETE ENROLLMENT/i }).click();
    await expect(page).toHaveURL(/\/app/);
  });

  test('Library page displays reference corpus, filters correctly, triggers analysis, and makes 0 foreign network requests', async ({
    page,
  }) => {
    const foreignRequests: string[] = [];

    // Track all network requests to verify NO browser request leaves our origin
    page.on('request', (req) => {
      const url = new URL(req.url());
      if (url.origin !== 'http://localhost:5173' && url.origin !== 'http://127.0.0.1:5173' && !url.protocol.startsWith('data:')) {
        foreignRequests.push(req.url());
      }
    });

    await page.goto('/app/library');
    await expect(page.locator('h1')).toContainText('Forensic Benchmark Library');

    // Verify sample cards are displayed
    const apolloSample = page.getByText('Apollo 11 Crew Press Conference');
    await expect(apolloSample).toBeVisible();

    const styleganSample = page.getByText('StyleGAN-v2 High Resolution Face Synthesis');
    await expect(styleganSample).toBeVisible();

    // Verify filter tabs
    const authenticTab = page.getByRole('button', { name: /AUTHENTIC/i });
    await authenticTab.click();
    await expect(apolloSample).toBeVisible();
    await expect(styleganSample).not.toBeVisible();

    const syntheticTab = page.getByRole('button', { name: /SYNTHETIC/i });
    await syntheticTab.click();
    await expect(styleganSample).toBeVisible();
    await expect(apolloSample).not.toBeVisible();

    const allTab = page.getByRole('button', { name: /ALL SAMPLES|^ALL/i });
    await allTab.click();
    await expect(apolloSample).toBeVisible();
    await expect(styleganSample).toBeVisible();

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe accessibility audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Screenshot Library page
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase8-library-${width}px.png`,
      fullPage: true,
    });

    // Verify zero third-party requests were initiated by the browser
    expect(foreignRequests).toEqual([]);

    // Click "Analyze Sample" on Apollo 11
    const analyzeBtn = page.getByRole('button', { name: /Analyze/i }).first();
    await analyzeBtn.click();

    // Verify auto-navigation to report page
    await expect(page).toHaveURL(/\/app\/analyses\/[0-9a-f-]+/, { timeout: 30_000 });
  });

  test('Intel page renders telemetry panels, displays provenance notice, and makes 0 foreign network requests', async ({
    page,
  }) => {
    const foreignRequests: string[] = [];

    // Track all network requests to verify NO browser request leaves our origin
    page.on('request', (req) => {
      const url = new URL(req.url());
      if (url.origin !== 'http://localhost:5173' && url.origin !== 'http://127.0.0.1:5173' && !url.protocol.startsWith('data:')) {
        foreignRequests.push(req.url());
      }
    });

    await page.goto('/app/intel');
    await expect(page.locator('h1')).toContainText(/Threat & research recon/i);

    // Verify provenance notice
    await expect(page.getByLabel('Provenance notice')).toBeVisible();

    // Verify the three telemetry panels
    await expect(page.getByText('Media Threats')).toBeVisible();
    await expect(page.getByText('Academic Literature')).toBeVisible();
    await expect(page.getByText('Technical Pulse')).toBeVisible();

    // Verify refresh telemetry button is present
    const refreshBtn = page.getByRole('button', { name: /Refresh feeds/i });
    await expect(refreshBtn).toBeVisible();

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe accessibility audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Screenshot Intel page
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase8-intel-${width}px.png`,
      fullPage: true,
    });

    // Verify zero third-party requests were initiated by the browser
    expect(foreignRequests).toEqual([]);
  });
});
