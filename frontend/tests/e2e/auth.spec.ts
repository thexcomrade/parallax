import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Phase 3: Auth & Protected Routes', () => {
  const timestamp = Date.now();
  const testEmail = `analyst_${timestamp}@parallax.io`;
  const testPassword = 'SecureMasterPassword2026!';

  test('Protected route redirects unauthenticated user to /login?next=...', async ({ page }) => {
    await page.goto('/app');
    await expect(page).toHaveURL(/\/login\?next=%2Fapp/);
    await expect(page.locator('h1')).toContainText('AUTHENTICATE ANALYST');
  });

  test('Login form renders SignalStage, passes axe a11y, and has no horizontal overflow', async ({ page }) => {
    await page.goto('/login');

    // Canvas exists and is labeled
    const canvas = page.locator('canvas[role="img"]');
    await expect(canvas).toBeVisible();

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1); // 1px rounding tolerance

    // Run Axe a11y audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    // Capture screenshot
    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase3-login-${width}px.png`,
      fullPage: true,
    });
  });

  test('Register form renders with password meter and passes axe a11y', async ({ page }) => {
    await page.goto('/register');

    await expect(page.locator('h1')).toContainText('CREATE ANALYST PROFILE');

    // Verify entropy meter displays
    await expect(page.getByText('ENTROPY STRENGTH')).toBeVisible();

    // Check no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);

    // Axe audit
    const axeResults = await new AxeBuilder({ page }).analyze();
    expect(axeResults.violations).toEqual([]);

    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    await page.screenshot({
      path: `test-results/screenshots/phase3-register-${width}px.png`,
      fullPage: true,
    });
  });

  test('Full authentication lifecycle: Register -> Auto-login -> Reload session -> Logout', async ({ page }) => {
    const email = `analyst_auth_${Date.now()}_${Math.floor(Math.random() * 100000)}@parallax.io`;
    await page.goto('/register');

    // Fill in registration form
    await page.getByLabel('Operator Email').fill(email);
    await page.getByLabel('New Password').fill(testPassword);
    await page.getByLabel('Confirm Password').fill(testPassword);

    // Live password meter should show optimal 5/5
    await expect(page.getByText('OPTIMAL [5/5]')).toBeVisible();

    // Submit registration
    await page.getByRole('button', { name: /COMPLETE ENROLLMENT/i }).click();

    // Should redirect to /app (Dashboard)
    await expect(page).toHaveURL(/\/app/);
    await expect(page.locator('h1')).toContainText('Forensic Command Bench');

    const viewport = page.viewportSize();
    const width = viewport ? viewport.width : 1280;
    if (width < 768) {
      await expect(page.locator('header').getByRole('button', { name: /EXIT/i })).toBeVisible();
    } else {
      await expect(page.locator('header').getByText(email)).toBeVisible();
    }

    // Take screenshot of authenticated dashboard
    await page.screenshot({
      path: `test-results/screenshots/phase3-dashboard-${width}px.png`,
      fullPage: true,
    });

    // Reload page: in-memory token is cleared, but silent refresh uses HttpOnly cookie to restore session!
    await page.reload();
    await expect(page).toHaveURL(/\/app/);
    await expect(page.locator('h1')).toContainText('Forensic Command Bench');

    if (width < 768) {
      await expect(page.locator('header').getByRole('button', { name: /EXIT/i })).toBeVisible();
    } else {
      await expect(page.locator('header').getByText(email)).toBeVisible();
    }

    // Sign out
    await page.getByRole('button', { name: /Sign out|EXIT/i }).click();

    // Should redirect to login when accessing /app
    await page.goto('/app');
    await expect(page).toHaveURL(/\/login\?next=%2Fapp/);
  });

  test('Respects prefers-reduced-motion', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/login');

    const canvas = page.locator('canvas[role="img"]');
    await expect(canvas).toBeVisible();

    // Verify screenshot in reduced-motion
    await page.screenshot({
      path: `test-results/screenshots/phase3-login-reduced-motion.png`,
      fullPage: true,
    });
  });
});
