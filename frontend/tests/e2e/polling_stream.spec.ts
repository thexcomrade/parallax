import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Workstream E: Frontend Polling Optimization & Adaptive Stream', () => {
  const testPassword = 'SecureMasterPassword2026!';
  const fixturePath = path.resolve(__dirname, '../fixtures/test_sample.mp4');

  test('Adaptive polling terminates immediately upon completion with zero phantom polls', async ({
    page,
  }) => {
    test.setTimeout(90000);

    // 1. Authenticate dedicated operator
    const uniqueEmail = `poll_test_${Date.now()}@parallax.io`;
    await page.goto('/register');
    await page.getByLabel('Operator Email').fill(uniqueEmail);
    await page.getByLabel('New Password').fill(testPassword);
    await page.getByLabel('Confirm Password').fill(testPassword);
    await page.getByRole('button', { name: /COMPLETE ENROLLMENT/i }).click();
    await expect(page).toHaveURL(/\/app/);

    // Track polling requests to /api/v1/analyses/[id]
    const pollTimestamps: number[] = [];
    let completedTimestamp: number | null = null;
    let postCompletionPolls = 0;

    await page.route(/\/api\/v1\/analyses\/[a-f0-9-]+$/, async (route) => {
      const now = Date.now();
      pollTimestamps.push(now);

      if (completedTimestamp !== null && now > completedTimestamp + 300) {
        postCompletionPolls++;
      }

      const response = await route.fetch();
      const json = await response.json().catch(() => ({}));
      if (json.status === 'completed' || json.status === 'failed') {
        if (completedTimestamp === null) {
          completedTimestamp = Date.now();
        }
      }

      await route.fulfill({ response });
    });

    // 2. Navigate to Analyze Workbench and submit video
    await page.goto('/app/analyze');
    const dropZone = page.getByRole('button', {
      name: /Upload video file for forensic deepfake analysis/i,
    });
    await expect(dropZone).toBeVisible();

    const fileChooserPromise = page.waitForEvent('filechooser');
    await dropZone.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(fixturePath);

    const startButton = page.getByRole('button', { name: /Run Forensic Analysis/i });
    await startButton.waitFor({ state: 'visible', timeout: 10000 });
    await startButton.dispatchEvent('click');

    // 3. Verify Pipeline displays and streams
    await expect(page.getByText('Forensic Execution Pipeline')).toBeVisible({ timeout: 15000 });

    // 4. Wait for completion and transition to Report
    await expect(page).toHaveURL(/\/app\/analyses\/[a-f0-9-]+/, { timeout: 60000 });
    await expect(page.getByText('Detection Result')).toBeVisible({ timeout: 60000 });

    // 5. Allow settling time (3 seconds) to verify no subsequent polls are issued
    await page.waitForTimeout(3000);

    // 6. Assertions: Zero polling requests dispatched after terminal status
    expect(postCompletionPolls).toBe(0);
    expect(completedTimestamp).not.toBeNull();
  });
});
