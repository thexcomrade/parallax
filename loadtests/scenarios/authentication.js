import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { checkStatus } from '../lib/checks.js';
import { authLatency, totalRequests } from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'smoke';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.10'], // Note rate limit may trigger 429
  http_req_duration: ['p(95)<1500', 'p(99)<2500'],
});

const BASE_URL = getBaseUrl();
const email = __ENV.TEST_USER_EMAIL || 'staging_load_test@example.com';
const password = __ENV.TEST_USER_PASSWORD || 'SecureStagingPass2026!';

export function setup() {
  // Ensure the account exists before running the load test
  const regPayload = JSON.stringify({ email, password });
  http.post(`${BASE_URL}/api/v1/auth/register`, regPayload, {
    headers: { 'Content-Type': 'application/json' },
  });
  return {};
}

export default function () {
  const loginPayload = JSON.stringify({ email, password });
  const t0 = new Date().getTime();
  const res = http.post(`${BASE_URL}/api/v1/auth/login`, loginPayload, {
    headers: { 'Content-Type': 'application/json' },
    tags: { endpoint: '/api/v1/auth/login', scenario: 'auth' },
  });
  authLatency.add(new Date().getTime() - t0);
  totalRequests.add(1);

  if (res.status === 200) {
    const token = res.json().access_token;
    // Authenticated profile call
    const meRes = http.get(`${BASE_URL}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      tags: { endpoint: '/api/v1/auth/me', scenario: 'auth' },
    });
    checkStatus(meRes, 200, 'GET /auth/me');
  } else if (res.status === 429) {
    // Expected under rate limit
    checkStatus(res, 429, 'Rate limit triggered');
  }

  sleep(0.5);
}
