import http from 'k6/http';
import { check } from 'k6';
import { getBaseUrl } from './client.js';
import { authLatency, requestErrors } from './metrics.js';

export function getOrCreateAuthToken(userIndex = 1) {
  const baseUrl = getBaseUrl();
  const email = __ENV.TEST_USER_EMAIL || `staging_load_vu_${userIndex}@example.com`;
  const password = __ENV.TEST_USER_PASSWORD || 'SecureStagingPass2026!';

  // Attempt login first
  const loginPayload = JSON.stringify({ email, password });
  const params = {
    headers: { 'Content-Type': 'application/json' },
    tags: { endpoint: '/api/v1/auth/login' },
  };

  const t0 = new Date().getTime();
  let res = http.post(`${baseUrl}/api/v1/auth/login`, loginPayload, params);
  authLatency.add(new Date().getTime() - t0);

  if (res.status === 200) {
    const body = res.json();
    return body.access_token;
  }

  // If login failed (e.g. user does not exist), register user
  if (res.status === 401 || res.status === 404 || res.status === 422) {
    const regPayload = JSON.stringify({ email, password });
    const regParams = {
      headers: { 'Content-Type': 'application/json' },
      tags: { endpoint: '/api/v1/auth/register' },
    };
    const regRes = http.post(`${baseUrl}/api/v1/auth/register`, regPayload, regParams);

    if (regRes.status === 201 || regRes.status === 400 || regRes.status === 409) {
      // Retry login after register
      res = http.post(`${baseUrl}/api/v1/auth/login`, loginPayload, params);
      if (res.status === 200) {
        return res.json().access_token;
      }
    }
  }

  requestErrors.add(1);
  return null;
}
