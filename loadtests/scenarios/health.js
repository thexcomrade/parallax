import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { checkStatus } from '../lib/checks.js';
import { apiLatency, totalRequests } from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'stage1_50';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.005'],
  http_req_duration: ['p(95)<250', 'p(99)<500'],
});

const BASE_URL = getBaseUrl();

export default function () {
  const endpoints = ['/health/liveness', '/health/readiness', '/api/v1/health', '/api/v1/ready'];
  const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];

  const t0 = new Date().getTime();
  const res = http.get(`${BASE_URL}${endpoint}`, {
    tags: { endpoint: endpoint, scenario: 'health' },
  });
  apiLatency.add(new Date().getTime() - t0);
  totalRequests.add(1);

  checkStatus(res, 200, `GET ${endpoint}`);
  sleep(0.1);
}
