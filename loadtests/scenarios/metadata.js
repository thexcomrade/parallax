import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { checkStatus } from '../lib/checks.js';
import { apiLatency, totalRequests } from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'stage1_50';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.01'],
  http_req_duration: ['p(95)<500', 'p(99)<1000'],
});

const BASE_URL = getBaseUrl();

export default function () {
  const t0 = new Date().getTime();
  const res = http.get(`${BASE_URL}/api/v1/model`, {
    tags: { endpoint: '/api/v1/model', scenario: 'metadata' },
  });
  apiLatency.add(new Date().getTime() - t0);
  totalRequests.add(1);

  checkStatus(res, 200, 'GET /api/v1/model');
  sleep(0.1);
}
