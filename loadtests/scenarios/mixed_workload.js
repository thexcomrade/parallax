import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { getOrCreateAuthToken } from '../lib/auth.js';
import { getSyntheticVideo } from '../lib/fixtures.js';
import { checkStatus } from '../lib/checks.js';
import {
  apiLatency,
  authLatency,
  uploadLatency,
  totalRequests,
  totalUploads,
  totalJobsEnqueued,
} from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'stage1_50';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.02'],
  http_req_duration: ['p(95)<1500', 'p(99)<3000'],
});

const BASE_URL = getBaseUrl();
const videoData = getSyntheticVideo();

export function setup() {
  const token = getOrCreateAuthToken(200);
  return { token };
}

export default function (data) {
  const rand = Math.random();
  const headers = data && data.token ? { Authorization: `Bearer ${data.token}` } : {};

  if (rand < 0.60) {
    // 60%: Lightweight Health & Readiness Probes
    const endpoints = ['/health/liveness', '/health/readiness', '/api/v1/health', '/api/v1/ready'];
    const ep = endpoints[Math.floor(Math.random() * endpoints.length)];
    const t0 = new Date().getTime();
    const res = http.get(`${BASE_URL}${ep}`, { tags: { endpoint: ep, workload: 'health' } });
    apiLatency.add(new Date().getTime() - t0);
    totalRequests.add(1);
    checkStatus(res, 200, `GET ${ep}`);
  } else if (rand < 0.85) {
    // 25%: Metadata & Model Card
    const t0 = new Date().getTime();
    const res = http.get(`${BASE_URL}/api/v1/model`, {
      tags: { endpoint: '/api/v1/model', workload: 'metadata' },
    });
    apiLatency.add(new Date().getTime() - t0);
    totalRequests.add(1);
    checkStatus(res, 200, 'GET /api/v1/model');
  } else if (rand < 0.95) {
    // 10%: Authenticated Me & Auth Queries
    if (data && data.token) {
      const t0 = new Date().getTime();
      const res = http.get(`${BASE_URL}/api/v1/auth/me`, {
        headers: headers,
        tags: { endpoint: '/api/v1/auth/me', workload: 'auth' },
      });
      authLatency.add(new Date().getTime() - t0);
      totalRequests.add(1);
      checkStatus(res, 200, 'GET /auth/me');
    }
  } else {
    // 5%: Upload & Analysis Submission (Heavy path)
    if (data && data.token && videoData) {
      const payload = {
        file: http.file(videoData, 'synthetic_video.mp4', 'video/mp4'),
      };
      const t0 = new Date().getTime();
      const res = http.post(`${BASE_URL}/api/v1/analyses`, payload, {
        headers: headers,
        tags: { endpoint: '/api/v1/analyses', workload: 'upload' },
      });
      uploadLatency.add(new Date().getTime() - t0);
      totalRequests.add(1);

      if (res.status === 202) {
        totalUploads.add(1);
        totalJobsEnqueued.add(1);
        checkStatus(res, 202, 'POST /analyses upload');
      } else if (res.status === 429) {
        checkStatus(res, 429, 'Rate limit enforced');
      }
    }
  }

  sleep(0.1);
}
