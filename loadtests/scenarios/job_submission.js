import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { getOrCreateAuthToken } from '../lib/auth.js';
import { getSyntheticVideo } from '../lib/fixtures.js';
import { checkStatus } from '../lib/checks.js';
import { totalJobsEnqueued, totalRequests, queueWaitTime } from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'smoke';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.05'],
  http_req_duration: ['p(95)<2000', 'p(99)<4000'],
});

const BASE_URL = getBaseUrl();
const videoData = getSyntheticVideo();

export function setup() {
  const token = getOrCreateAuthToken(101);
  return { token };
}

export default function (data) {
  if (!data || !data.token || !videoData) {
    sleep(1);
    return;
  }

  const payload = {
    file: http.file(videoData, 'synthetic_video.mp4', 'video/mp4'),
  };

  const headers = {
    Authorization: `Bearer ${data.token}`,
  };

  const t0 = new Date().getTime();
  const res = http.post(`${BASE_URL}/api/v1/analyses`, payload, {
    headers: headers,
    tags: { endpoint: '/api/v1/analyses', scenario: 'job_submission' },
  });
  totalRequests.add(1);

  if (res.status === 202) {
    totalJobsEnqueued.add(1);
    const body = res.json();
    const analysisId = body.id;

    // Check status once
    sleep(0.5);
    const statusRes = http.get(`${BASE_URL}/api/v1/analyses/${analysisId}`, {
      headers: headers,
      tags: { endpoint: '/api/v1/analyses/:id', scenario: 'job_submission' },
    });
    totalRequests.add(1);
    queueWaitTime.add(new Date().getTime() - t0);
    checkStatus(statusRes, 200, 'GET /analyses/:id');
  }

  sleep(1.0);
}
