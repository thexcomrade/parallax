import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { getOrCreateAuthToken } from '../lib/auth.js';
import { getSyntheticVideo } from '../lib/fixtures.js';
import { checkStatus } from '../lib/checks.js';
import { uploadLatency, totalUploads, totalRequests } from '../lib/metrics.js';
import { getStageOptions } from '../stages/progressive_stages.js';

const stageEnv = __ENV.STAGE || 'smoke';
export const options = getStageOptions(stageEnv, {
  http_req_failed: ['rate<0.10'],
  http_req_duration: ['p(95)<3000', 'p(99)<6000'],
});

const BASE_URL = getBaseUrl();
const videoData = getSyntheticVideo();

export function setup() {
  const token = getOrCreateAuthToken(100);
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

  const params = {
    headers: {
      Authorization: `Bearer ${data.token}`,
    },
    tags: { endpoint: '/api/v1/analyses', scenario: 'upload' },
  };

  const t0 = new Date().getTime();
  const res = http.post(`${BASE_URL}/api/v1/analyses`, payload, params);
  uploadLatency.add(new Date().getTime() - t0);
  totalRequests.add(1);

  if (res.status === 202) {
    totalUploads.add(1);
    checkStatus(res, 202, 'POST /analyses upload accepted');
  } else if (res.status === 429) {
    checkStatus(res, 429, 'Upload rate limit enforced');
  }

  sleep(1.0);
}
