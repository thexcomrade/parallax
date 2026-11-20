import http from 'k6/http';
import { sleep } from 'k6';
import { getBaseUrl } from '../lib/client.js';
import { getOrCreateAuthToken } from '../lib/auth.js';
import { getSyntheticVideo } from '../lib/fixtures.js';
import { checkStatus, checkPublicVerdict } from '../lib/checks.js';
import {
  totalJobsEnqueued,
  totalJobsCompleted,
  totalJobsFailed,
  totalRequests,
  inferenceDuration,
} from '../lib/metrics.js';

export const options = {
  scenarios: {
    inference_flow: {
      executor: 'shared-iterations',
      vus: 1,
      iterations: 3,
      maxDuration: '5m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
    inference_duration_ms: ['p(95)<25000'],
  },
};

const BASE_URL = getBaseUrl();
const videoData = getSyntheticVideo();

export function setup() {
  const token = getOrCreateAuthToken(102);
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
    tags: { endpoint: '/api/v1/analyses', scenario: 'inference' },
  });
  totalRequests.add(1);

  if (res.status !== 202) {
    totalJobsFailed.add(1);
    sleep(1);
    return;
  }

  totalJobsEnqueued.add(1);
  const analysisId = res.json().id;

  // Poll for completion (up to 45 seconds)
  let completed = false;
  let attempts = 0;
  while (!completed && attempts < 30) {
    sleep(1.5);
    attempts++;

    const pollRes = http.get(`${BASE_URL}/api/v1/analyses/${analysisId}`, {
      headers: headers,
      tags: { endpoint: '/api/v1/analyses/:id', scenario: 'inference' },
    });
    totalRequests.add(1);

    if (pollRes.status === 200) {
      const state = pollRes.json().status;
      if (state === 'completed') {
        completed = true;
        totalJobsCompleted.add(1);
        const duration = new Date().getTime() - t0;
        inferenceDuration.add(duration);
        break;
      } else if (state === 'failed') {
        totalJobsFailed.add(1);
        break;
      }
    }
  }

  if (completed) {
    // Verify public binary verdict
    const pubRes = http.get(`${BASE_URL}/api/v1/analyses/${analysisId}/public`, {
      tags: { endpoint: '/api/v1/analyses/:id/public', scenario: 'inference' },
    });
    totalRequests.add(1);
    checkPublicVerdict(pubRes);
  }

  sleep(2);
}
