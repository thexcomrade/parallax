import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL;
const GENERATOR_ID = __ENV.GENERATOR_ID || 'unknown_node';
const RUN_ID = __ENV.RUN_ID || 'run_default';
const TARGET_VUS = Number(__ENV.TARGET_VUS || 50);

const apiErrors = new Rate('parallax_api_errors');
const apiLatency = new Trend('parallax_api_latency_ms', true);
const totalRequests = new Counter('parallax_total_requests');

if (!BASE_URL) {
  throw new Error('[SECURITY FATAL] BASE_URL environment variable is required.');
}

const lowerUrl = BASE_URL.toLowerCase();
if (lowerUrl.includes('prod') || lowerUrl.includes('production')) {
  throw new Error(`[SECURITY FATAL] Refusing to execute against production URL: ${BASE_URL}`);
}

const approvedHosts = ['staging-api', 'localhost', '127.0.0.1', 'host.docker.internal'];
const isApproved = approvedHosts.some(h => lowerUrl.includes(h));
if (!isApproved) {
  throw new Error(`[SECURITY FATAL] Unapproved staging URL: ${BASE_URL}. Must be in allowlist: ${approvedHosts.join(', ')}`);
}

export const options = {
  scenarios: {
    distributed_api: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: TARGET_VUS },
        { duration: '60s', target: TARGET_VUS },
        { duration: '15s', target: 0 },
      ],
      gracefulRampDown: '15s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    parallax_api_errors: ['rate<0.02'],
  },
  tags: {
    project: 'PARALLAX',
    environment: 'staging',
    generator: GENERATOR_ID,
    run_id: RUN_ID,
  },
};

export function setup() {
  const res = http.get(`${BASE_URL}/health/readiness`, { timeout: '10s' });
  if (res.status !== 200) {
    throw new Error(`Staging API readiness probe failed before distributed run (Status: ${res.status})`);
  }
  return { started: true };
}

export default function () {
  const endpoints = ['/health/liveness', '/health/readiness', '/api/v1/health', '/api/v1/ready'];
  const ep = endpoints[Math.floor(Math.random() * endpoints.length)];

  const t0 = new Date().getTime();
  const response = http.get(`${BASE_URL}${ep}`, {
    tags: {
      endpoint: ep,
      generator: GENERATOR_ID,
      run_id: RUN_ID,
    },
    timeout: '10s',
  });

  const duration = new Date().getTime() - t0;
  apiLatency.add(duration);
  totalRequests.add(1);

  const success = check(response, {
    'HTTP 200': (r) => r.status === 200,
  });

  apiErrors.add(!success);
  sleep(0.1);
}

export function teardown() {
  // Non-destructive cleanup
}
