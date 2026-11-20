import { Counter, Rate, Trend } from 'k6/metrics';

export const requestErrors = new Rate('request_errors');
export const apiLatency = new Trend('api_latency_ms', true);
export const authLatency = new Trend('auth_latency_ms', true);
export const uploadLatency = new Trend('upload_latency_ms', true);
export const queueWaitTime = new Trend('queue_wait_ms', true);
export const inferenceDuration = new Trend('inference_duration_ms', true);

export const totalRequests = new Counter('total_requests');
export const totalUploads = new Counter('total_uploads');
export const totalJobsEnqueued = new Counter('total_jobs_enqueued');
export const totalJobsCompleted = new Counter('total_jobs_completed');
export const totalJobsFailed = new Counter('total_jobs_failed');
