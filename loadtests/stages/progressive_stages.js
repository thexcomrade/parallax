export const STAGE_CONFIGS = {
  smoke: [
    { duration: '10s', target: 5 },
    { duration: '20s', target: 5 },
    { duration: '5s', target: 0 },
  ],
  stage1_50: [
    { duration: '30s', target: 50 },
    { duration: '60s', target: 50 },
    { duration: '15s', target: 0 },
  ],
  stage2_100: [
    { duration: '30s', target: 100 },
    { duration: '60s', target: 100 },
    { duration: '15s', target: 0 },
  ],
  stage3_500: [
    { duration: '45s', target: 500 },
    { duration: '90s', target: 500 },
    { duration: '20s', target: 0 },
  ],
  stage4_1k: [
    { duration: '60s', target: 1000 },
    { duration: '120s', target: 1000 },
    { duration: '30s', target: 0 },
  ],
  stage5_5k: [
    { duration: '90s', target: 5000 },
    { duration: '180s', target: 5000 },
    { duration: '45s', target: 0 },
  ],
  stage6_10k: [
    { duration: '120s', target: 10000 },
    { duration: '180s', target: 10000 },
    { duration: '60s', target: 0 },
  ],
  stage7_20k: [
    { duration: '120s', target: 20000 },
    { duration: '180s', target: 20000 },
    { duration: '60s', target: 0 },
  ],
  stage8_30k: [
    { duration: '120s', target: 30000 },
    { duration: '180s', target: 30000 },
    { duration: '60s', target: 0 },
  ],
};

export function getStageOptions(stageName = 'stage1_50', customThresholds = {}) {
  const stages = STAGE_CONFIGS[stageName] || STAGE_CONFIGS.stage1_50;
  return {
    scenarios: {
      progressive_load: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: stages,
        gracefulRampDown: '15s',
      },
    },
    thresholds: Object.assign(
      {
        http_req_failed: ['rate<0.02'],
        http_req_duration: ['p(95)<1500', 'p(99)<3000'],
      },
      customThresholds
    ),
  };
}
