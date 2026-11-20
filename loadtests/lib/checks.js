import { check } from 'k6';
import { requestErrors } from './metrics.js';

export function checkStatus(response, expectedStatus = 200, name = 'status check') {
  const isOk = check(response, {
    [`${name} is ${expectedStatus}`]: (r) => r.status === expectedStatus,
  });

  if (!isOk) {
    requestErrors.add(1);
  } else {
    requestErrors.add(0);
  }

  return isOk;
}

export function checkPublicVerdict(response) {
  const isOk = check(response, {
    'public verdict returns 200': (r) => r.status === 200,
    'prediction is REAL or FAKE': (r) => {
      try {
        const body = r.json();
        return body && (body.prediction === 'REAL' || body.prediction === 'FAKE');
      } catch (e) {
        return false;
      }
    },
    'no internal p_fake exposed': (r) => {
      try {
        const body = r.json();
        return body && body.p_fake === undefined && body.p_authentic === undefined;
      } catch (e) {
        return false;
      }
    },
    'no internal dwf_weights exposed': (r) => {
      try {
        const body = r.json();
        return body && body.dwf_weights === undefined && body.timeline === undefined;
      } catch (e) {
        return false;
      }
    },
  });

  if (!isOk) {
    requestErrors.add(1);
  } else {
    requestErrors.add(0);
  }

  return isOk;
}
