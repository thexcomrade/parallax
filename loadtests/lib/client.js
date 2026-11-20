const APPROVED_HOSTS = ['staging-api', 'localhost', '127.0.0.1', 'host.docker.internal'];

export function getBaseUrl() {
  const url = __ENV.BASE_URL || 'http://staging-api:8000';
  validateTargetUrl(url);
  return url;
}

export function validateTargetUrl(url) {
  if (!url || typeof url !== 'string') {
    throw new Error('[SECURITY FATAL] BASE_URL must be specified and non-empty.');
  }

  // Refuse production domains or dangerous external hosts
  const lower = url.toLowerCase();
  if (lower.includes('prod') || lower.includes('production')) {
    throw new Error(`[SECURITY FATAL] Target URL contains 'prod' or 'production': ${url}`);
  }

  const isApproved = APPROVED_HOSTS.some(host => lower.includes(host));
  if (!isApproved) {
    throw new Error(`[SECURITY FATAL] Target URL host is not in approved staging allowlist (${APPROVED_HOSTS.join(', ')}): ${url}`);
  }

  return true;
}
