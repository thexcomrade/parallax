let videoBin = null;
try {
  // k6 init-context file open
  videoBin = open('/loadtests/fixtures/synthetic_video.mp4', 'b');
} catch (e) {
  try {
    videoBin = open('../fixtures/synthetic_video.mp4', 'b');
  } catch (err) {
    videoBin = null;
  }
}

export function getSyntheticVideo() {
  return videoBin;
}
