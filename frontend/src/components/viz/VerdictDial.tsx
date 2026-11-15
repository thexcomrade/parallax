/**
 * Verdict dial and confidence display.
 * Implements verdict reveal with confidence score and hazard-striping for FAKE.
 */

import React, { useEffect, useState } from 'react';
import type { Verdict } from '../../types/api';

interface VerdictDialProps {
  verdict: Verdict | null;
  p_fake: number | null;
  model_mode?: 'trained' | 'untrained';
}

export const VerdictDial: React.FC<VerdictDialProps> = ({
  verdict,
  p_fake,
  model_mode = 'untrained',
}) => {
  const [displayScore, setDisplayScore] = useState(0);

  // Confidence is p_fake if FAKE, (1 - p_fake) if AUTHENTIC
  const targetConfidence =
    p_fake !== null
      ? verdict === 'AUTHENTIC'
        ? Math.round((1 - p_fake) * 100)
        : Math.round(p_fake * 100)
      : 0;

  useEffect(() => {
    const start = 0;
    const duration = 600; // ms
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // ease-out
      const current = Math.round(start + (targetConfidence - start) * (1 - Math.pow(1 - progress, 3)));
      setDisplayScore(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [targetConfidence]);

  const isFake = verdict === 'FAKE';
  const isAuthentic = verdict === 'AUTHENTIC';
  const isInconclusive = verdict === 'INCONCLUSIVE';

  let verdictColor = 'text-text-400 border-line';
  let badgeBg = 'bg-black-800';

  if (isFake) {
    verdictColor = 'text-alert-500 border-alert-500';
    badgeBg = 'bg-alert-500/10';
  } else if (isAuthentic) {
    verdictColor = 'text-lime-400 border-lime-400';
    badgeBg = 'bg-lime-400/10';
  } else if (isInconclusive) {
    verdictColor = 'text-cobalt-300 border-cobalt-500';
    badgeBg = 'bg-cobalt-500/10';
  }

  return (
    <div className="relative flex flex-col items-center justify-center p-6 bg-black-900 border border-line rounded-lg overflow-hidden">
      {/* Untrained warning banner if applicable */}
      {model_mode === 'untrained' && (
        <div className="w-full bg-black-800 border-b border-alert-500/30 px-3 py-1.5 text-center mb-4 rounded">
          <span className="font-mono text-[11px] text-alert-500 uppercase tracking-wider font-semibold">
            WARNING // Untrained weights — scores are not meaningful
          </span>
        </div>
      )}

      {/* Hazard-stripe accent if FAKE */}
      {isFake && (
        <div
          className="absolute top-0 left-0 right-0 h-1 bg-[repeating-linear-gradient(45deg,#FF4D5E,#FF4D5E_8px,#04060B_8px,#04060B_16px)]"
          aria-hidden="true"
        />
      )}

      <div className="font-mono text-xs uppercase tracking-widest text-cobalt-300 mb-2">
        VERDICT ENGINE // CALIBRATED
      </div>

      {/* Primary verdict display with animated font-stretch styling */}
      <div
        className={`font-display text-4xl sm:text-5xl font-extrabold tracking-tight uppercase my-2 px-4 py-2 rounded border ${verdictColor} ${badgeBg}`}
        style={{ fontStretch: '110%' }}
      >
        {verdict ?? 'ANALYZING'}
      </div>

      {/* Confidence score readout */}
      <div className="flex items-baseline gap-2 mt-3 font-mono">
        <span className="text-3xl font-bold text-text-100">{displayScore}%</span>
        <span className="text-xs uppercase tracking-wider text-text-400">
          {isAuthentic ? 'AUTHENTIC CONFIDENCE' : isFake ? 'SYNTHETIC CONFIDENCE' : 'UNCERTAINTY MARGIN'}
        </span>
      </div>

      <div className="text-[11px] font-mono text-text-500 mt-4 text-center max-w-sm">
        Probability reflects model confidence on this clip, not forensic proof.
      </div>
    </div>
  );
};
