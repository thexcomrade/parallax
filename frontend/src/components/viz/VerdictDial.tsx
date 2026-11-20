/**
 * Verdict dial and confidence display.
 * Implements verdict reveal with confidence score, hazard-striping for FAKE,
 * probability explanation, and accessible WCAG contrast.
 */

import React, { useEffect, useState } from 'react';
import { clsx } from 'clsx';
import { CheckCircle2, AlertTriangle, HelpCircle, ShieldCheck, Info } from 'lucide-react';
import type { Verdict } from '../../types/api';

export interface VerdictDialProps {
  verdict: Verdict | null;
  p_fake: number | null;
  model_mode?: 'trained' | 'untrained';
  className?: string;
}

export const VerdictDial: React.FC<VerdictDialProps> = ({
  verdict,
  p_fake,
  model_mode = 'untrained',
  className,
}) => {
  const [displayScore, setDisplayScore] = useState(0);

  const isFake = verdict === 'FAKE';
  const isAuthentic = verdict === 'AUTHENTIC' || (verdict as string) === 'REAL';
  const isInconclusive = verdict === 'INCONCLUSIVE';

  // Confidence is p_fake if FAKE, (1 - p_fake) if AUTHENTIC or REAL, or distance from boundary if INCONCLUSIVE
  const targetConfidence =
    p_fake !== null
      ? isAuthentic
        ? Math.round((1 - p_fake) * 100)
        : isFake
        ? Math.round(p_fake * 100)
        : Math.round((1 - Math.abs(p_fake - 0.5) * 2) * 100)
      : 0;

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      setDisplayScore(targetConfidence);
      return;
    }

    const start = 0;
    const duration = 400;
    const startTime = performance.now();
    let frameId: number;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const current = Math.round(start + (targetConfidence - start) * (1 - Math.pow(1 - progress, 3)));
      setDisplayScore(current);

      if (progress < 1) {
        frameId = requestAnimationFrame(animate);
      }
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [targetConfidence]);

  return (
    <div
      role="region"
      aria-label="Forensic Classification Verdict"
      className={clsx(
        'relative flex flex-col p-6 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)] overflow-hidden',
        className,
      )}
    >
      {/* Hazard-stripe accent if FAKE (WCAG redundant visual coding) */}
      {isFake && (
        <div
          className="absolute top-0 left-0 right-0 h-1.5 bg-[repeating-linear-gradient(45deg,#FF4D5E,#FF4D5E_8px,#04060B_8px,#04060B_16px)]"
          aria-hidden="true"
        />
      )}

      {/* Header section with Model Mode Tag */}
      <div className="flex items-center justify-between pb-3 border-b border-[var(--border-subtle)]">
        <span className="font-mono text-xs uppercase tracking-wider text-[var(--cobalt-300)] flex items-center gap-1.5">
          <ShieldCheck size={14} className="text-[var(--cobalt-400)]" />
          Verdict Engine
        </span>

        <span
          className={clsx(
            'font-mono text-[10px] px-2 py-0.5 rounded font-semibold uppercase tracking-wider',
            model_mode === 'trained'
              ? 'bg-[var(--lime-400)]/10 text-[var(--lime-400)] border border-[var(--lime-400)]/30'
              : 'bg-[var(--alert-bg)] text-[var(--alert-500)] border border-[var(--alert-500)]/30',
          )}
        >
          {model_mode === 'trained' ? 'Calibrated Model' : 'Untrained Weights'}
        </span>
      </div>

      {/* Untrained mode disclaimer banner */}
      {model_mode === 'untrained' && (
        <div className="mt-3 p-2.5 rounded-[var(--radius-sm)] bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 flex items-start gap-2 text-xs font-mono text-[var(--alert-500)]">
          <AlertTriangle size={14} className="shrink-0 mt-0.5" />
          <span className="leading-snug">
            Research Mode: Model is using uncalibrated weights. Scores indicate feature variance rather than verified synthetic probability.
          </span>
        </div>
      )}

      {/* Verdict readout card */}
      <div className="flex flex-col items-center justify-center my-5 py-4 px-6 rounded-[var(--radius-md)] bg-[var(--surface-elevated)] border border-[var(--border-subtle)] text-center">
        <div className="flex items-center gap-2 mb-2">
          {isAuthentic ? (
            <CheckCircle2 size={24} className="text-[var(--lime-400)]" />
          ) : isFake ? (
            <AlertTriangle size={24} className="text-[var(--alert-500)]" />
          ) : (
            <HelpCircle size={24} className="text-[var(--cobalt-300)]" />
          )}

          <span
            className={clsx(
              'font-display text-4xl sm:text-5xl font-extrabold tracking-tight uppercase',
              isAuthentic
                ? 'text-[var(--lime-400)]'
                : isFake
                ? 'text-[var(--alert-500)]'
                : isInconclusive
                ? 'text-[var(--cobalt-300)]'
                : 'text-[var(--text-300)]',
            )}
          >
            {verdict ?? 'EVALUATING'}
          </span>
        </div>

        {/* Confidence metric */}
        <div className="flex items-baseline gap-2 font-mono mt-1">
          <span className="font-display text-3xl font-bold text-[var(--text-100)]">
            {displayScore}%
          </span>
          <span className="text-xs font-semibold uppercase tracking-wider text-[var(--text-300)]">
            {isAuthentic
              ? 'Authentic Confidence'
              : isFake
              ? 'Synthetic Confidence'
              : 'Ambiguity Margin'}
          </span>
        </div>

        <p className="font-sans text-xs text-[var(--text-400)] mt-2 text-center max-w-sm">
          {isAuthentic
            ? 'Cross-modal biometric and spectral features exhibit organic coherence consistent with genuine media capture.'
            : isFake
            ? 'Elevated spatial attention anomalies or temporal audio-visual phase incoherence detected.'
            : 'Statistical evidence falls within the inconclusive threshold (0.35 ≤ p ≤ 0.65). Additional verification recommended.'}
        </p>
      </div>

      {/* Technical Probability Threshold Breakdown */}
      <div className="flex flex-col gap-2 pt-3 border-t border-[var(--border-subtle)] text-xs font-mono">
        <div className="flex items-center justify-between text-[var(--text-300)]">
          <span className="text-[var(--text-400)]">Posterior P(Fake):</span>
          <span className="font-semibold text-[var(--text-100)]">
            {p_fake !== null ? p_fake.toFixed(4) : '—'}
          </span>
        </div>

        <div className="flex items-center justify-between text-[var(--text-300)]">
          <span className="text-[var(--text-400)]">Decision Thresholds:</span>
          <span className="text-[11px] text-[var(--text-400)]">
            &lt;0.35 Authentic · &gt;0.65 Fake
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] text-[var(--text-400)] mt-1 pt-2 border-t border-[var(--border-subtle)]/50">
          <Info size={12} className="text-[var(--cobalt-400)] shrink-0" />
          <span>Probability reflects model confidence on this media, not legal forensic proof.</span>
        </div>
      </div>
    </div>
  );
};
