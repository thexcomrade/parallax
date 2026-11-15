// Shared motion constants for all animated components.
// Every duration/easing value must come from here — never inline.

import { useReducedMotion } from 'motion/react';

/** Instrument-panel easing: fast-out-slow-in */
export const EASE_INSTRUMENT = [0.2, 0.8, 0.2, 1] as const;

/** Spring-like deceleration for data reveals */
export const EASE_DECEL = [0, 0, 0.2, 1] as const;

/** Fast acceleration for exits */
export const EASE_ACCEL = [0.4, 0, 1, 1] as const;

export const DUR = {
  /** Micro-interactions: icon swap, state toggle */
  xs: 0.12,
  /** Short transitions: hover, focus rings */
  sm: 0.18,
  /** Standard element transitions */
  md: 0.24,
  /** Route scanline wipe */
  route: 0.24,
  /** Verdict reveal, data instrument settle */
  lg: 0.32,
  /** SignalStage slow scan */
  xl: 0.4,
} as const;

/** Motion variants that respect prefers-reduced-motion.
 *  Pass `reducedMotion` from the hook to conditionally override. */
export function fadeSlide(
  reducedMotion: boolean,
  direction: 'up' | 'down' | 'left' | 'right' = 'up',
) {
  const offset = 16;
  const axis = direction === 'up' || direction === 'down' ? 'y' : 'x';
  const sign = direction === 'up' || direction === 'left' ? 1 : -1;

  return {
    hidden: reducedMotion
      ? { opacity: 0 }
      : { opacity: 0, [axis]: offset * sign },
    visible: reducedMotion
      ? { opacity: 1 }
      : { opacity: 1, [axis]: 0 },
    exit: reducedMotion
      ? { opacity: 0 }
      : { opacity: 0, [axis]: -offset * sign },
  };
}

/** Route-wipe transition variant (scanline-style opacity sweep) */
export const routeTransition = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: {
    duration: DUR.route,
    ease: EASE_INSTRUMENT,
  },
};

/** Hook: returns true when user prefers reduced motion */
export function useMotionPreference(): boolean {
  const reduced = useReducedMotion();
  return reduced ?? false;
}
