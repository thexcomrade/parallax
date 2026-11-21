// Shared motion constants. Durations stay inside the 120–320 ms band defined
// in docs/design/DESIGN.md; components must not invent their own timings.

/** Instrument-panel easing: fast-out, slow-in. */
export const EASE_INSTRUMENT = [0.2, 0.8, 0.2, 1] as const;

export const DUR = {
  /** Micro-interactions: hover, focus, icon swap. */
  xs: 0.12,
  /** Standard element transitions. */
  sm: 0.18,
  /** Route change (opacity + 4px translate). */
  route: 0.24,
} as const;
