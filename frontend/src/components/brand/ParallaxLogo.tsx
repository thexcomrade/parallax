import React, { useId } from 'react';

export interface ParallaxLogoProps {
  size?: number;
  className?: string;
  title?: string;
}

/**
 * PARALLAX Forensic Identity Logo
 * 
 * Precision dual-stream geometric mark:
 * - Left Prism Facet: Visual stream (Electric Cobalt gradient var(--cobalt-300) -> var(--cobalt-500))
 * - Right Harmonic Facet: Acoustic stream (Cyan harmonic gradient #38c6df -> #0891b2)
 * - Center Coherence Nexus: Dynamic Weight Fusion (DWF) core
 */
export const ParallaxLogo: React.FC<ParallaxLogoProps> = ({
  size = 24,
  className = '',
  title = 'PARALLAX Forensic System',
}) => {
  const rawId = useId();
  const id = rawId.replace(/[:]/g, '');
  const vGradId = `plx-v-grad-${id}`;
  const aGradId = `plx-a-grad-${id}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label={title}
      role="img"
    >
      <defs>
        <linearGradient id={vGradId} x1="6" y1="8" x2="16" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="var(--cobalt-300)" />
          <stop offset="100%" stopColor="var(--cobalt-500)" />
        </linearGradient>
        <linearGradient id={aGradId} x1="26" y1="8" x2="16" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#38c6df" />
          <stop offset="100%" stopColor="#0891b2" />
        </linearGradient>
      </defs>

      {/* Visual Stream: Angular Optic Prism Facet */}
      <path
        d="M7 11.5L16 6L16 14.5L9.5 18L7 11.5Z"
        fill={`url(#${vGradId})`}
      />
      <path
        d="M7 11.5L16 6L16 14.5L9.5 18Z"
        stroke="#c2d0ff"
        strokeWidth="0.75"
        strokeLinejoin="round"
        strokeOpacity="0.6"
      />

      {/* Acoustic Stream: Harmonic Waveform Facet */}
      <path
        d="M25 20.5L16 26L16 17.5L22.5 14L25 20.5Z"
        fill={`url(#${aGradId})`}
      />
      <path
        d="M25 20.5L16 26L16 17.5L22.5 14Z"
        stroke="#38c6df"
        strokeWidth="0.75"
        strokeLinejoin="round"
        strokeOpacity="0.6"
      />

      {/* Dynamic Weight Fusion Alignment Pulses */}
      <path
        d="M11 16H13.5M18.5 16H21"
        stroke="#eef2fa"
        strokeWidth="1.2"
        strokeLinecap="round"
      />

      {/* Central Coherence Nexus */}
      <circle cx="16" cy="16" r="2.2" fill="#FFFFFF" />
      <circle
        cx="16"
        cy="16"
        r="3.4"
        stroke="#38c6df"
        strokeWidth="0.8"
        strokeOpacity="0.7"
      />
    </svg>
  );
};
