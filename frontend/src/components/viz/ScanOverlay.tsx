/**
 * Forensic Scan Overlay inspector.
 * Renders high-resolution attention heatmap with opacity slider and scanline effect.
 */

import React, { useState } from 'react';

interface ScanOverlayProps {
  analysisId: string;
  artifactName: string | null;
}

export const ScanOverlay: React.FC<ScanOverlayProps> = ({
  analysisId,
  artifactName,
}) => {
  const [opacity, setOpacity] = useState(0.85);

  if (!artifactName) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-black-900 border border-line rounded-lg text-text-500 font-mono text-xs min-h-[300px]">
        SELECT AN EVIDENCE FRAME TO INSPECT ATTENTION ROLLOUT
      </div>
    );
  }

  const imageUrl = `/api/v1/analyses/${analysisId}/artifacts/${artifactName}`;

  return (
    <div className="flex flex-col gap-3 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-2 border-b border-line text-xs font-mono">
        <span className="text-cobalt-300 uppercase tracking-widest">
          ATTENTION ROLLOUT // {artifactName}
        </span>
        <div className="flex items-center gap-2">
          <label htmlFor="heatmap-opacity" className="text-text-400 text-[11px] uppercase">
            HEATMAP INTENSITY
          </label>
          <input
            id="heatmap-opacity"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-24 accent-lime-400"
          />
          <span className="text-lime-400 text-[11px] w-8 text-right">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      </div>

      <div className="relative w-full max-h-[420px] bg-black-950 rounded border border-line flex items-center justify-center overflow-hidden">
        {/* Corner viewfinder marks */}
        <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-lime-400 z-10" />
        <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-lime-400 z-10" />
        <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-lime-400 z-10" />
        <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-lime-400 z-10" />

        <img
          src={imageUrl}
          alt={`Attention heatmap for ${artifactName}`}
          className="max-h-[420px] w-auto object-contain transition-opacity duration-150"
          style={{ opacity }}
        />
      </div>

      <div className="flex items-center justify-between font-mono text-[11px] text-text-500 pt-1">
        <span>INFERNO COLORMAP: RED/YELLOW = PEAK SUSPICION ANOMALY</span>
        <span>XAI // ATTENTION ROLLOUT</span>
      </div>
    </div>
  );
};
