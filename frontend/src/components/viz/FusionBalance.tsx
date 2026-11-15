/**
 * Multimodal Fusion Balance (Dynamic Weight Fusion).
 * Visualizes the learned seesaw balance between Audio (lime) and Video (cobalt).
 */

import React from 'react';

interface FusionBalanceProps {
  w_audio: number | null;
  w_video: number | null;
  sync_score?: number | null;
  audio_present?: boolean;
}

export const FusionBalance: React.FC<FusionBalanceProps> = ({
  w_audio,
  w_video,
  sync_score,
  audio_present = true,
}) => {
  const wa = w_audio ?? 0.5;
  const wv = w_video ?? 0.5;

  const audioPercent = Math.round(wa * 100);
  const videoPercent = Math.round(wv * 100);

  return (
    <div className="flex flex-col gap-3 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-2 border-b border-line">
        <span className="font-mono text-xs uppercase tracking-widest text-cobalt-300">
          FUSION BALANCE // DWF
        </span>
        {sync_score !== undefined && sync_score !== null && (
          <span className="font-mono text-xs text-lime-400">
            SYNC {(sync_score).toFixed(2)}
          </span>
        )}
      </div>

      {/* Seesaw split bar */}
      <div className="flex flex-col gap-1.5">
        <div className="h-4 w-full bg-black-800 rounded overflow-hidden flex border border-line">
          {/* Audio section: Acid lime */}
          <div
            className="h-full bg-lime-400 transition-all duration-500 ease-out"
            style={{ width: `${audioPercent}%` }}
            title={`Audio weight: ${wa.toFixed(2)}`}
          />
          {/* Video section: Electric cobalt */}
          <div
            className="h-full bg-cobalt-500 transition-all duration-500 ease-out"
            style={{ width: `${videoPercent}%` }}
            title={`Video weight: ${wv.toFixed(2)}`}
          />
        </div>

        {/* Labels below bar */}
        <div className="flex items-center justify-between font-mono text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-lime-400 inline-block" />
            <span className="text-text-300">AUDIO W_A</span>
            <span className="font-bold text-lime-400">{wa.toFixed(2)}</span>
            {!audio_present && (
              <span className="text-[10px] text-text-500 uppercase ml-1">(MASKED)</span>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-cobalt-500 inline-block" />
            <span className="text-text-300">VIDEO W_V</span>
            <span className="font-bold text-cobalt-300">{wv.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
