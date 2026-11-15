/**
 * Forensic Timeline Multi-Line Chart.
 * Renders per-second fake probabilities for face, audio and fused modalities with shared cursor.
 */

import React, { useRef, useCallback } from 'react';
import { useCursorSync } from '../../hooks/useCursorSync';
import type { TimelinePoint } from '../../types/api';

interface TimelineChartProps {
  timeline: TimelinePoint[];
  duration_s?: number;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({
  timeline,
  duration_s,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();
  const svgRef = useRef<SVGSVGElement>(null);

  const points = timeline.length > 0 ? timeline : [
    { t: 0, p_face: 0, p_audio: 0, p_fused: 0 },
    { t: 1, p_face: 0, p_audio: 0, p_fused: 0 },
  ];

  const maxT = duration_s && duration_s > 0 ? duration_s : points[points.length - 1].t || 1.0;

  const width = 800;
  const height = 180;
  const padLeft = 40;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (t: number) => padLeft + (t / maxT) * chartW;
  const getY = (p: number) => padTop + (1 - p) * chartH;

  // Build path strings
  const fusedPath = points
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.t).toFixed(1)} ${getY(pt.p_fused).toFixed(1)}`)
    .join(' ');

  const facePath = points
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.t).toFixed(1)} ${getY(pt.p_face).toFixed(1)}`)
    .join(' ');

  const audioPath = points
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.t).toFixed(1)} ${getY(pt.p_audio).toFixed(1)}`)
    .join(' ');

  const handlePointer = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const relX = (clientX / rect.width) * width;
      const t = Math.max(0, Math.min(maxT, ((relX - padLeft) / chartW) * maxT));
      setCurrentTime(t);
    },
    [maxT, width, padLeft, chartW, setCurrentTime],
  );

  return (
    <div className="flex flex-col gap-2 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-1 border-b border-line text-xs font-mono">
        <span className="text-cobalt-300 uppercase tracking-widest">
          TIMELINE RECONSTRUCTION // MULTIMODAL
        </span>
        <div className="flex items-center gap-4 text-[11px]">
          <span className="flex items-center gap-1.5 text-text-300">
            <span className="w-3 h-0.5 bg-cobalt-400 inline-block" /> FUSED
          </span>
          <span className="flex items-center gap-1.5 text-text-400">
            <span className="w-3 h-0.5 border-b border-dashed border-cobalt-300 inline-block" /> FACE
          </span>
          <span className="flex items-center gap-1.5 text-lime-400">
            <span className="w-3 h-0.5 bg-lime-400 inline-block" /> AUDIO
          </span>
        </div>
      </div>

      <div className="relative w-full overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto cursor-crosshair select-none"
          onPointerDown={handlePointer}
          onPointerMove={(e) => {
            if (e.buttons === 1) handlePointer(e);
          }}
        >
          {/* Threshold lines: 0.65 FAKE and 0.35 AUTHENTIC */}
          <line
            x1={padLeft}
            y1={getY(0.65)}
            x2={width - padRight}
            y2={getY(0.65)}
            stroke="#FF4D5E"
            strokeDasharray="4 4"
            strokeOpacity="0.4"
          />
          <text
            x={padLeft - 6}
            y={getY(0.65) + 3}
            fill="#FF4D5E"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
            opacity="0.7"
          >
            0.65
          </text>

          <line
            x1={padLeft}
            y1={getY(0.35)}
            x2={width - padRight}
            y2={getY(0.35)}
            stroke="#C8FF2E"
            strokeDasharray="4 4"
            strokeOpacity="0.4"
          />
          <text
            x={padLeft - 6}
            y={getY(0.35) + 3}
            fill="#C8FF2E"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
            opacity="0.7"
          >
            0.35
          </text>

          {/* Grid bounds */}
          <line
            x1={padLeft}
            y1={height - padBottom}
            x2={width - padRight}
            y2={height - padBottom}
            stroke="#1B2231"
          />
          <line
            x1={padLeft}
            y1={padTop}
            x2={padLeft}
            y2={height - padBottom}
            stroke="#1B2231"
          />

          {/* Data paths */}
          <path d={facePath} fill="none" stroke="#8BA3FF" strokeWidth="1.5" strokeDasharray="3 3" />
          <path d={audioPath} fill="none" stroke="#C8FF2E" strokeWidth="1.5" />
          <path d={fusedPath} fill="none" stroke="#1F4DFF" strokeWidth="2.5" />

          {/* Shared time cursor line */}
          {currentTime !== null && (
            <g>
              <line
                x1={getX(currentTime)}
                y1={padTop}
                x2={getX(currentTime)}
                y2={height - padBottom}
                stroke="#C8FF2E"
                strokeWidth="1.5"
              />
              <circle
                cx={getX(currentTime)}
                cy={padTop}
                r="3"
                fill="#C8FF2E"
              />
            </g>
          )}

          {/* Time axis marks */}
          <text
            x={padLeft}
            y={height - padBottom + 16}
            fill="#6B778F"
            fontSize="10"
            fontFamily="monospace"
          >
            0.0s
          </text>
          <text
            x={width - padRight}
            y={height - padBottom + 16}
            fill="#6B778F"
            fontSize="10"
            fontFamily="monospace"
            textAnchor="end"
          >
            {maxT.toFixed(1)}s
          </text>
        </svg>
      </div>
    </div>
  );
};
