/**
 * Forensic Timeline Multi-Line Chart.
 * Renders per-window manipulation probabilities for visual (CH-V), acoustic (CH-A),
 * and fused (AVT²-DWF) streams with synchronized interactive cursor and threshold bands.
 */

import React, { useRef, useCallback, useState, useMemo } from 'react';
import { clsx } from 'clsx';
import { Activity, Clock } from 'lucide-react';
import { useCursorSync } from '../../hooks/useCursorSync';
import type { TimelinePoint } from '../../types/api';

export interface TimelineChartProps {
  timeline: TimelinePoint[];
  duration_s?: number;
  className?: string;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({
  timeline,
  duration_s,
  className,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();
  const [hoveredPoint, setHoveredPoint] = useState<TimelinePoint | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const points = useMemo(
    () =>
      timeline.length > 0
        ? timeline
        : [
            { t: 0, p_face: 0, p_audio: 0, p_fused: 0 },
            { t: 1, p_face: 0, p_audio: 0, p_fused: 0 },
          ],
    [timeline],
  );

  const maxT = duration_s && duration_s > 0 ? duration_s : points[points.length - 1].t || 1.0;

  const avgFused = useMemo(() => {
    if (points.length === 0) return 0;
    const sum = points.reduce((acc, p) => acc + p.p_fused, 0);
    return sum / points.length;
  }, [points]);

  const width = 800;
  const height = 200;
  const padLeft = 46;
  const padRight = 24;
  const padTop = 24;
  const padBottom = 34;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (t: number) => padLeft + (t / maxT) * chartW;
  const getY = (p: number) => padTop + (1 - Math.max(0, Math.min(1, p))) * chartH;

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

      // Find nearest point
      let nearest = points[0];
      let minDiff = Infinity;
      for (const p of points) {
        const diff = Math.abs(p.t - t);
        if (diff < minDiff) {
          minDiff = diff;
          nearest = p;
        }
      }
      setHoveredPoint(nearest);
    },
    [maxT, width, padLeft, chartW, points, setCurrentTime],
  );

  const activePoint =
    hoveredPoint ||
    (currentTime !== null
      ? points.reduce((prev, curr) =>
          Math.abs(curr.t - currentTime) < Math.abs(prev.t - currentTime) ? curr : prev,
        )
      : points[points.length - 1]);

  return (
    <div
      role="region"
      aria-label="Multimodal Temporal Forensic Timeline"
      className={clsx(
        'flex flex-col gap-4 p-6 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Accessible screen reader summary of timeline data */}
      <div className="sr-only">
        Forensic temporal timeline with {points.length} sample points from 0 to {maxT.toFixed(1)} seconds. Average fused synthetic probability is {(avgFused * 100).toFixed(1)}%. Current cursor at {activePoint.t.toFixed(2)} seconds with fused probability {(activePoint.p_fused * 100).toFixed(1)}%, visual probability {(activePoint.p_face * 100).toFixed(1)}%, and acoustic probability {(activePoint.p_audio * 100).toFixed(1)}%.
      </div>

      {/* Header & Legend */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-[var(--border-subtle)] gap-2">
        <div className="flex items-center gap-2">
          <Activity size={15} className="text-[var(--cobalt-400)]" />
          <span className="font-sans font-bold text-sm text-[var(--text-100)]">
            Temporal Anomaly Reconstruction
          </span>
          <span className="font-mono text-[10px] text-[var(--text-400)]">
            (P_fake over Time)
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 rounded-full bg-[var(--cobalt-500)] inline-block shadow-xs" />
            <span className="text-[var(--text-200)] font-semibold">AVT² FUSED</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-0.5 border-b border-dashed border-[var(--cobalt-300)] inline-block" />
            <span className="text-[var(--text-400)]">CH-V FACE</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 rounded-full bg-[var(--lime-400)] inline-block" />
            <span className="text-[var(--lime-400)] font-semibold">CH-A AUDIO</span>
          </div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="relative w-full overflow-hidden select-none">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto cursor-crosshair"
          onPointerDown={handlePointer}
          onPointerMove={(e) => {
            if (e.buttons === 1 || e.pressure > 0) handlePointer(e);
          }}
          onPointerLeave={() => setHoveredPoint(null)}
        >
          {/* Background threshold regions */}
          {/* Synthetic region (>= 0.65) */}
          <rect
            x={padLeft}
            y={padTop}
            width={chartW}
            height={getY(0.65) - padTop}
            fill="#FF4D5E"
            fillOpacity="0.04"
          />
          {/* Ambiguity region (0.35 - 0.65) */}
          <rect
            x={padLeft}
            y={getY(0.65)}
            width={chartW}
            height={getY(0.35) - getY(0.65)}
            fill="#8BA3FF"
            fillOpacity="0.02"
          />
          {/* Authentic region (<= 0.35) */}
          <rect
            x={padLeft}
            y={getY(0.35)}
            width={chartW}
            height={height - padBottom - getY(0.35)}
            fill="#C8FF2E"
            fillOpacity="0.03"
          />

          {/* Threshold Lines */}
          <line
            x1={padLeft}
            y1={getY(0.65)}
            x2={width - padRight}
            y2={getY(0.65)}
            stroke="#FF4D5E"
            strokeDasharray="4 4"
            strokeOpacity="0.35"
          />
          <text
            x={padLeft - 6}
            y={getY(0.65) + 3}
            fill="#FF4D5E"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
            opacity="0.8"
          >
            0.65 FAKE
          </text>

          <line
            x1={padLeft}
            y1={getY(0.35)}
            x2={width - padRight}
            y2={getY(0.35)}
            stroke="#C8FF2E"
            strokeDasharray="4 4"
            strokeOpacity="0.35"
          />
          <text
            x={padLeft - 6}
            y={getY(0.35) + 3}
            fill="#C8FF2E"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
            opacity="0.8"
          >
            0.35 AUTH
          </text>

          {/* Axis borders */}
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
          <path
            d={facePath}
            fill="none"
            stroke="#8BA3FF"
            strokeWidth="1.5"
            strokeDasharray="4 3"
            strokeOpacity="0.8"
          />
          <path
            d={audioPath}
            fill="none"
            stroke="#C8FF2E"
            strokeWidth="1.75"
            strokeOpacity="0.9"
          />
          <path
            d={fusedPath}
            fill="none"
            stroke="#1F4DFF"
            strokeWidth="2.75"
            strokeLinecap="round"
          />

          {/* Shared Time Cursor */}
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
                cy={getY(activePoint?.p_fused ?? 0)}
                r="4"
                fill="#1F4DFF"
                stroke="#C8FF2E"
                strokeWidth="1.5"
              />
            </g>
          )}

          {/* X Axis time marks */}
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

      {/* Scrubbed Time Readout HUD */}
      {activePoint && (
        <div className="flex flex-wrap items-center justify-between p-2.5 rounded-[var(--radius-sm)] bg-[var(--surface-elevated)] border border-[var(--border-subtle)] font-mono text-xs text-[var(--text-300)]">
          <div className="flex items-center gap-1.5">
            <Clock size={13} className="text-[var(--cobalt-400)]" />
            <span className="text-[var(--text-400)]">Timeline Cursor:</span>
            <span className="font-bold text-[var(--text-100)]">
              {(currentTime ?? activePoint.t).toFixed(2)}s
            </span>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-400)]">Fused P:</span>
              <span className="font-bold text-[var(--cobalt-300)]">
                {activePoint.p_fused.toFixed(3)}
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-400)]">CH-V:</span>
              <span className="font-semibold text-[var(--text-200)]">
                {activePoint.p_face.toFixed(3)}
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-400)]">CH-A:</span>
              <span className="font-semibold text-[var(--lime-400)]">
                {activePoint.p_audio.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
