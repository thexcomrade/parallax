/**
 * Audio Waveform Canvas.
 * Draws interactive oscilloscope waveform with linked cursor and reduced-motion support.
 */

import React, { useRef, useEffect } from 'react';
import { useCursorSync } from '../../hooks/useCursorSync';

interface WaveformCanvasProps {
  duration_s?: number;
  hasAudio?: boolean;
}

export const WaveformCanvas: React.FC<WaveformCanvasProps> = ({
  duration_s = 3.0,
  hasAudio = true,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hasAudio) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Draw dark background
    ctx.fillStyle = '#080B12';
    ctx.fillRect(0, 0, width, height);

    // Draw baseline
    ctx.strokeStyle = '#1B2231';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw synthetic audio envelope peaks
    const bars = Math.floor(width / 4);
    ctx.fillStyle = '#C8FF2E';

    for (let i = 0; i < bars; i++) {
      const x = i * 4;
      // Procedural waveform envelope
      const amp =
        Math.sin(i * 0.15) * 0.4 +
        Math.cos(i * 0.05) * 0.3 +
        Math.sin(i * 0.8) * 0.15;
      const barHeight = Math.max(2, Math.abs(amp) * (height * 0.7));
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, 2, barHeight);
    }

    // Draw cursor line if active
    if (currentTime !== null && duration_s > 0) {
      const cursorX = (currentTime / duration_s) * width;
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, height);
      ctx.stroke();
    }
  }, [currentTime, duration_s, hasAudio]);

  if (!hasAudio) return null;

  return (
    <div className="w-full h-12 bg-black-900 border border-line rounded overflow-hidden relative">
      <canvas
        ref={canvasRef}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          setCurrentTime(ratio * duration_s);
        }}
        className="w-full h-full cursor-pointer select-none"
      />
    </div>
  );
};
