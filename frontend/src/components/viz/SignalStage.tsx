import React, { useEffect, useRef } from 'react';
import { useMotionPreference } from '@/lib/motion';

interface Point3D {
  x: number;
  y: number;
  z: number;
}

export const SignalStage: React.FC<{ className?: string }> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const reducedMotion = useMotionPreference();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let isTabVisible = !document.hidden;

    // Generate procedural face point-cloud (oval cylinder / ellipsoidal mesh)
    const points: Point3D[] = [];
    const layers = 18;
    const pointsPerLayer = 26;

    for (let i = 0; i < layers; i++) {
      const v = (i / (layers - 1)) * 2 - 1; // -1 to 1 (vertical)
      const radiusAtY = Math.sqrt(Math.max(0, 1 - v * v * 0.75)) * 0.72;
      for (let j = 0; j < pointsPerLayer; j++) {
        const u = (j / pointsPerLayer) * Math.PI * 2;
        // Subtle facial features deformation (nose bridge + eye indent)
        let r = radiusAtY;
        const frontFacing = Math.cos(u);
        if (frontFacing > 0.4 && Math.abs(v) < 0.35) {
          // Eye sockets indent & nose ridge
          if (Math.abs(v) < 0.15 && Math.abs(Math.sin(u)) < 0.2) {
            r += 0.08; // nose tip
          } else if (Math.abs(v) > 0.1 && Math.abs(v) < 0.3 && Math.abs(Math.sin(u)) > 0.15) {
            r -= 0.05; // eye socket
          }
        }
        points.push({
          x: Math.sin(u) * r * 0.75,
          y: v * 0.9,
          z: Math.cos(u) * r,
        });
      }
    }

    let scanY = 0;
    let angle = 0;
    let wavePhase = 0;
    let isSyntheticMode = false;
    let modeTimer = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;

      ctx.fillStyle = '#04060B';
      ctx.fillRect(0, 0, width, height);

      // Grid background markings (hairline coordinates)
      ctx.strokeStyle = '#161D2B';
      ctx.lineWidth = 1;
      const step = 40;
      for (let x = 0; x < width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Optical face center
      const centerX = width * 0.5;
      const centerY = height * 0.42;
      const scale = Math.min(width, height) * 0.34;

      // Rotate and project point-cloud
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Draw points
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        // Rotate around Y
        const rx = p.x * cosA - p.z * sinA;
        const rz = p.x * sinA + p.z * cosA;
        const ry = p.y;

        // Perspective projection
        const cameraDist = 2.4;
        const perspective = cameraDist / (cameraDist + rz);
        const projX = centerX + rx * scale * perspective;
        const projY = centerY + ry * scale * perspective;

        // Distance from scanline
        const distToScan = Math.abs(projY - scanY);
        const isNearScan = distToScan < 22;

        if (isNearScan) {
          ctx.fillStyle = '#C8FF2E'; // Acid lime when excited by scan
          ctx.beginPath();
          ctx.arc(projX, projY, 1.8 * perspective, 0, Math.PI * 2);
          ctx.fill();
        } else if (rz > -0.2) {
          // Front points
          ctx.fillStyle = '#1F4DFF'; // Electric cobalt
          ctx.beginPath();
          ctx.arc(projX, projY, 1.2 * perspective, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // Back points (darker)
          ctx.fillStyle = '#2A3550';
          ctx.beginPath();
          ctx.arc(projX, projY, 0.8 * perspective, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Scanline beam
      const scanGradient = ctx.createLinearGradient(0, scanY - 14, 0, scanY + 14);
      scanGradient.addColorStop(0, 'rgba(200, 255, 46, 0)');
      scanGradient.addColorStop(0.5, 'rgba(200, 255, 46, 0.85)');
      scanGradient.addColorStop(1, 'rgba(200, 255, 46, 0)');

      ctx.fillStyle = scanGradient;
      ctx.fillRect(width * 0.1, scanY - 12, width * 0.8, 24);

      ctx.strokeStyle = '#C8FF2E';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(width * 0.1, scanY);
      ctx.lineTo(width * 0.9, scanY);
      ctx.stroke();

      // Lower waveform channel (audio modality visualizer)
      const waveY = height * 0.84;
      const waveHeight = height * 0.08;
      const waveWidth = width * 0.84;
      const waveStart = width * 0.08;

      ctx.strokeStyle = '#1B2231';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(waveStart, waveY);
      ctx.lineTo(waveStart + waveWidth, waveY);
      ctx.stroke();

      // Waveform line
      ctx.beginPath();
      ctx.strokeStyle = isSyntheticMode ? '#FF4D5E' : '#C8FF2E';
      ctx.lineWidth = 1.5;

      const segments = 120;
      for (let s = 0; s <= segments; s++) {
        const norm = s / segments;
        const x = waveStart + norm * waveWidth;
        let yOffset: number;

        if (isSyntheticMode) {
          // Jittered, artificial harmonic mismatch
          const jitter = Math.sin(norm * 40 + wavePhase * 3) * (Math.random() * 0.35 + 0.65);
          yOffset = Math.sin(norm * 14 + wavePhase) * Math.cos(norm * 7) * jitter * waveHeight;
        } else {
          // Coherent harmonic acoustic wave
          yOffset =
            (Math.sin(norm * 10 + wavePhase) * 0.6 + Math.sin(norm * 22 + wavePhase * 1.5) * 0.4) *
            Math.sin(norm * Math.PI) *
            waveHeight;
        }

        if (s === 0) {
          ctx.moveTo(x, waveY + yOffset);
        } else {
          ctx.lineTo(x, waveY + yOffset);
        }
      }
      ctx.stroke();

      // Lab Telemetry Labels
      ctx.font = '10px "Martian Mono", monospace';
      ctx.fillStyle = '#7886A0';
      ctx.fillText('MOD: OPTICAL_FLUX [V-CH]', width * 0.08, height * 0.08);
      ctx.fillText('SIG: 25.0 FPS / 224x224', width * 0.08, height * 0.11);

      ctx.fillStyle = isSyntheticMode ? '#FF4D5E' : '#C8FF2E';
      ctx.fillText(
        isSyntheticMode ? 'AUD_CH: JITTER / PHASE_ANOMALY' : 'AUD_CH: HARMONIC_STABLE',
        width * 0.08,
        height * 0.77
      );

      ctx.fillStyle = '#7886A0';
      ctx.textAlign = 'right';
      ctx.fillText('SYNC: 0.88', width * 0.92, height * 0.08);
      ctx.fillText('AVT²-DWF FORENSICS', width * 0.92, height * 0.11);
      ctx.textAlign = 'left';

      // Updates for next frame
      if (!reducedMotion) {
        angle += 0.008;
        scanY += 1.8;
        if (scanY > height * 0.7) {
          scanY = height * 0.14;
        }
        wavePhase += 0.06;

        modeTimer++;
        if (modeTimer > 240) {
          modeTimer = 0;
          isSyntheticMode = !isSyntheticMode;
        }
      }
    };

    const handleResize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.scale(dpr, dpr);
      scanY = canvas.height / dpr * 0.2;
      render();
    };

    const handleVisibilityChange = () => {
      isTabVisible = !document.hidden;
      if (isTabVisible && !reducedMotion) {
        loop();
      }
    };

    const loop = () => {
      if (!isTabVisible) return;
      render();
      if (!reducedMotion) {
        animationFrameId = requestAnimationFrame(loop);
      }
    };

    window.addEventListener('resize', handleResize);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    handleResize();

    if (!reducedMotion) {
      loop();
    } else {
      render();
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      cancelAnimationFrame(animationFrameId);
    };
  }, [reducedMotion]);

  return (
    <div className={`relative w-full h-full min-h-[360px] overflow-hidden bg-[var(--black-950)] border-r border-[var(--line)] ${className || ''}`}>
      <canvas
        ref={canvasRef}
        aria-label="Procedural face point-cloud with optical scanline and audio harmonic waveform"
        role="img"
        className="w-full h-full block"
      />
      {/* Corner Crop Marks for Lab Aesthetic */}
      <div className="absolute top-3 left-3 w-3 h-3 border-t-2 border-l-2 border-[var(--cobalt-500)] pointer-events-none" />
      <div className="absolute top-3 right-3 w-3 h-3 border-t-2 border-r-2 border-[var(--cobalt-500)] pointer-events-none" />
      <div className="absolute bottom-3 left-3 w-3 h-3 border-b-2 border-l-2 border-[var(--cobalt-500)] pointer-events-none" />
      <div className="absolute bottom-3 right-3 w-3 h-3 border-b-2 border-r-2 border-[var(--cobalt-500)] pointer-events-none" />
    </div>
  );
};
