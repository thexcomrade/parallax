import React from 'react';
import { BentoGrid } from '@/components/bento/BentoGrid';
import { BentoCell } from '@/components/bento/BentoCell';
import { useAuth } from '@/hooks/useAuth';
import { Link } from 'react-router-dom';
import { ScanEye, Activity, Cpu, ShieldCheck } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto">
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--lime-400)]" />
            <span className="font-mono text-xs uppercase tracking-widest text-[var(--text-500)]">
              ACTIVE SESSION // CH-FORENSICS
            </span>
          </div>
          <h1 className="font-display font-black text-2xl lg:text-3xl uppercase tracking-tight text-white mt-1">
            DETECTION BENCH
          </h1>
        </div>

        <Link
          to="/app/analyze"
          className="inline-flex items-center justify-center gap-2 min-h-[44px] px-5 rounded-[var(--radius-sm)] bg-[var(--lime-400)] text-[var(--lime-ink)] font-mono text-xs font-bold uppercase tracking-wider hover:bg-[var(--lime-300)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
        >
          <ScanEye size={16} />
          <span>NEW VIDEO SCAN</span>
        </Link>
      </div>

      {/* Main Bento Grid */}
      <BentoGrid>
        {/* Cell 1: Welcome & Telemetry */}
        <BentoCell label="OPERATOR STATUS" status="ONLINE" colSpan={8}>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded bg-[var(--black-800)] border border-[var(--cobalt-500)]/40 flex items-center justify-center text-[var(--cobalt-400)]">
                <ShieldCheck size={22} />
              </div>
              <div>
                <div className="font-mono text-sm font-semibold text-white truncate">
                  {user?.email || 'ANALYST'}
                </div>
                <div className="font-mono text-xs text-[var(--text-500)]">
                  ACCESS TIER: MULTIMODAL FORENSICS (V+A)
                </div>
              </div>
            </div>
            <p className="font-sans text-sm text-[var(--text-300)] leading-relaxed">
              AVT²-DWF dual-stream forensic pipeline is loaded and listening. Connect media stream or drop video clip to analyze optical frame tokens and acoustic harmonic consistency.
            </p>
          </div>
        </BentoCell>

        {/* Cell 2: Quick Metrics */}
        <BentoCell label="ENGINE VERDICT RATIO" status="SYNC" colSpan={4}>
          <div className="flex flex-col justify-between h-full gap-3">
            <div className="flex items-baseline gap-2">
              <span className="font-display font-black text-4xl text-[var(--lime-400)]">0</span>
              <span className="font-mono text-xs text-[var(--text-500)]">SCANS THIS SESSION</span>
            </div>
            <div className="h-2 w-full bg-[var(--black-800)] rounded-full overflow-hidden flex">
              <div className="h-full bg-[var(--lime-400)] w-0" />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-[var(--text-500)]">
              <span>AUTHENTIC [0%]</span>
              <span>FAKE [0%]</span>
            </div>
          </div>
        </BentoCell>

        {/* Cell 3: Threat Pulse Preview */}
        <BentoCell label="THREAT PULSE // INTEL" status="LIVE" colSpan={6}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-300)]">
              <Activity size={16} className="text-[var(--cobalt-400)]" />
              <span>GLOBAL SYNTHETIC THREAT ACTIVITY</span>
            </div>
            <div className="p-3 bg-[var(--black-800)] rounded-[var(--radius-sm)] border border-[var(--line)] font-mono text-xs text-[var(--text-500)]">
              System monitoring live deepfake event stream. Connect to Intel channel to view ingested feeds from arXiv and GDELT.
            </div>
          </div>
        </BentoCell>

        {/* Cell 4: Model Engine Card */}
        <BentoCell label="MODEL ARCHITECTURE" status="AVT²-DWF" colSpan={6}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-300)]">
              <Cpu size={16} className="text-[var(--lime-400)]" />
              <span>DYNAMIC WEIGHT FUSION (DWF)</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2 bg-[var(--black-800)] border border-[var(--line)] rounded">
                <span className="text-[var(--text-500)] block text-[10px]">VIDEO BACKBONE</span>
                <span className="text-white font-medium">ViT-B/16 (T=32)</span>
              </div>
              <div className="p-2 bg-[var(--black-800)] border border-[var(--line)] rounded">
                <span className="text-[var(--text-500)] block text-[10px]">AUDIO BACKBONE</span>
                <span className="text-white font-medium">MFCC + Log-Mel</span>
              </div>
            </div>
          </div>
        </BentoCell>
      </BentoGrid>
    </div>
  );
};
