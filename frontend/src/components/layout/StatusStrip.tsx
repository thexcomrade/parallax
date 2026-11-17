import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { api } from '@/lib/api';
import type { ModelCardResponse, ReadyResponse } from '@/types/api';
import { useAuth } from '@/hooks/useAuth';
import {
  Shield,
  Plus,
  LogOut,
  Settings,
  ChevronDown,
  Activity,
  Menu,
  X,
  Database,
  Server,
  Zap,
  User,
} from 'lucide-react';

export interface StatusStripProps {
  onToggleMobileNav?: () => void;
  isMobileNavOpen?: boolean;
  collapsed?: boolean;
}

export const StatusStrip: React.FC<StatusStripProps> = ({
  onToggleMobileNav,
  isMobileNavOpen = false,
}) => {
  const { user, logout, isLoggingOut } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const telemetryRef = useRef<HTMLDivElement>(null);

  // Dynamic context breadcrumb based on active route
  const context = useMemo(() => {
    const p = location.pathname;
    if (p.startsWith('/app/analyze')) return { section: 'Platform', title: 'Analyze' };
    if (p.startsWith('/app/history')) return { section: 'Platform', title: 'History' };
    if (p.startsWith('/app/library')) return { section: 'Platform', title: 'Library' };
    if (p.startsWith('/app/intel')) return { section: 'Platform', title: 'Threat Intelligence' };
    if (p.startsWith('/app/settings')) return { section: 'Platform', title: 'Settings' };
    if (p.includes('/report/') || p.startsWith('/app/analyses/')) return { section: 'Platform', title: 'Forensic Report' };
    return { section: 'Platform', title: 'Dashboard' };
  }, [location.pathname]);

  // Actual backend system health queries
  const { data: readyData } = useQuery<ReadyResponse>({
    queryKey: ['system-ready'],
    queryFn: () => api.get<ReadyResponse>('/ready'),
    refetchInterval: 15_000,
  });

  const { data: modelData } = useQuery<ModelCardResponse>({
    queryKey: ['system-model'],
    queryFn: () => api.get<ModelCardResponse>('/model'),
    staleTime: 60_000,
  });

  const isDegraded = readyData?.status === 'degraded';
  const isUntrained = modelData?.model_mode === 'untrained';

  // Extract operator initials for avatar inside dropdown
  const userInitials = useMemo(() => {
    if (!user?.email) return 'AN';
    const namePart = user.email.split('@')[0] || '';
    if (namePart.length >= 2) {
      return namePart.slice(0, 2).toUpperCase();
    }
    return namePart.toUpperCase() || 'AN';
  }, [user?.email]);

  // Handle outside click and Escape key dismissal
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
      if (telemetryRef.current && !telemetryRef.current.contains(e.target as Node)) {
        setTelemetryOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setUserMenuOpen(false);
        setTelemetryOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <header
      className="w-full h-14 md:h-[72px] bg-[#0D131E] border-b border-[#253247] flex items-center z-30 shrink-0 select-none relative px-4 md:px-8"
      aria-label="Global Application Header"
    >
      <div className="w-full max-w-[1600px] mx-auto flex items-center justify-between gap-4">
        {/* ── LEFT SECTION — MOBILE BRAND & HAMBURGER (Visible on mobile only) ── */}
        <div className="md:hidden flex items-center gap-3">
          {onToggleMobileNav && (
            <button
              type="button"
              onClick={onToggleMobileNav}
              aria-label={isMobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
              className="w-9 h-9 inline-flex items-center justify-center rounded-lg text-[#91A0B8] hover:text-[#E8EDF5] bg-[#111827] border border-[#253247] transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF]"
            >
              {isMobileNavOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
          )}

          <Link
            to="/app"
            className="flex items-center gap-2.5 focus-visible:outline-2 focus-visible:outline-[#648CFF] rounded-lg"
            title="PARALLAX Dashboard"
          >
            <div className="w-8 h-8 rounded-lg bg-[#111827] border border-[#253247] flex items-center justify-center text-[#648CFF] shadow-xs shrink-0">
              <Shield size={17} />
            </div>
            <span className="font-bold text-sm tracking-wider text-[#E8EDF5] uppercase">
              PARALLAX
            </span>
          </Link>
        </div>

        {/* ── LEFT SECTION — CLEAN MINIMAL BREADCRUMB (Desktop) ── */}
        <nav
          aria-label="Global breadcrumb"
          className="hidden md:flex items-center gap-2.5 text-[13px] font-sans"
        >
          <span className="text-[#91A0B8] font-normal tracking-wide">
            {context.section}
          </span>
          <span className="text-[#374866] select-none text-[12px]" aria-hidden="true">
            /
          </span>
          <span
            className="text-[#E8EDF5] font-medium tracking-tight"
            aria-current="page"
          >
            {context.title}
          </span>
        </nav>

        {/* ── RIGHT SECTION — GLOBAL ACTIONS (Operational, New Analysis, Profile) ── */}
        <div className="flex items-center gap-3 sm:gap-3.5 md:gap-4 shrink-0 ml-auto">
          {/* 1. Operational Status Control */}
          <div className="relative" ref={telemetryRef}>
            <button
              type="button"
              onClick={() => setTelemetryOpen(!telemetryOpen)}
              className="h-10 inline-flex items-center gap-2 px-3 rounded-[9px] bg-[#111927] hover:bg-[#171F2D] border border-[#253247] hover:border-[#374866] text-[#E8EDF5] text-[13px] font-medium transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF] cursor-pointer"
              aria-expanded={telemetryOpen}
              aria-label={isDegraded ? 'System Degraded diagnostics' : 'System Operational diagnostics'}
            >
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  isDegraded ? 'bg-[#F16F78]' : 'bg-[#35C99A]'
                }`}
                aria-hidden="true"
              />
              <span className="hidden sm:inline font-sans text-[13px] text-[#E8EDF5] font-medium leading-none">
                {isDegraded ? 'Degraded' : 'Operational'}
              </span>
              <ChevronDown
                size={14}
                className={`text-[#91A0B8] shrink-0 transition-transform duration-150 ${
                  telemetryOpen ? 'rotate-180' : ''
                }`}
              />
            </button>

            {/* Diagnostics Popover */}
            {telemetryOpen && (
              <div
                role="dialog"
                aria-label="System Runtime Diagnostics"
                className="absolute -right-20 sm:right-0 mt-2 w-[272px] sm:w-72 rounded-xl bg-[#171F2D] border border-[#253247] shadow-2xl p-4 text-xs font-sans flex flex-col gap-3 z-50 animate-in fade-in zoom-in-95 duration-100"
              >
                <div className="flex items-center justify-between border-b border-[#253247] pb-2.5">
                  <span className="font-semibold text-[#E8EDF5] flex items-center gap-2">
                    <Activity size={15} className="text-[#648CFF]" />
                    System Diagnostics
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#111827] border border-[#253247] font-mono text-[#35C99A]">
                    LIVE
                  </span>
                </div>

                <div className="flex flex-col gap-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[#91A0B8] flex items-center gap-1.5">
                      <Server size={13} className="text-[#91A0B8]" />
                      API Gateway:
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        isDegraded
                          ? 'bg-[#F16F78]/15 text-[#F16F78] border border-[#F16F78]/30'
                          : 'bg-[#35C99A]/15 text-[#35C99A] border border-[#35C99A]/30'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-current" />
                      {isDegraded ? 'Degraded' : 'Online'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#91A0B8] flex items-center gap-1.5">
                      <Database size={13} className="text-[#91A0B8]" />
                      Forensic Database:
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        readyData?.db_connected
                          ? 'bg-[#35C99A]/15 text-[#35C99A] border border-[#35C99A]/30'
                          : 'bg-[#F16F78]/15 text-[#F16F78] border border-[#F16F78]/30'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-current" />
                      {readyData?.db_connected ? 'Connected' : 'Offline'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#91A0B8] flex items-center gap-1.5">
                      <Zap size={13} className="text-[#91A0B8]" />
                      Checkpoint:
                    </span>
                    <span className="font-mono text-[11px] text-[#E8EDF5]">
                      {modelData ? `${modelData.model_name} ${modelData.model_version}` : 'AVT²-DWF v1.0'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between pt-1 border-t border-[#253247]/60">
                    <span className="text-[#91A0B8]">Weights Mode:</span>
                    <span
                      className={`text-[11px] font-medium px-2 py-0.5 rounded ${
                        isUntrained
                          ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                          : 'bg-[#648CFF]/15 text-[#648CFF] border border-[#648CFF]/30'
                      }`}
                    >
                      {isUntrained ? 'Untrained Benchmark' : 'Verified Weights'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 2. New Analysis Button */}
          <button
            type="button"
            onClick={() => navigate('/app/analyze')}
            className="inline-flex items-center justify-center gap-2 h-10 px-3.5 sm:px-4 rounded-[9px] bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white font-semibold text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF] shadow-xs cursor-pointer leading-none shrink-0"
            aria-label="New Analysis"
          >
            <Plus size={16} strokeWidth={2.5} className="shrink-0" />
            <span className="hidden xs:inline sm:inline">New Analysis</span>
          </button>

          {/* 3. User Profile Control & Dropdown */}
          {user && (
            <div className="relative" ref={userMenuRef}>
              <button
                type="button"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="h-10 inline-flex items-center gap-2.5 px-2.5 sm:px-3 rounded-[9px] bg-[#111827] hover:bg-[#171F2D] border border-[#253247] hover:border-[#374866] transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF] cursor-pointer"
                aria-expanded={userMenuOpen}
                aria-label={`Open operator profile options: ${user.email}`}
              >
                <div className="w-7 h-7 rounded-[7px] bg-[#648CFF]/15 border border-[#648CFF]/25 flex items-center justify-center text-[#648CFF] shrink-0">
                  <User size={14} />
                </div>
                <span className="font-sans text-[13px] font-medium text-[#E8EDF5] hidden md:inline truncate max-w-[130px] lg:max-w-[180px] xl:max-w-[220px] leading-none">
                  {user.email}
                </span>
                <ChevronDown
                  size={14}
                  className={`text-[#91A0B8] shrink-0 transition-transform duration-150 ${
                    userMenuOpen ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {userMenuOpen && (
                <div
                  role="menu"
                  aria-label="Account Menu"
                  className="absolute right-0 mt-2 w-64 rounded-xl bg-[#171F2D] border border-[#253247] shadow-2xl p-1.5 text-xs font-sans flex flex-col gap-1 z-50 animate-in fade-in zoom-in-95 duration-100"
                >
                  {/* User Identity Header */}
                  <div className="px-3 py-2.5 border-b border-[#253247] flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-[#648CFF]/15 border border-[#648CFF]/30 flex items-center justify-center text-[#648CFF] font-semibold text-xs shrink-0">
                      {userInitials}
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="text-[10px] uppercase font-mono text-[#91A0B8] leading-none">
                        Authenticated Analyst
                      </span>
                      <span className="font-medium text-[#E8EDF5] truncate text-xs mt-1 leading-tight">
                        {user.email}
                      </span>
                    </div>
                  </div>

                  {/* Account Settings Link */}
                  <Link
                    to="/app/settings"
                    role="menuitem"
                    onClick={() => setUserMenuOpen(false)}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[#E8EDF5] hover:bg-[#111827] transition-colors"
                  >
                    <Settings size={15} className="text-[#91A0B8]" />
                    <span>Account Settings</span>
                  </Link>

                  {/* Divider */}
                  <div className="h-px bg-[#253247] my-0.5" />

                  {/* Sign Out Action */}
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setUserMenuOpen(false);
                      logout();
                    }}
                    disabled={isLoggingOut}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[#F16F78] hover:bg-[#F16F78]/10 transition-colors w-full text-left cursor-pointer disabled:opacity-50"
                  >
                    <LogOut size={15} />
                    <span>Sign Out</span>
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
