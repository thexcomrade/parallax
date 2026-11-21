import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { api } from '@/lib/api';
import type { ModelCardResponse, ReadyResponse } from '@/types/api';
import { useAuth } from '@/hooks/useAuth';
import { ParallaxLogo } from '@/components/brand/ParallaxLogo';
import {
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
      className="w-full h-14 md:h-[72px] bg-[var(--surface-card)] border-b border-[var(--line)] flex items-center z-30 shrink-0 select-none relative px-4 md:px-8"
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
              className="w-9 h-9 inline-flex items-center justify-center rounded-lg text-[var(--text-300)] hover:text-[var(--text-100)] bg-[var(--black-950)] border border-[var(--line)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)]"
            >
              {isMobileNavOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
          )}

          <Link
            to="/app"
            className="flex items-center gap-2.5 focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] rounded-lg"
            title="PARALLAX Dashboard"
          >
            <div className="w-8 h-8 rounded-lg bg-[var(--black-950)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] shadow-xs shrink-0">
              <ParallaxLogo size={20} />
            </div>
            <span className="font-bold text-sm tracking-wider text-[var(--text-100)] uppercase">
              PARALLAX
            </span>
          </Link>
        </div>

        {/* ── LEFT SECTION — CLEAN MINIMAL BREADCRUMB (Desktop) ── */}
        <nav
          aria-label="Global breadcrumb"
          className="hidden md:flex items-center gap-2.5 text-[13px] font-sans"
        >
          <span className="text-[var(--text-300)] font-normal tracking-wide">
            {context.section}
          </span>
          <span className="text-[var(--line-strong)] select-none text-[12px]" aria-hidden="true">
            /
          </span>
          <span
            className="text-[var(--text-100)] font-medium tracking-tight"
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
              className="h-10 inline-flex items-center gap-2 px-3 rounded-[9px] bg-[var(--black-950)] hover:bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] text-[var(--text-100)] text-[13px] font-medium transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] cursor-pointer"
              aria-expanded={telemetryOpen}
              aria-label={isDegraded ? 'System Degraded diagnostics' : 'System Operational diagnostics'}
            >
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  isDegraded ? 'bg-[var(--alert-500)]' : 'bg-[var(--success-500)]'
                }`}
                aria-hidden="true"
              />
              <span className="hidden sm:inline font-sans text-[13px] text-[var(--text-100)] font-medium leading-none">
                {isDegraded ? 'Degraded' : 'Operational'}
              </span>
              <ChevronDown
                size={14}
                className={`text-[var(--text-300)] shrink-0 transition-transform duration-150 ${
                  telemetryOpen ? 'rotate-180' : ''
                }`}
              />
            </button>

            {/* Diagnostics Popover */}
            {telemetryOpen && (
              <div
                role="dialog"
                aria-label="System Runtime Diagnostics"
                className="absolute right-0 mt-2 w-[min(288px,calc(100vw-2rem))] rounded-[var(--radius-cell)] bg-[var(--surface-overlay)] border border-[var(--line-strong)] shadow-[var(--shadow-overlay)] p-4 text-xs font-sans flex flex-col gap-3 z-50 panel-pop"
              >
                <div className="flex items-center justify-between border-b border-[var(--line)] pb-2.5">
                  <span className="font-semibold text-[var(--text-100)] flex items-center gap-2">
                    <Activity size={15} className="text-[var(--cobalt-300)]" />
                    System Diagnostics
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--black-950)] border border-[var(--line)] font-mono text-[var(--success-500)]">
                    LIVE
                  </span>
                </div>

                <div className="flex flex-col gap-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--text-300)] flex items-center gap-1.5">
                      <Server size={13} className="text-[var(--text-300)]" />
                      API Gateway:
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        isDegraded
                          ? 'bg-[var(--alert-500)]/15 text-[var(--alert-500)] border border-[var(--alert-500)]/30'
                          : 'bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-current" />
                      {isDegraded ? 'Degraded' : 'Online'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[var(--text-300)] flex items-center gap-1.5">
                      <Database size={13} className="text-[var(--text-300)]" />
                      Forensic Database:
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        readyData?.db_connected
                          ? 'bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30'
                          : 'bg-[var(--alert-500)]/15 text-[var(--alert-500)] border border-[var(--alert-500)]/30'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-current" />
                      {readyData?.db_connected ? 'Connected' : 'Offline'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[var(--text-300)] flex items-center gap-1.5">
                      <Zap size={13} className="text-[var(--text-300)]" />
                      Checkpoint:
                    </span>
                    <span className="font-mono text-[11px] text-[var(--text-100)]">
                      {modelData ? `${modelData.model_name} ${modelData.model_version}` : 'AVT²-DWF v1.0'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between pt-1 border-t border-[var(--line)]/60">
                    <span className="text-[var(--text-300)]">Weights Mode:</span>
                    <span
                      className={`text-[11px] font-medium px-2 py-0.5 rounded ${
                        isUntrained
                          ? 'bg-[var(--warning-500)]/15 text-[var(--warning-500)] border border-[var(--warning-500)]/30'
                          : 'bg-[var(--cobalt-300)]/15 text-[var(--cobalt-300)] border border-[var(--cobalt-300)]/30'
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
            className="inline-flex items-center justify-center gap-2 h-10 px-3.5 sm:px-4 rounded-[9px] bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] font-semibold text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] shadow-xs cursor-pointer leading-none shrink-0"
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
                className="h-10 inline-flex items-center gap-2.5 px-2.5 sm:px-3 rounded-[9px] bg-[var(--black-950)] hover:bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] cursor-pointer"
                aria-expanded={userMenuOpen}
                aria-label={`Open operator profile options: ${user.email}`}
              >
                <div className="w-7 h-7 rounded-[7px] bg-[var(--cobalt-300)]/15 border border-[var(--cobalt-300)]/25 flex items-center justify-center text-[var(--cobalt-300)] shrink-0">
                  <User size={14} />
                </div>
                <span className="font-sans text-[13px] font-medium text-[var(--text-100)] hidden md:inline truncate max-w-[130px] lg:max-w-[180px] xl:max-w-[220px] leading-none">
                  {user.email}
                </span>
                <ChevronDown
                  size={14}
                  className={`text-[var(--text-300)] shrink-0 transition-transform duration-150 ${
                    userMenuOpen ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {userMenuOpen && (
                <div
                  role="menu"
                  aria-label="Account Menu"
                  className="absolute right-0 mt-2 w-64 rounded-[var(--radius-cell)] bg-[var(--surface-overlay)] border border-[var(--line-strong)] shadow-[var(--shadow-overlay)] p-1.5 text-xs font-sans flex flex-col gap-1 z-50 panel-pop"
                >
                  {/* User Identity Header */}
                  <div className="px-3 py-2.5 border-b border-[var(--line)] flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-[var(--cobalt-300)]/15 border border-[var(--cobalt-300)]/30 flex items-center justify-center text-[var(--cobalt-300)] font-semibold text-xs shrink-0">
                      {userInitials}
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="text-[10px] uppercase font-mono text-[var(--text-300)] leading-none">
                        Authenticated Analyst
                      </span>
                      <span className="font-medium text-[var(--text-100)] truncate text-xs mt-1 leading-tight">
                        {user.email}
                      </span>
                    </div>
                  </div>

                  {/* Account Settings Link */}
                  <Link
                    to="/app/settings"
                    role="menuitem"
                    onClick={() => setUserMenuOpen(false)}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[var(--text-100)] hover:bg-[var(--black-950)] transition-colors"
                  >
                    <Settings size={15} className="text-[var(--text-300)]" />
                    <span>Account Settings</span>
                  </Link>

                  {/* Divider */}
                  <div className="h-px bg-[var(--line)] my-0.5" />

                  {/* Sign Out Action */}
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setUserMenuOpen(false);
                      logout();
                    }}
                    disabled={isLoggingOut}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[var(--alert-500)] hover:bg-[var(--alert-500)]/10 transition-colors w-full text-left cursor-pointer disabled:opacity-50"
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
