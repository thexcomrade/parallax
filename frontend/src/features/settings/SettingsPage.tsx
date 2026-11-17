/**
 * SettingsPage — Operator Credentials, Active Cryptographic Sessions & Privacy Lifecycle.
 *
 * Implements Phase 13/15: PARALLAX SETTINGS PAGE REDESIGN
 * - Modern dark SaaS settings workspace (#090D14 / #111927 / #171F2D / #253247)
 * - 2-column desktop layout (Left navigation ~210px + Right content panel)
 * - Mobile responsive horizontal tab/pill navigation bar
 * - Operator Identity & authenticated profile telemetry (zero third-party CDN leaks)
 * - Cryptographic Master Password update with real-time feedback
 * - Global active session revocation across all browsers and devices
 * - High-friction permanent account & forensic archive purge confirmation dialog
 * - Consistent sans application typography matching Dashboard, History, and Library
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { api } from '@/lib/api';
import { Dialog } from '@/components/ui/Dialog';
import { Toast } from '@/components/ui/Toast';
import { formatRelativeTime } from '@/lib/format';
import {
  User,
  KeyRound,
  ShieldAlert,
  LogOut,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Lock,
  Fingerprint,
  Eye,
  EyeOff,
  Loader2,
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Active navigation section
  const [activeSection, setActiveSection] = useState<'account' | 'security' | 'sessions' | 'danger'>('account');

  // Password state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPass, setShowCurrentPass] = useState(false);
  const [showNewPass, setShowNewPass] = useState(false);
  const [showConfirmPass, setShowConfirmPass] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null);

  // Dialog & Action states
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [revokeLoading, setRevokeLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const scrollToSection = (id: 'account' | 'security' | 'sessions' | 'danger') => {
    setActiveSection(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError(null);
    setPasswordSuccess(null);

    if (newPassword.length < 12) {
      setPasswordError('New password must be at least 12 characters.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('Password confirmation does not match.');
      return;
    }

    setPasswordLoading(true);
    try {
      await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      const successMsg = 'Password successfully updated. All other active sessions have been revoked.';
      setPasswordSuccess(successMsg);
      setToastMsg(successMsg);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to update operator password';
      setPasswordError(msg);
    } finally {
      setPasswordLoading(false);
    }
  };

  const handleRevokeAllSessions = async () => {
    setRevokeLoading(true);
    try {
      await api.post('/auth/revoke-all');
      await logout();
      navigate('/login?message=All+sessions+revoked');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to revoke active sessions';
      setPasswordError(msg);
      setRevokeLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    setDeleteLoading(true);
    try {
      await api.delete('/auth/account');
      await logout();
      navigate('/login?message=Account+purged');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to purge operator account';
      setPasswordError(msg);
      setDeleteLoading(false);
      setDeleteModalOpen(false);
    }
  };

  // Compute initials for zero-leak offline avatar
  const emailInitials = user?.email
    ? user.email.slice(0, 2).toUpperCase()
    : 'OP';

  return (
    <div className="flex flex-col gap-6 w-full max-w-7xl mx-auto py-1 sm:py-2 text-[#E8EDF5]">
      {/* Toast Alert */}
      {toastMsg && (
        <Toast
          message={toastMsg}
          variant="success"
          onDismiss={() => setToastMsg(null)}
        />
      )}

      {/* ── 1. Page Header (Matches Dashboard, History & Library) ──────────── */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#253247]/60">
        <div>
          <span className="text-[11px] font-semibold tracking-wider text-[#648CFF] uppercase block">
            PREFERENCES
          </span>
          <h1 className="text-2xl sm:text-[28px] font-semibold text-[#E8EDF5] tracking-tight mt-0.5">
            Settings <span className="sr-only">Analyst Account & Security</span>
          </h1>
          <p className="text-[13px] sm:text-sm text-[#91A0B8] mt-1 leading-normal">
            Manage your account and application preferences.
          </p>
        </div>
      </header>

      {/* ── 2. Mobile Horizontal Navigation Bar (< md) ───────────────────── */}
      <nav
        aria-label="Settings mobile navigation"
        tabIndex={0}
        className="flex md:hidden items-center gap-1.5 p-1 bg-[#111927] border border-[#253247] rounded-xl overflow-x-auto w-full text-xs focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
      >
        <button
          type="button"
          onClick={() => scrollToSection('account')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'account'
              ? 'bg-[#1C2737] text-white border border-[#253247]'
              : 'text-[#91A0B8] hover:text-[#E8EDF5]'
          }`}
        >
          Account
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('security')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'security'
              ? 'bg-[#1C2737] text-white border border-[#253247]'
              : 'text-[#91A0B8] hover:text-[#E8EDF5]'
          }`}
        >
          Password
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('sessions')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'sessions'
              ? 'bg-[#1C2737] text-white border border-[#253247]'
              : 'text-[#91A0B8] hover:text-[#E8EDF5]'
          }`}
        >
          Sessions
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('danger')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'danger'
              ? 'bg-red-500/15 text-red-300 border border-red-500/30'
              : 'text-red-400 hover:text-red-300'
          }`}
        >
          Danger Zone
        </button>
      </nav>

      {/* ── 3. Main Layout: 2-Column Desktop Grid ─────────────────────────── */}
      <div className="w-full flex flex-col md:flex-row gap-6 lg:gap-8 items-start">
        {/* Left Sidebar Navigation (Desktop) */}
        <nav
          aria-label="Settings navigation"
          className="w-[210px] shrink-0 hidden md:flex flex-col gap-1 sticky top-6 self-start p-1.5 bg-[#111927] border border-[#253247] rounded-xl"
        >
          <button
            type="button"
            onClick={() => scrollToSection('account')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'account'
                ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                : 'text-[#91A0B8] hover:text-[#E8EDF5] hover:bg-[#171F2D]/50'
            }`}
          >
            <User className="w-4 h-4 text-[#648CFF]" />
            <span>Account</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('security')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'security'
                ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                : 'text-[#91A0B8] hover:text-[#E8EDF5] hover:bg-[#171F2D]/50'
            }`}
          >
            <KeyRound className="w-4 h-4 text-[#648CFF]" />
            <span>Password</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('sessions')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'sessions'
                ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                : 'text-[#91A0B8] hover:text-[#E8EDF5] hover:bg-[#171F2D]/50'
            }`}
          >
            <Fingerprint className="w-4 h-4 text-[#648CFF]" />
            <span>Active Sessions</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('danger')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'danger'
                ? 'bg-red-500/15 text-red-300 border border-red-500/30 shadow-xs'
                : 'text-red-400 hover:text-red-300 hover:bg-red-500/10'
            }`}
          >
            <AlertTriangle className="w-4 h-4 text-[#F16F78]" />
            <span>Danger Zone</span>
          </button>
        </nav>

        {/* Right Settings Content Area */}
        <div className="flex-1 min-w-0 flex flex-col gap-6 w-full">
          {/* ── Section A: Operator Identity ───────────────────────────────── */}
          <section
            id="account"
            aria-labelledby="operator-identity-heading"
            className="rounded-[14px] bg-[#171F2D] border border-[#253247] p-5 sm:p-6 flex flex-col gap-5 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[#253247]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[#1C2737] border border-[#253247] flex items-center justify-center text-[#648CFF]">
                  <User className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="operator-identity-heading"
                    className="text-base font-semibold text-[#E8EDF5]"
                  >
                    Operator Identity
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-0.5">
                    Authenticated analyst profile and workspace parameters.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#35C99A]/10 text-[#35C99A] border border-[#35C99A]/20">
                Session Active
              </span>
            </div>

            {/* Profile Row */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl bg-[#111927] border border-[#253247]">
              <div className="flex items-center gap-3.5">
                <div
                  className="w-11 h-11 rounded-full bg-[#1C2737] border-2 border-[#648CFF]/60 flex items-center justify-center font-bold text-sm text-[#648CFF] select-none shrink-0"
                  aria-label={`Avatar for operator ${user?.email || 'analyst'}`}
                >
                  {emailInitials}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-[#E8EDF5]">
                    {user?.email || 'analyst@parallax.io'}
                  </span>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-[#91A0B8] mt-0.5">
                    <span>Analyst ID: #{user?.id || 1}</span>
                    <span>•</span>
                    <span>
                      Enrolled: {user?.created_at ? formatRelativeTime(user.created_at) : 'Recent'}
                    </span>
                  </div>
                </div>
              </div>

              <span className="px-2.5 py-1 rounded-md bg-[#171F2D] border border-[#253247] text-xs font-medium text-[#91A0B8]">
                Full Multi-Modal Access
              </span>
            </div>

            {/* Read-Only Environment Parameters */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-[#111927] rounded-lg border border-[#253247]/60 flex items-center justify-between">
                <span className="text-[#91A0B8]">Authentication Token:</span>
                <span className="text-[#E8EDF5] font-medium">Argon2id + JWT HS256</span>
              </div>
              <div className="p-3 bg-[#111927] rounded-lg border border-[#253247]/60 flex items-center justify-between">
                <span className="text-[#91A0B8]">Tenant Isolation:</span>
                <span className="text-[#35C99A] font-medium">SQLite Multi-Tenant Filter</span>
              </div>
              <div className="p-3 bg-[#111927] rounded-lg border border-[#253247]/60 flex items-center justify-between">
                <span className="text-[#91A0B8]">Workspace Appearance:</span>
                <span className="text-[#E8EDF5] font-medium">Midnight Dark (Calibrated)</span>
              </div>
              <div className="p-3 bg-[#111927] rounded-lg border border-[#253247]/60 flex items-center justify-between">
                <span className="text-[#91A0B8]">Network Boundary:</span>
                <span className="text-[#35C99A] font-medium">Zero Third-Party Ingestion</span>
              </div>
            </div>
          </section>

          {/* ── Section B: Authentication Credentials ──────────────────────── */}
          <section
            id="security"
            aria-labelledby="auth-phrase-heading"
            className="rounded-[14px] bg-[#171F2D] border border-[#253247] p-5 sm:p-6 flex flex-col gap-5 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[#253247]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[#1C2737] border border-[#253247] flex items-center justify-center text-[#648CFF]">
                  <KeyRound className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="auth-phrase-heading"
                    className="text-base font-semibold text-[#E8EDF5]"
                  >
                    Authentication Phrase
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-0.5">
                    Update master cryptographic password used for workspace authentication.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#1C2737] text-[#91A0B8] border border-[#253247]">
                Argon2id Key Derivation
              </span>
            </div>

            <form onSubmit={handlePasswordChange} className="flex flex-col gap-4 max-w-lg">
              {passwordError && (
                <div
                  role="alert"
                  className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-xs text-red-300 flex items-center gap-2"
                >
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{passwordError}</span>
                </div>
              )}

              {passwordSuccess && (
                <div
                  role="status"
                  className="p-3 rounded-lg bg-[#35C99A]/10 border border-[#35C99A]/25 text-xs text-[#35C99A] flex items-center gap-2"
                >
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>{passwordSuccess}</span>
                </div>
              )}

              {/* Current Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="current-password"
                  className="text-xs font-medium text-[#91A0B8]"
                >
                  Current Master Password
                </label>
                <div className="relative flex items-center w-full">
                  <input
                    id="current-password"
                    type={showCurrentPass ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    required
                    autoComplete="current-password"
                    placeholder="••••••••••••"
                    className="w-full h-[38px] px-3.5 pr-10 bg-[#090D14] border border-[#253247] hover:border-[#374866] focus:border-[#648CFF] focus:outline-none focus:ring-1 focus:ring-[#648CFF] rounded-lg text-[13px] text-[#E8EDF5] placeholder-[#91A0B8]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPass(!showCurrentPass)}
                    aria-label={showCurrentPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[#91A0B8] hover:text-[#E8EDF5] transition-colors cursor-pointer"
                  >
                    {showCurrentPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* New Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="new-password"
                  className="text-xs font-medium text-[#91A0B8]"
                >
                  New Master Password
                </label>
                <div className="relative flex items-center w-full">
                  <input
                    id="new-password"
                    type={showNewPass ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                    placeholder="••••••••••••"
                    className="w-full h-[38px] px-3.5 pr-10 bg-[#090D14] border border-[#253247] hover:border-[#374866] focus:border-[#648CFF] focus:outline-none focus:ring-1 focus:ring-[#648CFF] rounded-lg text-[13px] text-[#E8EDF5] placeholder-[#91A0B8]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPass(!showNewPass)}
                    aria-label={showNewPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[#91A0B8] hover:text-[#E8EDF5] transition-colors cursor-pointer"
                  >
                    {showNewPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <span className="text-[11px] text-[#91A0B8]">
                  Minimum 12 characters with letters, numbers, and symbols.
                </span>
              </div>

              {/* Confirm New Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="confirm-password"
                  className="text-xs font-medium text-[#91A0B8]"
                >
                  Confirm New Password
                </label>
                <div className="relative flex items-center w-full">
                  <input
                    id="confirm-password"
                    type={showConfirmPass ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                    placeholder="••••••••••••"
                    className="w-full h-[38px] px-3.5 pr-10 bg-[#090D14] border border-[#253247] hover:border-[#374866] focus:border-[#648CFF] focus:outline-none focus:ring-1 focus:ring-[#648CFF] rounded-lg text-[13px] text-[#E8EDF5] placeholder-[#91A0B8]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPass(!showConfirmPass)}
                    aria-label={showConfirmPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[#91A0B8] hover:text-[#E8EDF5] transition-colors cursor-pointer"
                  >
                    {showConfirmPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <div className="pt-1">
                <button
                  type="submit"
                  disabled={passwordLoading}
                  className="h-[38px] px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white text-xs font-semibold inline-flex items-center gap-2 transition-colors shadow-xs cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
                >
                  {passwordLoading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Updating...</span>
                    </>
                  ) : (
                    <>
                      <Lock className="w-3.5 h-3.5" />
                      <span>Update Master Credentials</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </section>

          {/* ── Section C: Session Security ─────────────────────────────────── */}
          <section
            id="sessions"
            aria-labelledby="session-security-heading"
            className="rounded-[14px] bg-[#171F2D] border border-[#253247] p-5 sm:p-6 flex flex-col gap-4 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[#253247]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[#1C2737] border border-[#253247] flex items-center justify-center text-[#648CFF]">
                  <Fingerprint className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="session-security-heading"
                    className="text-base font-semibold text-[#E8EDF5]"
                  >
                    Session Lifecycle
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-0.5">
                    Rotating cryptographic refresh tokens and cross-device sessions.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#1C2737] text-[#91A0B8] border border-[#253247]">
                Active Token Vault
              </span>
            </div>

            <div className="p-4 bg-[#111927] border border-[#253247] rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1 max-w-lg">
                <span className="text-sm font-semibold text-[#E8EDF5]">
                  Revoke All Active Sessions
                </span>
                <span className="text-xs text-[#91A0B8] leading-relaxed">
                  Sign out across all browsers and devices immediately. All outstanding refresh tokens will be revoked in the backend token registry.
                </span>
              </div>

              <button
                type="button"
                onClick={handleRevokeAllSessions}
                disabled={revokeLoading}
                className="h-[36px] px-3.5 rounded-lg bg-[#1C2737] hover:bg-[#253247] active:bg-[#111927] border border-[#253247] text-[#E8EDF5] text-xs font-semibold inline-flex items-center gap-2 transition-colors shrink-0 cursor-pointer disabled:opacity-50 shadow-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
              >
                {revokeLoading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Revoking...</span>
                  </>
                ) : (
                  <>
                    <LogOut className="w-3.5 h-3.5" />
                    <span>Sign Out Everywhere</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ── Section D: Danger Zone ──────────────────────────────────────── */}
          <section
            id="danger"
            aria-labelledby="danger-zone-heading"
            className="rounded-[14px] bg-[#171F2D] border border-red-500/30 p-5 sm:p-6 flex flex-col gap-4 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[#253247]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center justify-center text-[#F16F78]">
                  <ShieldAlert className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="danger-zone-heading"
                    className="text-base font-semibold text-[#F16F78]"
                  >
                    Danger Zone
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-0.5">
                    Permanent purge of operator record and forensic evidence.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-red-500/10 text-[#F16F78] border border-red-500/20">
                Irreversible
              </span>
            </div>

            <div className="p-4 bg-red-950/20 border border-red-500/25 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1 max-w-lg">
                <span className="text-sm font-semibold text-[#F16F78]">
                  Permanently Purge Operator Account & Analyses
                </span>
                <span className="text-xs text-[#91A0B8] leading-relaxed">
                  Irreversibly delete operator identity, all forensic analysis dossiers, audio-visual attention rollout heatmaps, and uploaded media artifacts from disk storage.
                </span>
              </div>

              <button
                type="button"
                onClick={() => setDeleteModalOpen(true)}
                className="h-[36px] px-3.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 active:bg-red-500/30 border border-red-500/40 text-[#F16F78] text-xs font-semibold inline-flex items-center gap-2 transition-colors shrink-0 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Purge Account & All Data</span>
              </button>
            </div>
          </section>
        </div>
      </div>

      {/* Confirmation Dialog */}
      <Dialog
        open={deleteModalOpen}
        title="PERMANENTLY PURGE OPERATOR ACCOUNT?"
        description="This action cannot be undone. All video scans, extracted keyframes, spectrogram tensors, attention heatmaps, and account credentials will be permanently erased from database storage."
        confirmLabel="YES, PERMANENTLY PURGE"
        cancelLabel="ABORT"
        variant="danger"
        loading={deleteLoading}
        onConfirm={handleDeleteAccount}
        onClose={() => setDeleteModalOpen(false)}
      />
    </div>
  );
};
