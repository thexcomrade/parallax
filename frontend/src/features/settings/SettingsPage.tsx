/**
 * SettingsPage — Operator Credentials, Active Cryptographic Sessions & Privacy Lifecycle.
 *
 * Implements Phase 13/15: PARALLAX SETTINGS PAGE REDESIGN
 * - Modern dark SaaS settings workspace (var(--surface-base) / var(--black-950) / var(--black-900) / var(--line))
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
    <div className="flex flex-col gap-6 w-full max-w-7xl mx-auto py-1 sm:py-2 text-[var(--text-100)]">
      {/* Toast Alert */}
      {toastMsg && (
        <Toast
          message={toastMsg}
          variant="success"
          onDismiss={() => setToastMsg(null)}
        />
      )}

      {/* ── 1. Page Header (Matches Dashboard, History & Library) ──────────── */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[var(--line)]/60">
        <div>
          <span className="text-[11px] font-semibold tracking-wider text-[var(--cobalt-300)] uppercase block">
            PREFERENCES
          </span>
          <h1 className="text-2xl sm:text-[28px] font-semibold text-[var(--text-100)] tracking-tight mt-0.5">
            Settings <span className="sr-only">Analyst Account & Security</span>
          </h1>
          <p className="text-[13px] sm:text-sm text-[var(--text-300)] mt-1 leading-normal">
            Manage your account and application preferences.
          </p>
        </div>
      </header>

      {/* ── 2. Mobile Horizontal Navigation Bar (< md) ───────────────────── */}
      <nav
        aria-label="Settings mobile navigation"
        tabIndex={0}
        className="flex md:hidden items-center gap-1.5 p-1 bg-[var(--black-950)] border border-[var(--line)] rounded-xl overflow-x-auto w-full text-xs focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--cobalt-300)]"
      >
        <button
          type="button"
          onClick={() => scrollToSection('account')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'account'
              ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)]'
              : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
          }`}
        >
          Account
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('security')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'security'
              ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)]'
              : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
          }`}
        >
          Password
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('sessions')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'sessions'
              ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)]'
              : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
          }`}
        >
          Sessions
        </button>
        <button
          type="button"
          onClick={() => scrollToSection('danger')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap cursor-pointer shrink-0 ${
            activeSection === 'danger'
              ? 'bg-[var(--alert-bg)] text-[var(--alert-400)] border border-[var(--alert-500)]/30'
              : 'text-[var(--alert-400)] hover:text-[var(--alert-400)]'
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
          className="w-[210px] shrink-0 hidden md:flex flex-col gap-1 sticky top-6 self-start p-1.5 bg-[var(--black-950)] border border-[var(--line)] rounded-xl"
        >
          <button
            type="button"
            onClick={() => scrollToSection('account')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'account'
                ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                : 'text-[var(--text-300)] hover:text-[var(--text-100)] hover:bg-[var(--black-900)]/50'
            }`}
          >
            <User className="w-4 h-4 text-[var(--cobalt-300)]" />
            <span>Account</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('security')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'security'
                ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                : 'text-[var(--text-300)] hover:text-[var(--text-100)] hover:bg-[var(--black-900)]/50'
            }`}
          >
            <KeyRound className="w-4 h-4 text-[var(--cobalt-300)]" />
            <span>Password</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('sessions')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'sessions'
                ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                : 'text-[var(--text-300)] hover:text-[var(--text-100)] hover:bg-[var(--black-900)]/50'
            }`}
          >
            <Fingerprint className="w-4 h-4 text-[var(--cobalt-300)]" />
            <span>Active Sessions</span>
          </button>
          <button
            type="button"
            onClick={() => scrollToSection('danger')}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2.5 text-left cursor-pointer ${
              activeSection === 'danger'
                ? 'bg-[var(--alert-bg)] text-[var(--alert-400)] border border-[var(--alert-500)]/30 shadow-xs'
                : 'text-[var(--alert-400)] hover:text-[var(--alert-400)] hover:bg-[var(--alert-bg)]'
            }`}
          >
            <AlertTriangle className="w-4 h-4 text-[var(--alert-500)]" />
            <span>Danger Zone</span>
          </button>
        </nav>

        {/* Right Settings Content Area */}
        <div className="flex-1 min-w-0 flex flex-col gap-6 w-full">
          {/* ── Section A: Operator Identity ───────────────────────────────── */}
          <section
            id="account"
            aria-labelledby="operator-identity-heading"
            className="rounded-[14px] bg-[var(--black-900)] border border-[var(--line)] p-5 sm:p-6 flex flex-col gap-5 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[var(--black-700)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)]">
                  <User className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="operator-identity-heading"
                    className="text-base font-semibold text-[var(--text-100)]"
                  >
                    Operator Identity
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-0.5">
                    Authenticated analyst profile and workspace parameters.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[var(--success-500)]/10 text-[var(--success-500)] border border-[var(--success-500)]/20">
                Session Active
              </span>
            </div>

            {/* Profile Row */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl bg-[var(--black-950)] border border-[var(--line)]">
              <div className="flex items-center gap-3.5">
                <div
                  className="w-11 h-11 rounded-full bg-[var(--black-700)] border-2 border-[var(--cobalt-300)]/60 flex items-center justify-center font-bold text-sm text-[var(--cobalt-300)] select-none shrink-0"
                  aria-label={`Avatar for operator ${user?.email || 'analyst'}`}
                >
                  {emailInitials}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-[var(--text-100)]">
                    {user?.email || 'analyst@parallax.io'}
                  </span>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--text-300)] mt-0.5">
                    <span>Analyst ID: #{user?.id || 1}</span>
                    <span>•</span>
                    <span>
                      Enrolled: {user?.created_at ? formatRelativeTime(user.created_at) : 'Recent'}
                    </span>
                  </div>
                </div>
              </div>

              <span className="px-2.5 py-1 rounded-md bg-[var(--black-900)] border border-[var(--line)] text-xs font-medium text-[var(--text-300)]">
                Full Multi-Modal Access
              </span>
            </div>

            {/* Read-Only Environment Parameters */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-[var(--black-950)] rounded-lg border border-[var(--line)]/60 flex items-center justify-between">
                <span className="text-[var(--text-300)]">Authentication Token:</span>
                <span className="text-[var(--text-100)] font-medium">Argon2id + JWT HS256</span>
              </div>
              <div className="p-3 bg-[var(--black-950)] rounded-lg border border-[var(--line)]/60 flex items-center justify-between">
                <span className="text-[var(--text-300)]">Tenant Isolation:</span>
                <span className="text-[var(--success-500)] font-medium">SQLite Multi-Tenant Filter</span>
              </div>
              <div className="p-3 bg-[var(--black-950)] rounded-lg border border-[var(--line)]/60 flex items-center justify-between">
                <span className="text-[var(--text-300)]">Workspace Appearance:</span>
                <span className="text-[var(--text-100)] font-medium">Midnight Dark (Calibrated)</span>
              </div>
              <div className="p-3 bg-[var(--black-950)] rounded-lg border border-[var(--line)]/60 flex items-center justify-between">
                <span className="text-[var(--text-300)]">Network Boundary:</span>
                <span className="text-[var(--success-500)] font-medium">Zero Third-Party Ingestion</span>
              </div>
            </div>
          </section>

          {/* ── Section B: Authentication Credentials ──────────────────────── */}
          <section
            id="security"
            aria-labelledby="auth-phrase-heading"
            className="rounded-[14px] bg-[var(--black-900)] border border-[var(--line)] p-5 sm:p-6 flex flex-col gap-5 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[var(--black-700)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)]">
                  <KeyRound className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="auth-phrase-heading"
                    className="text-base font-semibold text-[var(--text-100)]"
                  >
                    Authentication Phrase
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-0.5">
                    Update master cryptographic password used for workspace authentication.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[var(--black-700)] text-[var(--text-300)] border border-[var(--line)]">
                Argon2id Key Derivation
              </span>
            </div>

            <form onSubmit={handlePasswordChange} className="flex flex-col gap-4 max-w-lg">
              {passwordError && (
                <div
                  role="alert"
                  className="p-3 rounded-lg bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 text-xs text-[var(--alert-400)] flex items-center gap-2"
                >
                  <AlertTriangle className="w-4 h-4 text-[var(--alert-400)] shrink-0" />
                  <span>{passwordError}</span>
                </div>
              )}

              {passwordSuccess && (
                <div
                  role="status"
                  className="p-3 rounded-lg bg-[var(--success-500)]/10 border border-[var(--success-500)]/25 text-xs text-[var(--success-500)] flex items-center gap-2"
                >
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>{passwordSuccess}</span>
                </div>
              )}

              {/* Current Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="current-password"
                  className="text-xs font-medium text-[var(--text-300)]"
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
                    className="w-full h-[38px] px-3.5 pr-10 bg-[var(--surface-base)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-300)] focus:outline-none focus:ring-1 focus:ring-[var(--cobalt-300)] rounded-lg text-[13px] text-[var(--text-100)] placeholder-[var(--text-300)]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPass(!showCurrentPass)}
                    aria-label={showCurrentPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors cursor-pointer"
                  >
                    {showCurrentPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* New Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="new-password"
                  className="text-xs font-medium text-[var(--text-300)]"
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
                    className="w-full h-[38px] px-3.5 pr-10 bg-[var(--surface-base)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-300)] focus:outline-none focus:ring-1 focus:ring-[var(--cobalt-300)] rounded-lg text-[13px] text-[var(--text-100)] placeholder-[var(--text-300)]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPass(!showNewPass)}
                    aria-label={showNewPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors cursor-pointer"
                  >
                    {showNewPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <span className="text-[11px] text-[var(--text-300)]">
                  Minimum 12 characters with letters, numbers, and symbols.
                </span>
              </div>

              {/* Confirm New Password */}
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="confirm-password"
                  className="text-xs font-medium text-[var(--text-300)]"
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
                    className="w-full h-[38px] px-3.5 pr-10 bg-[var(--surface-base)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-300)] focus:outline-none focus:ring-1 focus:ring-[var(--cobalt-300)] rounded-lg text-[13px] text-[var(--text-100)] placeholder-[var(--text-300)]/50 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPass(!showConfirmPass)}
                    aria-label={showConfirmPass ? 'Hide password' : 'Show password'}
                    className="absolute right-2.5 p-1 text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors cursor-pointer"
                  >
                    {showConfirmPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <div className="pt-1">
                <button
                  type="submit"
                  disabled={passwordLoading}
                  className="h-[38px] px-4 rounded-lg bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] text-xs font-semibold inline-flex items-center gap-2 transition-colors shadow-xs cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)]"
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
            className="rounded-[14px] bg-[var(--black-900)] border border-[var(--line)] p-5 sm:p-6 flex flex-col gap-4 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[var(--black-700)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)]">
                  <Fingerprint className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="session-security-heading"
                    className="text-base font-semibold text-[var(--text-100)]"
                  >
                    Session Lifecycle
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-0.5">
                    Rotating cryptographic refresh tokens and cross-device sessions.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[var(--black-700)] text-[var(--text-300)] border border-[var(--line)]">
                Active Token Vault
              </span>
            </div>

            <div className="p-4 bg-[var(--black-950)] border border-[var(--line)] rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1 max-w-lg">
                <span className="text-sm font-semibold text-[var(--text-100)]">
                  Revoke All Active Sessions
                </span>
                <span className="text-xs text-[var(--text-300)] leading-relaxed">
                  Sign out across all browsers and devices immediately. All outstanding refresh tokens will be revoked in the backend token registry.
                </span>
              </div>

              <button
                type="button"
                onClick={handleRevokeAllSessions}
                disabled={revokeLoading}
                className="h-[36px] px-3.5 rounded-lg bg-[var(--black-700)] hover:bg-[var(--line)] active:bg-[var(--black-950)] border border-[var(--line)] text-[var(--text-100)] text-xs font-semibold inline-flex items-center gap-2 transition-colors shrink-0 cursor-pointer disabled:opacity-50 shadow-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)]"
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
            className="rounded-[14px] bg-[var(--black-900)] border border-[var(--alert-500)]/30 p-5 sm:p-6 flex flex-col gap-4 scroll-mt-6 shadow-xs"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 flex items-center justify-center text-[var(--alert-500)]">
                  <ShieldAlert className="w-4 h-4" />
                </div>
                <div>
                  <h2
                    id="danger-zone-heading"
                    className="text-base font-semibold text-[var(--alert-500)]"
                  >
                    Danger Zone
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-0.5">
                    Permanent purge of operator record and forensic evidence.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[var(--alert-bg)] text-[var(--alert-500)] border border-[var(--alert-500)]/20">
                Irreversible
              </span>
            </div>

            <div className="p-4 bg-[var(--alert-bg)] border border-[var(--alert-500)]/25 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1 max-w-lg">
                <span className="text-sm font-semibold text-[var(--alert-500)]">
                  Permanently Purge Operator Account & Analyses
                </span>
                <span className="text-xs text-[var(--text-300)] leading-relaxed">
                  Irreversibly delete operator identity, all forensic analysis dossiers, audio-visual attention rollout heatmaps, and uploaded media artifacts from disk storage.
                </span>
              </div>

              <button
                type="button"
                onClick={() => setDeleteModalOpen(true)}
                className="h-[36px] px-3.5 rounded-lg bg-[var(--alert-bg)] hover:bg-[var(--alert-bg)] active:bg-[var(--alert-500)]/25 border border-[var(--alert-500)]/40 text-[var(--alert-500)] text-xs font-semibold inline-flex items-center gap-2 transition-colors shrink-0 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--alert-500)]"
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
