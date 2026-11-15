import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { api } from '@/lib/api';
import { BentoCell } from '@/components/bento/BentoCell';
import { Button } from '@/components/ui/Button';
import { Field } from '@/components/ui/Field';
import { Dialog } from '@/components/ui/Dialog';
import { Toast } from '@/components/ui/Toast';
import { formatRelativeTime } from '@/lib/format';

export const SettingsPage: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Password state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  // Dialog & Toast states
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [revokeLoading, setRevokeLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError(null);

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
      setToastMsg('Password successfully updated; other sessions revoked.');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to update password';
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
      const msg = err instanceof Error ? err.message : 'Failed to revoke sessions';
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
      const msg = err instanceof Error ? err.message : 'Failed to purge account';
      setPasswordError(msg);
      setDeleteLoading(false);
      setDeleteModalOpen(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6 py-4 pb-24 md:pb-8">
      {/* Toast Alert */}
      {toastMsg && (
        <Toast
          message={toastMsg}
          variant="success"
          onDismiss={() => setToastMsg(null)}
        />
      )}

      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="font-mono text-xs uppercase text-[var(--cobalt-400)] tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--cobalt-400)]" />
            STATION // CH-07 // ANALYST SECURITY PREFERENCES
          </div>
          <h1 className="font-display font-black text-2xl md:text-3xl uppercase tracking-tight text-white mt-1">
            Analyst Account Settings
          </h1>
          <p className="font-sans text-xs md:text-sm text-[var(--text-300)] mt-0.5">
            Manage operator credentials, active cryptographic refresh sessions, and privacy lifecycle.
          </p>
        </div>
      </header>

      {/* Operator Profile Card */}
      <BentoCell label="OPERATOR PROFILE // CREDENTIALS" status="ACTIVE">
        <div className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-full bg-[var(--black-800)] border border-[var(--cobalt-400)]/40 flex items-center justify-center font-mono font-bold text-sm text-[var(--cobalt-300)] overflow-hidden">
              {user ? (
                <img
                  src={`https://api.dicebear.com/9.x/shapes/svg?seed=${user.id}`}
                  alt="Operator Avatar"
                  className="w-full h-full object-cover"
                />
              ) : (
                'OP'
              )}
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-sm font-bold text-white">
                {user?.email || 'analyst@parallax.io'}
              </span>
              <span className="font-mono text-xs text-[var(--text-500)]">
                OPERATOR ID // #{user?.id || 1} • REGISTERED {user?.created_at ? formatRelativeTime(user.created_at) : 'RECENTLY'}
              </span>
            </div>
          </div>

          <div className="px-3 py-1 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] font-mono text-xs text-[var(--lime-400)]">
            AUTH TIER: MULTI-MODAL FULL
          </div>
        </div>
      </BentoCell>

      {/* Password Change Card */}
      <BentoCell label="AUTHENTICATION // CHANGE CREDENTIALS" status="ARGON2 PROTECTED">
        <form onSubmit={handlePasswordChange} className="p-4 flex flex-col gap-4 max-w-md">
          {passwordError && (
            <div role="alert" className="p-3 bg-[var(--black-950)] border border-[var(--alert-500)] text-[var(--alert-500)] rounded-[var(--radius-sm)] font-mono text-xs">
              {passwordError}
            </div>
          )}

          <Field
            label="Current Master Password"
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            required
            autoComplete="current-password"
          />

          <Field
            label="New Master Password"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            hint="Minimum 12 characters. Use letters, numbers, and symbols."
            required
            autoComplete="new-password"
          />

          <Field
            label="Confirm New Password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            autoComplete="new-password"
          />

          <Button
            type="submit"
            variant="primary"
            size="md"
            loading={passwordLoading}
            className="mt-2"
          >
            UPDATE OPERATOR PASSWORD
          </Button>
        </form>
      </BentoCell>

      {/* Session Invalidation Card */}
      <BentoCell label="SESSION SECURITY // ROTATING REFRESH TOKENS" status="SESSIONS">
        <div className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex flex-col gap-1 max-w-lg">
            <span className="font-sans text-xs text-white font-bold">
              Revoke All Active Sessions
            </span>
            <span className="font-sans text-xs text-[var(--text-300)] leading-relaxed">
              Sign out across all devices and invalidates all issued cryptographic refresh tokens immediately.
            </span>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={handleRevokeAllSessions}
            loading={revokeLoading}
          >
            SIGN OUT EVERYWHERE
          </Button>
        </div>
      </BentoCell>

      {/* Danger Zone: Purge Account Card */}
      <BentoCell label="DANGER ZONE // PURGE LIFECYCLE" status="PERMANENT DELETION" hazard>
        <div className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex flex-col gap-1 max-w-lg">
            <span className="font-sans text-xs text-[var(--alert-500)] font-bold">
              Permanently Purge Account & Forensic Analyses
            </span>
            <span className="font-sans text-xs text-[var(--text-300)] leading-relaxed">
              Irreversibly delete operator credentials, forensic analysis logs, heatmaps, and all uploaded media files.
            </span>
          </div>

          <Button
            variant="alert"
            size="sm"
            onClick={() => setDeleteModalOpen(true)}
          >
            PURGE ACCOUNT & ALL DATA
          </Button>
        </div>
      </BentoCell>

      {/* Confirmation Dialog */}
      <Dialog
        open={deleteModalOpen}
        title="PERMANENTLY PURGE OPERATOR ACCOUNT?"
        description="This action cannot be undone. All video scans, extracted frames, attention heatmaps, and account credentials will be permanently erased from database storage."
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
