import React, { useEffect } from 'react';
import { clsx } from 'clsx';

export interface ToastProps {
  message: string;
  variant?: 'info' | 'success' | 'alert';
  durationMs?: number;
  onDismiss: () => void;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  variant = 'info',
  durationMs = 4000,
  onDismiss,
}) => {
  useEffect(() => {
    if (durationMs > 0) {
      const timer = setTimeout(onDismiss, durationMs);
      return () => clearTimeout(timer);
    }
  }, [durationMs, onDismiss]);

  const borderVariant = {
    info: 'border-[var(--cobalt-500)] text-white',
    success: 'border-[var(--lime-400)] text-[var(--lime-400)]',
    alert: 'border-[var(--alert-500)] text-[var(--alert-500)]',
  }[variant];

  return (
    <div
      role="status"
      aria-live="polite"
      className={clsx(
        'fixed bottom-6 right-6 z-50 flex items-center gap-3 px-4 py-3 bg-[var(--black-900)] border rounded-[var(--radius-sm)] shadow-xl font-mono text-xs',
        borderVariant
      )}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      <span className="text-white font-sans text-xs">{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="ml-2 text-[var(--text-400)] hover:text-white uppercase font-mono text-[10px]"
        aria-label="Dismiss status notification"
      >
        [×]
      </button>
    </div>
  );
};
