import React, { useEffect, useRef } from 'react';
import { Button } from './Button';

export interface DialogProps {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'primary';
  loading?: boolean;
  onConfirm: () => void;
  onClose: () => void;
  children?: React.ReactNode;
}

export const Dialog: React.FC<DialogProps> = ({
  open,
  title,
  description,
  confirmLabel = 'CONFIRM',
  cancelLabel = 'CANCEL',
  variant = 'danger',
  loading = false,
  onConfirm,
  onClose,
  children,
}) => {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (open) {
      previousActiveElement.current = document.activeElement as HTMLElement;

      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          onClose();
        }

        // Trap focus inside dialog
        if (e.key === 'Tab' && dialogRef.current) {
          const focusables = dialogRef.current.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          );
          if (focusables.length === 0) return;
          const first = focusables[0];
          const last = focusables[focusables.length - 1];

          if (e.shiftKey && document.activeElement === first) {
            last.focus();
            e.preventDefault();
          } else if (!e.shiftKey && document.activeElement === last) {
            first.focus();
            e.preventDefault();
          }
        }
      };

      document.addEventListener('keydown', handleKeyDown);

      // Focus first interactive element inside dialog
      requestAnimationFrame(() => {
        if (dialogRef.current) {
          const first = dialogRef.current.querySelector<HTMLElement>('button');
          first?.focus();
        }
      });

      return () => {
        document.removeEventListener('keydown', handleKeyDown);
        previousActiveElement.current?.focus();
      };
    }
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[var(--black-950)]/85"
      onClick={(e) => {
        if (e.target === e.currentTarget && !loading) {
          onClose();
        }
      }}
      aria-modal="true"
      role="dialog"
      aria-labelledby="dialog-title"
      aria-describedby={description ? 'dialog-description' : undefined}
    >
      <div
        ref={dialogRef}
        className="relative w-full max-w-md bg-[var(--black-900)] border border-[var(--line-strong)] rounded-[var(--radius-md)] p-6 shadow-2xl flex flex-col gap-4"
      >
        {/* Corner Crop Marks */}
        <span className="crop-mark crop-tl" aria-hidden="true" />
        <span className="crop-mark crop-tr" aria-hidden="true" />
        <span className="crop-mark crop-bl" aria-hidden="true" />
        <span className="crop-mark crop-br" aria-hidden="true" />

        <div className="flex flex-col gap-1.5">
          <div className="font-mono text-[10px] text-[var(--cobalt-300)] tracking-[0.12em] uppercase">
            Operator confirmation
          </div>
          <h2 id="dialog-title" className="font-display font-bold text-lg tracking-tight text-[var(--text-100)]">
            {title}
          </h2>
          {description && (
            <p id="dialog-description" className="font-sans text-xs text-[var(--text-300)] leading-relaxed">
              {description}
            </p>
          )}
        </div>

        {children && <div className="py-2">{children}</div>}

        <div className="flex items-center justify-end gap-3 pt-3 border-t border-[var(--line)]">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            disabled={loading}
          >
            {cancelLabel}
          </Button>
          <Button
            variant={variant === 'danger' ? 'alert' : 'primary'}
            size="sm"
            onClick={onConfirm}
            loading={loading}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
};
