import React, { useState, useId } from 'react';
import { clsx } from 'clsx';
import { Eye, EyeOff } from 'lucide-react';

export interface FieldProps extends React.InputHTMLAttributes<HTMLInputElement | HTMLTextAreaElement> {
  label: string;
  hint?: string;
  error?: string;
  mono?: boolean;
  as?: 'input' | 'textarea';
  rows?: number;
  icon?: React.ReactNode;
}

export const Field = React.forwardRef<HTMLInputElement & HTMLTextAreaElement, FieldProps>(
  (
    {
      label,
      hint,
      error,
      mono = false,
      type = 'text',
      as = 'input',
      rows = 3,
      icon,
      className,
      id: customId,
      ...props
    },
    ref,
  ) => {
    const generatedId = useId();
    const id = customId || generatedId;
    const hintId = `${id}-hint`;
    const errorId = `${id}-error`;

    const [showPassword, setShowPassword] = useState(false);
    const isPassword = type === 'password';
    const effectiveType = isPassword ? (showPassword ? 'text' : 'password') : type;

    const controlClasses = clsx(
      'w-full min-h-[42px] px-3.5 rounded-[var(--radius-sm)] bg-[var(--surface-elevated)] text-[var(--text-100)] border transition-all duration-150',
      mono ? 'font-mono text-xs' : 'font-sans text-sm',
      error
        ? 'border-[var(--alert-500)] focus:border-[var(--alert-500)]'
        : 'border-[var(--border-default)] hover:border-[var(--border-strong)] focus:border-[var(--cobalt-500)]',
      'focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] focus-visible:outline-offset-1',
      icon && 'pl-10',
      isPassword && 'pr-11',
      className,
    );

    return (
      <div className="flex flex-col gap-1.5 w-full">
        <div className="flex items-center justify-between">
          <label
            htmlFor={id}
            className="font-sans text-xs font-medium text-[var(--text-200)]"
          >
            {label}
          </label>
          {hint && (
            <span id={hintId} className="font-mono text-[11px] text-[var(--text-400)]">
              {hint}
            </span>
          )}
        </div>

        <div className="relative flex items-center">
          {icon && (
            <span className="absolute left-3 text-[var(--text-400)] pointer-events-none flex items-center">
              {icon}
            </span>
          )}

          {as === 'textarea' ? (
            <textarea
              ref={ref as unknown as React.Ref<HTMLTextAreaElement>}
              id={id}
              rows={rows}
              aria-invalid={!!error}
              aria-describedby={clsx(hint && hintId, error && errorId) || undefined}
              className={clsx(controlClasses, 'py-2.5 resize-y')}
              {...(props as React.TextareaHTMLAttributes<HTMLTextAreaElement>)}
            />
          ) : (
            <input
              ref={ref as unknown as React.Ref<HTMLInputElement>}
              id={id}
              type={effectiveType}
              aria-invalid={!!error}
              aria-describedby={clsx(hint && hintId, error && errorId) || undefined}
              className={controlClasses}
              {...(props as React.InputHTMLAttributes<HTMLInputElement>)}
            />
          )}

          {isPassword && (
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
              className="absolute right-2.5 p-1.5 text-[var(--text-400)] hover:text-[var(--text-200)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] rounded"
            >
              {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          )}
        </div>

        {error && (
          <span
            id={errorId}
            role="alert"
            aria-live="polite"
            className="font-mono text-[11px] font-medium text-[var(--alert-500)] tracking-wide flex items-center gap-1.5 mt-0.5"
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--alert-500)] shrink-0" />
            {error}
          </span>
        )}
      </div>
    );
  },
);

Field.displayName = 'Field';
