import React, { useState, useId } from 'react';
import { clsx } from 'clsx';
import { Eye, EyeOff } from 'lucide-react';

export interface FieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hint?: string;
  error?: string;
  mono?: boolean;
}

export const Field = React.forwardRef<HTMLInputElement, FieldProps>(({
  label,
  hint,
  error,
  mono = false,
  type = 'text',
  className,
  id: customId,
  ...props
}, ref) => {
  const generatedId = useId();
  const id = customId || generatedId;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;

  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === 'password';
  const effectiveType = isPassword ? (showPassword ? 'text' : 'password') : type;

  return (
    <div className="flex flex-col gap-1.5 w-full">
      <div className="flex items-center justify-between">
        <label
          htmlFor={id}
          className="font-mono text-xs font-semibold uppercase tracking-wider text-[var(--text-300)]"
        >
          {label}
        </label>
        {hint && (
          <span id={hintId} className="font-mono text-[11px] text-[var(--text-500)]">
            {hint}
          </span>
        )}
      </div>

      <div className="relative flex items-center">
        <input
          ref={ref}
          id={id}
          type={effectiveType}
          aria-invalid={!!error}
          aria-describedby={clsx(hint && hintId, error && errorId) || undefined}
          className={clsx(
            'w-full min-h-[44px] px-3.5 rounded-[var(--radius-sm)] bg-[var(--black-800)] text-[var(--text-100)] border transition-all duration-150',
            mono ? 'font-mono text-xs' : 'font-sans text-sm',
            error
              ? 'border-[var(--alert-500)] focus:border-[var(--alert-500)]'
              : 'border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-500)]',
            'focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] focus-visible:outline-offset-1',
            isPassword && 'pr-11',
            className
          )}
          {...props}
        />

        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            className="absolute right-2.5 p-1.5 text-[var(--text-500)] hover:text-[var(--text-300)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] rounded"
          >
            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        )}
      </div>

      {error && (
        <span
          id={errorId}
          role="alert"
          className="font-mono text-[11px] font-medium text-[var(--alert-500)] tracking-wide flex items-center gap-1"
        >
          <span className="inline-block w-1 h-1 rounded-full bg-[var(--alert-500)]" />
          {error}
        </span>
      )}
    </div>
  );
});

Field.displayName = 'Field';
