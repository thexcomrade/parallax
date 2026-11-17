import React from 'react';
import { clsx } from 'clsx';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from './Button';

export interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Operation Error',
  message,
  onRetry,
  className,
}) => {
  return (
    <div
      role="alert"
      className={clsx(
        'p-5 md:p-6 rounded-[var(--radius-md)] bg-[var(--alert-bg)] border border-[var(--alert-500)]/40 flex flex-col sm:flex-row items-start gap-4',
        className,
      )}
    >
      <div className="w-10 h-10 rounded-full bg-[var(--alert-500)]/15 border border-[var(--alert-500)]/30 flex items-center justify-center text-[var(--alert-500)] shrink-0">
        <AlertTriangle size={20} />
      </div>

      <div className="flex-1 flex flex-col gap-1">
        <h4 className="font-sans font-bold text-sm text-[var(--text-100)] uppercase tracking-wider">
          {title}
        </h4>
        <p className="font-sans text-xs text-[var(--text-200)] leading-relaxed">
          {message}
        </p>
      </div>

      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          icon={<RefreshCw size={14} />}
          className="shrink-0 self-end sm:self-center"
        >
          RETRY
        </Button>
      )}
    </div>
  );
};
