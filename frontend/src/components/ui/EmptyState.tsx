import React from 'react';
import { clsx } from 'clsx';
import { Button, type ButtonProps } from './Button';

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
    variant?: ButtonProps['variant'];
    icon?: React.ReactNode;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  secondaryAction,
  className,
}) => {
  return (
    <div
      role="region"
      aria-label={title}
      className={clsx(
        'flex flex-col items-center justify-center text-center p-8 md:p-12 rounded-[var(--radius-md)] bg-[var(--surface-card)] border border-[var(--border-default)] max-w-xl mx-auto my-6',
        className,
      )}
    >
      {icon && (
        <div className="w-12 h-12 rounded-full bg-[var(--surface-elevated)] border border-[var(--border-strong)] flex items-center justify-center text-[var(--cobalt-400)] mb-4">
          {icon}
        </div>
      )}

      <h3 className="font-sans font-bold text-base md:text-lg text-[var(--text-100)] tracking-tight">
        {title}
      </h3>

      <p className="font-sans text-xs md:text-sm text-[var(--text-300)] leading-relaxed max-w-md mt-1.5 mb-6">
        {description}
      </p>

      {(action || secondaryAction) && (
        <div className="flex flex-wrap items-center justify-center gap-3">
          {action && (
            <Button
              variant={action.variant || 'primary'}
              size="md"
              icon={action.icon}
              onClick={action.onClick}
            >
              {action.label}
            </Button>
          )}

          {secondaryAction && (
            <Button
              variant="outline"
              size="md"
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
};
