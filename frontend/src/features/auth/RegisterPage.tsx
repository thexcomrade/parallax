import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, AlertCircle, Check, ShieldCheck } from 'lucide-react';
import { registerSchema, type RegisterFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import { ApiError } from '@/lib/api';
import { ParallaxLogo } from '@/components/brand/ParallaxLogo';
import { Button } from '@/components/ui/Button';
import { Field } from '@/components/ui/Field';

const CRITERIA = [
  { id: 'length', label: '12 characters or more', test: (value: string) => value.length >= 12 },
  { id: 'upper', label: 'Uppercase letter', test: (value: string) => /[A-Z]/.test(value) },
  { id: 'lower', label: 'Lowercase letter', test: (value: string) => /[a-z]/.test(value) },
  { id: 'number', label: 'Number', test: (value: string) => /[0-9]/.test(value) },
  { id: 'symbol', label: 'Symbol', test: (value: string) => /[^A-Za-z0-9]/.test(value) },
];

export const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const next = searchParams.get('next') || '/app';

  const { register: registerAnalyst, login, isRegistering } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: { email: '', password: '', confirmPassword: '' },
    mode: 'onSubmit',
  });

  const passwordValue = watch('password', '');
  const results = CRITERIA.map((criterion) => ({
    ...criterion,
    passed: criterion.test(passwordValue),
  }));
  const passedCount = results.filter((criterion) => criterion.passed).length;
  const strengthLabel =
    passedCount === CRITERIA.length
      ? 'Optimal [5/5]'
      : `${passedCount}/${CRITERIA.length} required`;

  const loginHref = next ? `/login?next=${encodeURIComponent(next)}` : '/login';

  const onSubmit = async (data: RegisterFormData) => {
    setServerError(null);
    try {
      await registerAnalyst(data);
      await login({ email: data.email, password: data.password });
      navigate(next, { replace: true });
    } catch (error: unknown) {
      if (error instanceof ApiError) {
        if (error.status === 429) {
          setServerError('Rate limit reached. Wait 60 seconds before retrying.');
        } else {
          setServerError(error.message || 'Enrolment failed.');
        }
      } else {
        setServerError('Enrolment failed. Check the connection to the API.');
      }
    }
  };

  return (
    <main className="min-h-screen bg-[var(--surface-base)] text-[var(--text-100)] flex flex-col items-center justify-center px-5 py-12 font-sans">
      <div className="w-full max-w-[440px] flex flex-col gap-6">
        <header className="flex flex-col gap-3">
          <div className="flex items-center gap-2.5">
            <span className="w-9 h-9 rounded-[var(--radius-lg)] bg-[var(--surface-card)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)]">
              <ParallaxLogo size={22} />
            </span>
            <span className="font-display text-[14px] font-bold tracking-[0.1em]">PARALLAX</span>
          </div>

          <h1 className="text-[24px] sm:text-[28px] font-semibold tracking-tight leading-tight">
            Enrol an operator
            <span className="sr-only"> CREATE ANALYST PROFILE</span>
          </h1>
          <p className="text-[13px] text-[var(--text-300)] leading-relaxed">
            Accounts are scoped to a single operator. Every analysis you upload is bound to this
            identity and invisible to other operators.
          </p>
        </header>

        <div className="rounded-[var(--radius-cell)] border border-[var(--line)] bg-[var(--surface-card)] p-5 sm:p-6 flex flex-col gap-5">
          {serverError && (
            <div
              role="alert"
              className="flex items-start gap-2.5 rounded-[var(--radius-lg)] border border-[var(--alert-500)]/40 bg-[var(--alert-bg)] px-3.5 py-3 text-[12px] text-[var(--text-200)]"
            >
              <AlertCircle size={15} className="text-[var(--alert-400)] shrink-0 mt-0.5" aria-hidden="true" />
              <span className="leading-relaxed">{serverError}</span>
            </div>
          )}

          <form onSubmit={handleSubmit(onSubmit)} noValidate className="flex flex-col gap-4">
            <Field
              label="Operator email"
              type="email"
              autoComplete="email"
              placeholder="analyst@institution.org"
              error={errors.email?.message}
              {...register('email')}
            />

            <Field
              label="New password"
              hint="12+ characters"
              type="password"
              autoComplete="new-password"
              placeholder="••••••••••••"
              error={errors.password?.message}
              {...register('password')}
            />

            <section aria-label="Password policy" className="flex flex-col gap-2.5 -mt-1">
              <div className="flex items-center justify-between font-mono text-[10px] tracking-[0.1em] uppercase">
                <span className="text-[var(--text-400)]">Entropy strength</span>
                <span
                  className={
                    passedCount === CRITERIA.length
                      ? 'text-[var(--success-500)]'
                      : 'text-[var(--text-300)]'
                  }
                >
                  {strengthLabel}
                </span>
              </div>

              <div className="grid grid-cols-5 gap-1 h-1" aria-hidden="true">
                {CRITERIA.map((criterion, index) => (
                  <span
                    key={criterion.id}
                    className={
                      passedCount > index
                        ? 'rounded-full bg-[var(--success-500)]'
                        : 'rounded-full bg-[var(--line-strong)]'
                    }
                  />
                ))}
              </div>

              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5 list-none mt-0.5">
                {results.map((criterion) => (
                  <li
                    key={criterion.id}
                    className="flex items-center gap-1.5 text-[11px] text-[var(--text-400)]"
                  >
                    <Check
                      size={12}
                      aria-hidden="true"
                      className={
                        criterion.passed
                          ? 'text-[var(--success-500)]'
                          : 'text-[var(--text-500)]'
                      }
                    />
                    <span className={criterion.passed ? 'text-[var(--text-200)]' : undefined}>
                      {criterion.label}
                    </span>
                  </li>
                ))}
              </ul>
            </section>

            <Field
              label="Confirm password"
              type="password"
              autoComplete="new-password"
              placeholder="••••••••••••"
              error={errors.confirmPassword?.message}
              {...register('confirmPassword')}
            />

            <Button
              type="submit"
              size="md"
              loading={isRegistering}
              icon={isRegistering ? undefined : <ArrowRight size={15} />}
              iconPosition="right"
              className="w-full mt-1"
            >
              {isRegistering ? 'Creating account' : 'Complete enrollment'}
            </Button>
          </form>

          <div className="border-t border-[var(--line)] pt-4 text-[12px] text-[var(--text-400)]">
            Already enrolled?{' '}
            <Link
              to={loginHref}
              className="font-medium text-[var(--cobalt-300)] hover:text-[var(--cobalt-200)] underline-offset-2 hover:underline"
            >
              Sign in
            </Link>
          </div>
        </div>

        <p className="flex items-center justify-center gap-2 text-[11px] font-mono text-[var(--text-400)]">
          <ShieldCheck size={13} className="text-[var(--success-500)] shrink-0" aria-hidden="true" />
          Argon2id hashing · rotating refresh sessions
        </p>
      </div>
    </main>
  );
};
