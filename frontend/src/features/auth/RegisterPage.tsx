import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { SignalStage } from '@/components/viz/SignalStage';
import { Field } from '@/components/ui/Field';
import { Button } from '@/components/ui/Button';
import { registerSchema, type RegisterFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import { UserPlus, ArrowRight, Check, X } from 'lucide-react';
import { ApiError } from '@/lib/api';

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
    defaultValues: {
      email: '',
      password: '',
      confirmPassword: '',
    },
    mode: 'onChange',
  });

  const passwordVal = watch('password', '');

  // Live Password Strength Criteria
  const hasMinLength = passwordVal.length >= 12;
  const hasUpper = /[A-Z]/.test(passwordVal);
  const hasLower = /[a-z]/.test(passwordVal);
  const hasNumber = /[0-9]/.test(passwordVal);
  const hasSpecial = /[^A-Za-z0-9]/.test(passwordVal);

  const criteria = [
    { label: '≥ 12 CHARACTERS', passed: hasMinLength },
    { label: 'UPPERCASE [A-Z]', passed: hasUpper },
    { label: 'LOWERCASE [a-z]', passed: hasLower },
    { label: 'NUMERAL [0-9]', passed: hasNumber },
    { label: 'SYMBOL [!@#$]', passed: hasSpecial },
  ];

  const passedCount = criteria.filter((c) => c.passed).length;

  const onSubmit = async (data: RegisterFormData) => {
    setServerError(null);
    try {
      await registerAnalyst(data);
      // Auto-authenticate upon registration
      await login({ email: data.email, password: data.password });
      navigate(next, { replace: true });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setServerError('Rate limit exceeded. Please wait 60 seconds.');
        } else {
          setServerError(err.message || 'Registration failed.');
        }
      } else {
        setServerError('Registration failed. Try again.');
      }
    }
  };

  return (
    <main className="min-h-screen bg-[var(--black-950)] flex flex-col lg:flex-row text-[var(--text-100)] select-none">
      {/* Left Stage */}
      <aside
        aria-label="Optical and acoustic telemetry"
        className="lg:w-7/12 h-64 lg:h-auto shrink-0 relative"
      >
        <SignalStage className="h-full" />
      </aside>

      {/* Right Stage: Register Form */}
      <section
        aria-label="Analyst Enrollment Form"
        className="lg:w-5/12 flex flex-col justify-center px-6 py-8 md:px-12 lg:px-16 bg-[var(--black-900)] border-t lg:border-t-0 lg:border-l border-[var(--line)]"
      >
        <div className="max-w-md w-full mx-auto flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <UserPlus className="text-[var(--lime-400)]" size={20} />
              <span className="font-mono text-xs uppercase tracking-widest text-[var(--lime-400)]">
                OPERATOR ENROLLMENT
              </span>
            </div>
            <h1 className="font-display font-black text-2xl lg:text-3xl tracking-tight uppercase text-white">
              CREATE ANALYST PROFILE
            </h1>
            <p className="font-mono text-xs text-[var(--text-500)] leading-relaxed">
              Enroll your credentials with zero-knowledge password hashing (argon2id) and rotating tokens.
            </p>
          </div>

          {serverError && (
            <div
              role="alert"
              className="px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--alert-bg)] border border-[var(--alert-500)] text-[var(--alert-500)] font-mono text-xs flex items-center gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--alert-500)] animate-ping" />
              <span>{serverError}</span>
            </div>
          )}

          <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4" noValidate>
            <Field
              label="Operator Email"
              type="email"
              autoComplete="email"
              placeholder="analyst@parallax.io"
              error={errors.email?.message}
              hint="REQUIRED"
              {...register('email')}
            />

            <Field
              label="New Password"
              type="password"
              autoComplete="new-password"
              placeholder="••••••••••••"
              error={errors.password?.message}
              hint="ARGON2 PROTECTED"
              {...register('password')}
            />

            {/* Live Password Strength Meter */}
            <div className="p-3 bg-[var(--black-800)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-2">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="text-[var(--text-500)] uppercase">ENTROPY STRENGTH</span>
                <span
                  className={
                    passedCount === 5
                      ? 'text-[var(--lime-400)] font-bold'
                      : passedCount >= 3
                      ? 'text-amber-300 font-bold'
                      : 'text-[var(--alert-500)] font-bold'
                  }
                >
                  {passedCount === 5 ? 'OPTIMAL [5/5]' : `SEC_LEVEL [${passedCount}/5]`}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-1 bg-[var(--black-900)] rounded-full overflow-hidden flex gap-0.5">
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <div
                    key={lvl}
                    className={`flex-1 h-full transition-colors ${
                      passedCount >= lvl
                        ? passedCount === 5
                          ? 'bg-[var(--lime-400)]'
                          : passedCount >= 3
                          ? 'bg-amber-400'
                          : 'bg-[var(--alert-500)]'
                        : 'bg-[var(--line)]'
                    }`}
                  />
                ))}
              </div>

              {/* Individual ticks */}
              <div className="grid grid-cols-2 gap-1.5 pt-1">
                {criteria.map((c) => (
                  <div key={c.label} className="flex items-center gap-1.5 text-[10px] font-mono">
                    {c.passed ? (
                      <Check size={12} className="text-[var(--lime-400)] shrink-0" />
                    ) : (
                      <X size={12} className="text-[var(--text-500)] shrink-0" />
                    )}
                    <span className={c.passed ? 'text-[var(--text-100)]' : 'text-[var(--text-500)]'}>
                      {c.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <Field
              label="Confirm Password"
              type="password"
              autoComplete="new-password"
              placeholder="••••••••••••"
              error={errors.confirmPassword?.message}
              hint="MATCH REQUIRED"
              {...register('confirmPassword')}
            />

            <div className="pt-2">
              <Button
                type="submit"
                variant="primary"
                size="lg"
                loading={isRegistering}
                className="w-full"
                icon={<ArrowRight size={16} />}
              >
                COMPLETE ENROLLMENT
              </Button>
            </div>
          </form>

          <div className="pt-4 border-t border-[var(--line)] flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono">
            <span className="text-[var(--text-500)]">Already registered?</span>
            <Link
              to={`/login${next ? `?next=${encodeURIComponent(next)}` : ''}`}
              className="text-[var(--cobalt-300)] hover:text-white underline decoration-[var(--cobalt-500)] underline-offset-4 focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
            >
              AUTHENTICATE EXISTING →
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
};
