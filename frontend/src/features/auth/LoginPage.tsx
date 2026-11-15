import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { SignalStage } from '@/components/viz/SignalStage';
import { Field } from '@/components/ui/Field';
import { Button } from '@/components/ui/Button';
import { loginSchema, type LoginFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import { ShieldCheck, ArrowRight } from 'lucide-react';
import { ApiError } from '@/lib/api';

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const next = searchParams.get('next') || '/app';

  const { login, isLoggingIn } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginFormData) => {
    setServerError(null);
    try {
      await login(data);
      navigate(next, { replace: true });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setServerError('Rate limit exceeded. Please wait 60 seconds.');
        } else {
          setServerError(err.message || 'Invalid email or password.');
        }
      } else {
        setServerError('Authentication failed. Check your credentials.');
      }
    }
  };

  return (
    <main className="min-h-screen bg-[var(--black-950)] flex flex-col lg:flex-row text-[var(--text-100)] select-none">
      {/* Left Stage: Optical & Acoustic Telemetry Visualizer */}
      <aside
        aria-label="Optical and acoustic telemetry"
        className="lg:w-7/12 h-64 lg:h-auto shrink-0 relative"
      >
        <SignalStage className="h-full" />
      </aside>

      {/* Right Stage: Forensic Access Form */}
      <section
        aria-label="Analyst Authentication Form"
        className="lg:w-5/12 flex flex-col justify-center px-6 py-8 md:px-12 lg:px-16 bg-[var(--black-900)] border-t lg:border-t-0 lg:border-l border-[var(--line)]"
      >
        <div className="max-w-md w-full mx-auto flex flex-col gap-6">
          {/* Header */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="text-[var(--lime-400)]" size={20} />
              <span className="font-mono text-xs uppercase tracking-widest text-[var(--lime-400)]">
                SECURE ACCESS TERMINAL
              </span>
            </div>
            <h1 className="font-display font-black text-2xl lg:text-3xl tracking-tight uppercase text-white">
              AUTHENTICATE ANALYST
            </h1>
            <p className="font-mono text-xs text-[var(--text-500)] leading-relaxed">
              Enter your credentials to access the multimodal deepfake detection bench and active scan jobs.
            </p>
          </div>

          {/* Error Banner */}
          {serverError && (
            <div
              role="alert"
              className="px-4 py-3 rounded-[var(--radius-sm)] bg-[var(--alert-bg)] border border-[var(--alert-500)] text-[var(--alert-500)] font-mono text-xs flex items-center gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--alert-500)] animate-ping" />
              <span>{serverError}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4" noValidate>
            <Field
              label="Operator Email"
              type="email"
              autoComplete="email"
              placeholder="analyst@parallax.io"
              error={errors.email?.message}
              hint="SECURE_ID"
              {...register('email')}
            />

            <Field
              label="Password Phrase"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••••••"
              error={errors.password?.message}
              hint="MIN 12 CHARS"
              {...register('password')}
            />

            <div className="pt-2">
              <Button
                type="submit"
                variant="primary"
                size="lg"
                loading={isLoggingIn}
                className="w-full"
                icon={<ArrowRight size={16} />}
              >
                INITIALIZE SESSION
              </Button>
            </div>
          </form>

          {/* Footer note & register link */}
          <div className="pt-4 border-t border-[var(--line)] flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono">
            <span className="text-[var(--text-500)]">No operator account?</span>
            <Link
              to={`/register${next ? `?next=${encodeURIComponent(next)}` : ''}`}
              className="text-[var(--cobalt-300)] hover:text-white underline decoration-[var(--cobalt-500)] underline-offset-4 focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
            >
              REGISTER NEW ANALYST →
            </Link>
          </div>

          <div className="text-[10px] font-mono text-[var(--text-500)] tracking-widest text-center mt-4">
            CH-V/CH-A DUAL-STREAM FORENSICS // 256-BIT ROTATING SESSION
          </div>
        </div>
      </section>
    </main>
  );
};
