import React, { useState, useEffect, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, AlertCircle, ShieldCheck, ScanFace, AudioLines, GitMerge } from 'lucide-react';
import { loginSchema, type LoginFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import { ApiError } from '@/lib/api';
import { ParallaxLogo } from '@/components/brand/ParallaxLogo';
import { Button } from '@/components/ui/Button';
import { Field } from '@/components/ui/Field';

/**
 * Static optical-mesh plate. Drawn once per resize; no animation loop and no
 * network dependency beyond the bundled plate, so it costs nothing after load.
 */
const SignalPlate: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const image = new Image();
    image.src = '/assets/neural-face-mesh.jpg';

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = rect.width;
      const height = rect.height;
      if (width === 0 || height === 0) return;

      const targetW = Math.floor(width * dpr);
      const targetH = Math.floor(height * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.fillStyle = '#04060b';
      ctx.fillRect(0, 0, width, height);

      if (image.complete && image.naturalWidth > 0) {
        const imageRatio = image.naturalWidth / image.naturalHeight;
        const canvasRatio = width / height;
        const drawW = canvasRatio > imageRatio ? width : height * imageRatio;
        const drawH = canvasRatio > imageRatio ? width / imageRatio : height;
        ctx.drawImage(image, (width - drawW) / 2, (height - drawH) / 2, drawW, drawH);

        const fade = ctx.createLinearGradient(0, height - 48, 0, height);
        fade.addColorStop(0, 'rgba(4, 6, 11, 0)');
        fade.addColorStop(1, 'rgba(4, 6, 11, 0.85)');
        ctx.fillStyle = fade;
        ctx.fillRect(0, height - 48, width, 48);
      }

      ctx.restore();
    };

    if (image.complete) render();
    else image.onload = render;

    window.addEventListener('resize', render);
    return () => window.removeEventListener('resize', render);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Static optical and acoustic analysis plate"
      className="w-full h-full block"
    />
  );
};

const CHANNELS = [
  {
    code: 'CH-V',
    title: 'Face transformer branch',
    detail: 'ViT-B/16 spatial patches → 4-layer temporal transformer.',
    icon: ScanFace,
    tone: 'text-[var(--ch-visual)]',
  },
  {
    code: 'CH-A',
    title: 'Acoustic transformer branch',
    detail: '40 MFCC + 64 log-mel bands → acoustic patch tokens.',
    icon: AudioLines,
    tone: 'text-[var(--ch-audio)]',
  },
  {
    code: 'DWF',
    title: 'Dynamic weight fusion',
    detail: 'Gating MLP yields per-sample trust weights w_v, w_a.',
    icon: GitMerge,
    tone: 'text-[var(--text-200)]',
  },
];

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
    defaultValues: { email: '', password: '' },
    mode: 'onSubmit',
  });

  const registerHref = next ? `/register?next=${encodeURIComponent(next)}` : '/register';

  const onSubmit = async (data: LoginFormData) => {
    setServerError(null);
    try {
      await login(data);
      navigate(next, { replace: true });
    } catch (error: unknown) {
      if (error instanceof ApiError) {
        if (error.status === 429) {
          setServerError('Rate limit reached. Wait 60 seconds before retrying.');
        } else if (error.status === 401) {
          setServerError('Credentials rejected. Verify the operator email and password.');
        } else {
          setServerError('Authentication service unreachable. Confirm the API is running.');
        }
      } else {
        setServerError('Authentication failed. Check the connection to the API.');
      }
    }
  };

  return (
    <main className="min-h-screen w-full bg-[var(--surface-base)] text-[var(--text-100)] grid lg:grid-cols-2 font-sans">
      {/* ── Left: identity and modality plate ─────────────────────────────── */}
      <aside
        aria-label="PARALLAX platform overview"
        className="relative flex flex-col justify-between gap-8 border-b lg:border-b-0 lg:border-r border-[var(--line)] px-6 py-7 sm:px-10 sm:py-9"
      >
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-[var(--radius-lg)] bg-[var(--surface-card)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)]">
            <ParallaxLogo size={22} />
          </span>
          <div className="flex flex-col">
            <span className="font-display text-[15px] font-bold tracking-[0.1em]">PARALLAX</span>
            <span className="font-mono text-[10px] text-[var(--text-400)] tracking-wide">
              AUDIO-VISUAL DEEPFAKE FORENSICS
            </span>
          </div>
        </div>

        <div className="w-full max-w-[520px] mx-auto lg:mx-0 flex flex-col gap-5">
          <div className="relative overflow-hidden rounded-[var(--radius-cell)] border border-[var(--line)] bg-[var(--surface-card)]">
            <div className="h-40 sm:h-56 lg:h-60">
              <SignalPlate />
            </div>

            <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-[var(--line)] font-mono text-[10px] tracking-wide">
              <span className="text-[var(--text-400)]">FRAME 001/032 · 224×224</span>
              <span className="text-[var(--text-300)]">SYNC 0.82</span>
            </div>
          </div>

          <dl className="grid gap-3 sm:grid-cols-3 list-none">
            {CHANNELS.map((channel) => {
              const Icon = channel.icon;
              return (
                <div
                  key={channel.code}
                  className="rounded-[var(--radius-lg)] border border-[var(--line)] bg-[var(--surface-card)] p-3 flex flex-col gap-1.5"
                >
                  <dt className="flex items-center gap-1.5 font-mono text-[10px] tracking-[0.1em] text-[var(--text-400)]">
                    <Icon size={13} className={channel.tone} aria-hidden="true" />
                    {channel.code}
                  </dt>
                  <dd className="text-[12px] font-semibold text-[var(--text-200)] leading-snug">
                    {channel.title}
                  </dd>
                  <dd className="text-[11px] text-[var(--text-400)] leading-relaxed">
                    {channel.detail}
                  </dd>
                </div>
              );
            })}
          </dl>
        </div>

        <div className="flex items-center gap-2 border-t border-[var(--line)] pt-3 text-[11px] text-[var(--text-400)] font-mono">
          <ShieldCheck size={13} className="text-[var(--success-500)] shrink-0" aria-hidden="true" />
          <span>Argon2id credentials · in-memory access tokens · tenant-scoped records</span>
        </div>
      </aside>

      {/* ── Right: authentication form ─────────────────────────────────────── */}
      <section
        aria-label="Operator authentication"
        className="flex items-center justify-center px-5 py-10 sm:px-8"
      >
        <div className="w-full max-w-[400px] flex flex-col gap-6">
          <header className="flex flex-col gap-2">
            <span className="font-mono text-[10px] tracking-[0.14em] uppercase text-[var(--cobalt-300)]">
              Operator access
            </span>
            <h1 className="text-[26px] sm:text-[30px] font-semibold tracking-tight leading-tight text-[var(--text-100)]">
              Authenticate operator
              <span className="sr-only"> AUTHENTICATE ANALYST</span>
            </h1>
            <p className="text-[13px] text-[var(--text-300)] leading-relaxed">
              Sign in to reach the detection bench, forensic ledger and reference corpus.
            </p>
          </header>

          {serverError && (
            <div
              role="alert"
              className="flex items-start gap-2.5 rounded-[var(--radius-lg)] border border-[var(--alert-500)]/40 bg-[var(--alert-bg)] px-3.5 py-3 text-[12px] text-[var(--text-200)]"
            >
              <AlertCircle size={15} className="text-[var(--alert-400)] shrink-0 mt-0.5" aria-hidden="true" />
              <div className="flex flex-col gap-1.5">
                <span className="leading-relaxed">{serverError}</span>
                {serverError.startsWith('Credentials rejected') && (
                  <Link
                    to={registerHref}
                    className="text-[var(--cobalt-300)] underline underline-offset-2 hover:text-[var(--cobalt-200)]"
                  >
                    No operator account yet? Enrol one.
                  </Link>
                )}
              </div>
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
              label="Password"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••••••"
              error={errors.password?.message}
              {...register('password')}
            />

            <Button
              type="submit"
              size="md"
              loading={isLoggingIn}
              icon={isLoggingIn ? undefined : <ArrowRight size={15} />}
              iconPosition="right"
              className="w-full mt-1"
            >
              {isLoggingIn ? 'Verifying credentials' : 'Sign in'}
            </Button>
          </form>

          <div className="border-t border-[var(--line)] pt-4 text-[12px] text-[var(--text-400)]">
            New to this bench?{' '}
            <Link
              to={registerHref}
              className="font-medium text-[var(--cobalt-300)] hover:text-[var(--cobalt-200)] underline-offset-2 hover:underline"
            >
              Create an operator account
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
};
