import React, { useState, useEffect, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { loginSchema, type LoginFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import { ApiError } from '@/lib/api';
import {
  ArrowRight,
  Eye,
  EyeOff,
  AlertCircle,
  Loader2,
  ShieldCheck,
} from 'lucide-react';

// ─── Neural Network Face Mesh Canvas (Static Image Display) ─────────────────
const NeuralFaceCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = '/assets/neural-face-mesh.jpg';

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width;
      const h = rect.height;
      if (w === 0 || h === 0) return;

      const targetW = Math.floor(w * dpr);
      const targetH = Math.floor(h * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      // Deep dark background matching theme
      ctx.fillStyle = '#090D14';
      ctx.fillRect(0, 0, w, h);

      if (img.complete && img.naturalWidth > 0) {
        // Draw image covering the canvas (object-fit: cover, centered)
        const imgRatio = img.naturalWidth / img.naturalHeight;
        const canvasRatio = w / h;
        let drawW: number;
        let drawH: number;
        let offsetX: number;
        let offsetY: number;

        if (canvasRatio > imgRatio) {
          drawW = w;
          drawH = w / imgRatio;
          offsetX = 0;
          offsetY = (h - drawH) / 2;
        } else {
          drawH = h;
          drawW = h * imgRatio;
          offsetX = (w - drawW) / 2;
          offsetY = (h - drawH) / 2;
        }

        ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

        // Subtle gradient vignette around bottom and top edges
        const vignetteBottom = ctx.createLinearGradient(0, h - 32, 0, h);
        vignetteBottom.addColorStop(0, 'rgba(17, 24, 39, 0)');
        vignetteBottom.addColorStop(1, 'rgba(17, 24, 39, 0.45)');
        ctx.fillStyle = vignetteBottom;
        ctx.fillRect(0, h - 32, w, 32);

        const vignetteTop = ctx.createLinearGradient(0, 0, 0, 24);
        vignetteTop.addColorStop(0, 'rgba(17, 24, 39, 0.35)');
        vignetteTop.addColorStop(1, 'rgba(17, 24, 39, 0)');
        ctx.fillStyle = vignetteTop;
        ctx.fillRect(0, 0, w, 24);
      }
      ctx.restore();
    };

    if (img.complete) {
      render();
    } else {
      img.onload = () => render();
    }

    window.addEventListener('resize', render);
    return () => {
      window.removeEventListener('resize', render);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Forensic optical flux and telemetry stage"
      className="w-full h-full block"
    />
  );
};

// ─── Main Login Page ────────────────────────────────────────────────────────
export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const next = searchParams.get('next') || '/app';
  const { login, isLoggingIn } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: '', password: '' },
    mode: 'onSubmit',
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
        } else if (err.status === 401) {
          setServerError('Invalid credentials. Verify your email and password.');
        } else {
          setServerError('Backend server unavailable. Please ensure the backend is running on port 8000.');
        }
      } else {
        setServerError('Authentication failed. Check your connection to the server.');
      }
    }
  };

  return (
    <main className="min-h-screen w-full bg-[#090D14] text-[#E8EDF5] flex flex-col lg:flex-row relative overflow-hidden font-sans">
      {/* ── LEFT SECTION — Brand & Neural Face Visual Area (50% on desktop) ── */}
      <aside
        className="w-full lg:w-1/2 bg-[#0D131E] border-b lg:border-b-0 lg:border-r border-[#253247] flex flex-col justify-between relative overflow-hidden shrink-0"
        style={{ padding: 'clamp(20px, 3.5vw, 44px)' }}
        aria-label="PARALLAX Forensic Intelligence Platform"
      >
        {/* Subtle radial ambient background glow */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
          <div className="absolute -top-24 -left-24 w-96 h-96 bg-[#648CFF]/8 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-[#35C9D0]/5 rounded-full blur-3xl" />
        </div>

        {/* Top Brand Identity */}
        <div className="relative z-10 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#111827] border border-[#253247] shadow-sm flex items-center justify-center text-[#648CFF]">
            <svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-wider text-[#E8EDF5]">PARALLAX</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#171F2D] border border-[#253247] text-[#648CFF] font-medium">
              </span>
            </div>
            <span className="text-xs text-[#91A0B8]">Multimodal Audio-Visual Deepfake Detection System</span>
          </div>
        </div>

        {/* Center Visual Stage: Neural Face Mesh inside Telemetry Box & Feature Cards */}
        <div className="relative z-10 my-6 lg:my-auto flex flex-col w-full max-w-[540px] mx-auto" style={{ gap: '16px' }}>
          {/* Main Visual Box */}
          <div className="w-full h-44 sm:h-56 lg:h-64 rounded-xl bg-[#111827] border border-[#253247] overflow-hidden relative shadow-inner">
            <NeuralFaceCanvas />

            {/* Stream Indicators */}
            <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#090D14]/90 border border-[#253247] text-[11px] font-medium text-[#648CFF] backdrop-blur-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-[#648CFF]" />
                Visual Stream 
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#090D14]/90 border border-[#253247] text-[11px] font-medium text-[#06B6D4] backdrop-blur-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-[#06B6D4]" />
                Acoustic Stream 
              </span>
            </div>

            <div className="absolute bottom-3 right-3 pointer-events-none">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#090D14]/90 border border-[#253247] text-[11px] font-medium text-[#35C99A] backdrop-blur-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-[#35C99A]" />
                Dual-Stream Coherence
              </span>
            </div>
          </div>

          {/* Forensic Feature Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div
              className="rounded-lg bg-[#111827]/70 border border-[#253247] flex flex-col"
              style={{ padding: '12px 14px', gap: '4px' }}
            >
              <span className="text-xs font-semibold text-[#E8EDF5]">Spatial-Temporal Artifacts</span>
              <p className="text-[11px] text-[#91A0B8] leading-relaxed">
                Facial warping, boundary inconsistency, and temporal photoplethysmography.
              </p>
            </div>
            <div
              className="rounded-lg bg-[#111827]/70 border border-[#253247] flex flex-col"
              style={{ padding: '12px 14px', gap: '4px' }}
            >
              <span className="text-xs font-semibold text-[#E8EDF5]">Neural Voice Coherence</span>
              <p className="text-[11px] text-[#91A0B8] leading-relaxed">
                Spectral acoustic anomalies, synthetic harmonic markers, and vocal timbre decay.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Status Strip */}
        <div className="relative z-10 pt-3 border-t border-[#253247]/60 flex items-center justify-between text-xs text-[#91A0B8]">
          <span>Multi-modal deepfake detection bench</span>
          <span className="text-[11px] font-medium text-[#648CFF]">256-BIT ROTATING SESSIONS</span>
        </div>
      </aside>

      {/* ── RIGHT SECTION — Login Form (50% on desktop) ── */}
      <section
        className="w-full lg:w-1/2 flex flex-col justify-center items-center relative overflow-hidden"
        style={{ padding: 'clamp(28px, 4vw, 48px) clamp(16px, 3vw, 32px)' }}
        aria-label="User Authentication Form"
      >
        <div className="w-full max-w-[420px] relative z-10 flex flex-col">
          {/* Elevated Card */}
          <div
            className="w-full bg-[#111827] border border-[#253247] rounded-2xl shadow-xl flex flex-col"
            style={{ padding: 'clamp(24px, 3.5vw, 36px)' }}
          >
            {/* Header inside Card */}
            <div className="text-left" style={{ marginBottom: '24px' }}>
              <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#E8EDF5]">
                Welcome back
                <span className="sr-only">AUTHENTICATE ANALYST</span>
              </h1>
              <p className="text-sm text-[#91A0B8] leading-relaxed" style={{ marginTop: '8px' }}>
                Sign in to continue to your deepfake detection workspace.
              </p>
            </div>

            {serverError && (
              <div
                role="alert"
                className="rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5"
                style={{ padding: '12px 14px', marginBottom: '20px' }}
              >
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <div className="flex-1 leading-relaxed">{serverError}</div>
              </div>
            )}

            {/* Form Elements */}
            <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col" style={{ gap: '18px' }} noValidate>
              {/* Email Field */}
              <div className="flex flex-col" style={{ gap: '6px' }}>
                <label
                  htmlFor="auth-email"
                  className="block text-xs font-medium text-[#91A0B8]"
                >
                  Email address <span className="sr-only">Operator Email</span>
                </label>
                <div className="relative flex items-center w-full">
                  <input
                    id="auth-email"
                    type="email"
                    autoComplete="email"
                    placeholder="analyst@parallax.io"
                    aria-invalid={!!errors.email}
                    aria-describedby={errors.email ? 'auth-email-error' : undefined}
                    style={{
                      paddingLeft: '14px',
                      paddingRight: '14px',
                      height: '44px',
                    }}
                    className={`w-full bg-[#090D14] text-[#E8EDF5] placeholder-[#91A0B8]/50 text-sm rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-[#648CFF] focus:border-transparent ${
                      errors.email ? 'border-red-500/60' : 'border-[#253247] hover:border-[#374866]'
                    }`}
                    {...register('email')}
                  />
                </div>
                {errors.email && (
                  <p id="auth-email-error" role="alert" className="text-xs text-red-400 flex items-center gap-1.5" style={{ marginTop: '4px' }}>
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.email.message}</span>
                  </p>
                )}
              </div>

              {/* Password Field */}
              <div className="flex flex-col" style={{ gap: '6px' }}>
                <label
                  htmlFor="auth-password"
                  className="block text-xs font-medium text-[#91A0B8]"
                >
                  Password <span className="sr-only">Password Phrase</span>
                </label>
                <div className="relative flex items-center w-full">
                  <input
                    id="auth-password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    placeholder="••••••••••••"
                    aria-invalid={!!errors.password}
                    aria-describedby={errors.password ? 'auth-password-error' : undefined}
                    style={{
                      paddingLeft: '14px',
                      paddingRight: '44px',
                      height: '44px',
                    }}
                    className={`w-full bg-[#090D14] text-[#E8EDF5] placeholder-[#91A0B8]/50 text-sm rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-[#648CFF] focus:border-transparent ${
                      errors.password ? 'border-red-500/60' : 'border-[#253247] hover:border-[#374866]'
                    }`}
                    {...register('password')}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password text' : 'Show password text'}
                    style={{ padding: '8px' }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-[#91A0B8] hover:text-[#E8EDF5] rounded transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {errors.password && (
                  <p id="auth-password-error" role="alert" className="text-xs text-red-400 flex items-center gap-1.5" style={{ marginTop: '4px' }}>
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.password.message}</span>
                  </p>
                )}
              </div>

              {/* Submit Button */}
              <div style={{ paddingTop: '6px' }}>
                <button
                  type="submit"
                  disabled={isLoggingIn}
                  style={{ height: '44px', padding: '0 20px' }}
                  className="w-full bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white font-medium text-sm rounded-lg transition-all duration-150 flex items-center justify-center gap-2 shadow-sm hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] focus-visible:ring-offset-2 focus-visible:ring-offset-[#111827] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  {isLoggingIn ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Signing in...</span>
                    </>
                  ) : (
                    <>
                      <span>Sign In</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Registration Navigation Link */}
            <div
              className="border-t border-[#253247] text-center text-xs text-[#91A0B8]"
              style={{ marginTop: '24px', paddingTop: '18px' }}
            >
              <span>Don't have an account? </span>
              <Link
                to={`/register${next ? `?next=${encodeURIComponent(next)}` : ''}`}
                className="font-medium text-[#648CFF] hover:text-[#88A7FF] hover:underline transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] rounded"
              >
                Create account
              </Link>
            </div>
          </div>

          {/* Security Footnote */}
          <div
            className="flex items-center justify-center gap-2 text-center text-xs text-[#91A0B8]"
            style={{ marginTop: '20px' }}
          >
            <ShieldCheck className="w-4 h-4 text-[#648CFF] shrink-0" />
            <span>Argon2id zero-knowledge hashing · 256-bit rotating sessions</span>
          </div>
        </div>
      </section>
    </main>
  );
};

