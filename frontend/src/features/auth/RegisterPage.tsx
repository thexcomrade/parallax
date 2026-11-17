import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { registerSchema, type RegisterFormData } from './authSchemas';
import { useAuth } from '@/hooks/useAuth';
import {
  ArrowRight,
  Check,
  Circle,
  Eye,
  EyeOff,
  AlertCircle,
  Loader2,
  ShieldCheck,
} from 'lucide-react';
import { ApiError } from '@/lib/api';

export const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const next = searchParams.get('next') || '/app';

  const { register: registerAnalyst, login, isRegistering } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

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
    mode: 'onSubmit',
  });

  const passwordVal = watch('password', '');

  // Live Password Strength Criteria
  const hasMinLength = passwordVal.length >= 12;
  const hasUpper = /[A-Z]/.test(passwordVal);
  const hasLower = /[a-z]/.test(passwordVal);
  const hasNumber = /[0-9]/.test(passwordVal);
  const hasSpecial = /[^A-Za-z0-9]/.test(passwordVal);

  const criteria = [
    { id: 'length', label: '12+ characters', passed: hasMinLength },
    { id: 'upper', label: 'Uppercase letter (A-Z)', passed: hasUpper },
    { id: 'lower', label: 'Lowercase letter (a-z)', passed: hasLower },
    { id: 'number', label: 'Number (0-9)', passed: hasNumber },
    { id: 'special', label: 'Special symbol (!@#$)', passed: hasSpecial },
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
    <main
      className="min-h-screen bg-[#090D14] text-[#E8EDF5] flex flex-col justify-center items-center relative overflow-hidden font-sans"
      style={{ padding: '48px 16px' }}
    >
      {/* Subtle background ambient lighting */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[640px] h-[480px] bg-[#648CFF]/5 rounded-full blur-3xl pointer-events-none" />
      </div>

      <div className="w-full max-w-[440px] relative z-10 flex flex-col items-center">
        {/* Brand Header */}
        <div className="flex flex-col items-center text-center" style={{ marginBottom: '28px' }}>
          <div
            className="inline-flex items-center gap-2 rounded-full bg-[#111827] border border-[#253247] shadow-sm"
            style={{ padding: '6px 14px', marginBottom: '16px' }}
          >
            <span className="flex h-2 w-2 relative">
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#648CFF]" />
            </span>
            <span className="text-[11px] font-bold tracking-widest text-[#E8EDF5] uppercase">
              PARALLAX
            </span>
            <span className="text-[10px] text-[#475569]">/</span>
            <span className="text-[11px] font-medium text-[#91A0B8]">
              Forensics
            </span>
          </div>

          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#E8EDF5]">
            Create Analyst Profile
            <span className="sr-only">CREATE ANALYST PROFILE</span>
          </h1>
          <p className="text-sm text-[#91A0B8] max-w-sm" style={{ marginTop: '8px', lineHeight: '1.5' }}>
            Sign up for zero-knowledge protected forensic analysis access
          </p>
        </div>

        {/* Card Container */}
        <div
          className="w-full bg-[#111827] border border-[#253247] rounded-2xl shadow-2xl relative z-10"
          style={{ padding: '32px' }}
        >
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

          <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col" style={{ gap: '18px' }} noValidate>
            {/* Email Field */}
            <div className="flex flex-col" style={{ gap: '6px' }}>
              <label
                htmlFor="operator-email"
                className="block text-xs font-medium text-[#91A0B8]"
              >
                Operator Email
              </label>
              <div className="relative flex items-center w-full">
                <input
                  id="operator-email"
                  type="email"
                  autoComplete="email"
                  placeholder="analyst@parallax.io"
                  aria-invalid={!!errors.email}
                  aria-describedby={errors.email ? 'email-error' : undefined}
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
                <p id="email-error" role="alert" className="text-xs text-red-400 flex items-center gap-1.5" style={{ marginTop: '4px' }}>
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{errors.email.message}</span>
                </p>
              )}
            </div>

            {/* Password Field */}
            <div className="flex flex-col" style={{ gap: '6px' }}>
              <label
                htmlFor="new-password"
                className="block text-xs font-medium text-[#91A0B8]"
              >
                New Password
              </label>
              <div className="relative flex items-center w-full">
                <input
                  id="new-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  placeholder="••••••••••••"
                  aria-invalid={!!errors.password}
                  aria-describedby={errors.password ? 'password-error' : undefined}
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
                <p id="password-error" role="alert" className="text-xs text-red-400 flex items-center gap-1.5" style={{ marginTop: '4px' }}>
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{errors.password.message}</span>
                </p>
              )}
            </div>

            {/* Password Validation & Entropy Strength Presentation */}
            <div
              className="rounded-xl bg-[#090D14]/80 border border-[#253247] flex flex-col"
              style={{ padding: '14px 16px', gap: '10px' }}
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold tracking-wider text-[#91A0B8] uppercase">
                  ENTROPY STRENGTH
                </span>
                <span
                  className={`text-xs font-semibold ${
                    passedCount === 5
                      ? 'text-emerald-400'
                      : passedCount >= 3
                      ? 'text-amber-400'
                      : 'text-[#91A0B8]'
                  }`}
                >
                  {passedCount === 5 ? 'OPTIMAL [5/5]' : `${passedCount}/5 Requirements Met`}
                </span>
              </div>

              {/* Progress bar with 5 segments */}
              <div className="grid grid-cols-5 gap-1.5 w-full" style={{ height: '6px' }}>
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <div
                    key={lvl}
                    className={`h-full rounded-full transition-colors duration-200 ${
                      passedCount >= lvl
                        ? passedCount === 5
                          ? 'bg-emerald-400'
                          : passedCount >= 3
                          ? 'bg-amber-400'
                          : 'bg-red-400'
                        : 'bg-[#253247]'
                    }`}
                  />
                ))}
              </div>

              {/* Individual Criteria Checklist */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2" style={{ paddingTop: '4px' }}>
                {criteria.map((c) => (
                  <div
                    key={c.id}
                    className={`flex items-center gap-2 text-xs transition-colors duration-150 ${
                      c.passed ? 'text-[#E8EDF5]' : 'text-[#91A0B8]'
                    }`}
                  >
                    {c.passed ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    ) : (
                      <Circle className="w-3.5 h-3.5 text-[#475569] shrink-0" />
                    )}
                    <span>{c.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Confirm Password Field */}
            <div className="flex flex-col" style={{ gap: '6px' }}>
              <label
                htmlFor="confirm-password"
                className="block text-xs font-medium text-[#91A0B8]"
              >
                Confirm Password
              </label>
              <div className="relative flex items-center w-full">
                <input
                  id="confirm-password"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  placeholder="••••••••••••"
                  aria-invalid={!!errors.confirmPassword}
                  aria-describedby={errors.confirmPassword ? 'confirm-password-error' : undefined}
                  style={{
                    paddingLeft: '14px',
                    paddingRight: '44px',
                    height: '44px',
                  }}
                  className={`w-full bg-[#090D14] text-[#E8EDF5] placeholder-[#91A0B8]/50 text-sm rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-[#648CFF] focus:border-transparent ${
                    errors.confirmPassword
                      ? 'border-red-500/60'
                      : 'border-[#253247] hover:border-[#374866]'
                  }`}
                  {...register('confirmPassword')}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  aria-label={showConfirmPassword ? 'Hide password text' : 'Show password text'}
                  style={{ padding: '8px' }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-[#91A0B8] hover:text-[#E8EDF5] rounded transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.confirmPassword && (
                <p
                  id="confirm-password-error"
                  role="alert"
                  className="text-xs text-red-400 flex items-center gap-1.5"
                  style={{ marginTop: '4px' }}
                >
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{errors.confirmPassword.message}</span>
                </p>
              )}
            </div>

            {/* Submit Button */}
            <div style={{ paddingTop: '6px' }}>
              <button
                type="submit"
                disabled={isRegistering}
                style={{ height: '44px', padding: '0 20px' }}
                className="w-full bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white font-medium text-sm rounded-lg transition-all duration-150 flex items-center justify-center gap-2 shadow-sm hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] focus-visible:ring-offset-2 focus-visible:ring-offset-[#111827] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                {isRegistering ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Completing Enrollment...</span>
                  </>
                ) : (
                  <>
                    <span>Complete Enrollment</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Navigation to Login */}
          <div
            className="border-t border-[#253247] text-center text-xs text-[#91A0B8]"
            style={{ marginTop: '20px', paddingTop: '18px' }}
          >
            <span>Already have an account? </span>
            <Link
              to={`/login${next ? `?next=${encodeURIComponent(next)}` : ''}`}
              className="font-medium text-[#648CFF] hover:text-[#88A7FF] hover:underline transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] rounded"
            >
              Sign in
            </Link>
          </div>
        </div>

        {/* Security & Credentials Note */}
        <div
          className="flex items-center justify-center gap-2 text-center text-xs text-[#91A0B8]"
          style={{ marginTop: '24px' }}
        >
          <ShieldCheck className="w-4 h-4 text-[#648CFF] shrink-0" />
          <span>Argon2id zero-knowledge hashing · 256-bit rotating sessions</span>
        </div>
      </div>
    </main>
  );
};
