import React, { useRef, useState, useCallback } from 'react';
import { clsx } from 'clsx';
import { Upload, AlertCircle, FolderOpen } from 'lucide-react';

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  isUploading?: boolean;
  disabled?: boolean;
  className?: string;
  onErrorChange?: (error: string | null) => void;
}

const MAX_SIZE_MB = 200;
const MAX_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm', '.mkv'];

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelected,
  isUploading = false,
  disabled = false,
  className,
  onErrorChange,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const inactive = disabled || isUploading;

  const setValidationError = useCallback(
    (message: string | null) => {
      setError(message);
      onErrorChange?.(message);
    },
    [onErrorChange],
  );

  const validateAndSelect = useCallback(
    (file: File) => {
      setValidationError(null);
      const extension = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`;

      if (!ACCEPTED_EXTENSIONS.includes(extension)) {
        setValidationError(
          `Unsupported container ${extension.toUpperCase()}. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}.`,
        );
        return;
      }
      if (file.size > MAX_BYTES) {
        setValidationError(`File exceeds the ${MAX_SIZE_MB} MB ingest limit.`);
        return;
      }
      if (file.size === 0) {
        setValidationError('Selected file is empty (0 bytes).');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected, setValidationError],
  );

  const openFileDialog = () => {
    if (!inactive) inputRef.current?.click();
  };

  return (
    <div role="region" aria-label="Media upload" className={className}>
      {/* The entire surface is one native control: click, Enter and Space all
          open the picker, and drag events share the same hit area. */}
      <button
        type="button"
        onClick={openFileDialog}
        disabled={inactive}
        aria-describedby="dropzone-formats"
        onDragOver={(event) => {
          event.preventDefault();
          if (!inactive) setIsDragOver(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragOver(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragOver(false);
          if (inactive) return;
          const file = event.dataTransfer.files?.[0];
          if (file) validateAndSelect(file);
        }}
        className={clsx(
          'w-full min-h-[248px] rounded-[var(--radius-cell)] border border-dashed cursor-pointer select-none',
          'flex flex-col items-center justify-center text-center px-6 py-8',
          'transition-colors duration-[var(--motion-normal)]',
          isDragOver
            ? 'border-[var(--cobalt-400)] bg-[var(--surface-elevated)]'
            : 'border-[var(--line-strong)] bg-[var(--surface-card)] hover:border-[var(--cobalt-500)]/70 hover:bg-[var(--surface-elevated)]',
          inactive && 'opacity-50 cursor-not-allowed',
        )}
      >
        <span className="w-11 h-11 rounded-[var(--radius-lg)] bg-[var(--surface-base)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] mb-3">
          <Upload size={19} aria-hidden="true" />
        </span>

        <span className="sr-only">Upload video file for forensic deepfake analysis.</span>

        <span className="text-[15px] font-semibold text-[var(--text-100)]">
          {isDragOver ? 'Release to stage this file' : 'Drop a video, or choose a file'}
        </span>

        <span className="text-[12px] text-[var(--text-300)] mt-1">
          The clip is decoded and analysed inside the backend worker.
        </span>

        <span className="mt-4 inline-flex items-center gap-2 h-9 px-3.5 rounded-[var(--radius-md)] bg-[var(--cobalt-500)] text-[var(--text-100)] text-[12px] font-semibold">
          <FolderOpen size={14} aria-hidden="true" />
          Browse files
        </span>

        <span
          id="dropzone-formats"
          className="mt-4 font-mono text-[10px] text-[var(--text-400)] tracking-wide"
        >
          MP4 · MOV · WEBM · AVI · MKV — max {MAX_SIZE_MB} MB
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(',')}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        disabled={inactive}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) validateAndSelect(file);
          event.target.value = '';
        }}
      />

      {error && (
        <p
          role="alert"
          className="mt-3 inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--alert-500)]/40 bg-[var(--alert-bg)] px-3 py-2 text-[12px] text-[var(--text-200)]"
        >
          <AlertCircle size={14} className="text-[var(--alert-400)] shrink-0" aria-hidden="true" />
          {error}
        </p>
      )}
    </div>
  );
};
