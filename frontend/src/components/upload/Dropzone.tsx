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

  const setValidationError = useCallback(
    (msg: string | null) => {
      setError(msg);
      onErrorChange?.(msg);
    },
    [onErrorChange],
  );

  const validateAndSelect = useCallback(
    (file: File) => {
      setValidationError(null);
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setValidationError(`Unsupported file format. Accepted formats: ${ACCEPTED_EXTENSIONS.join(', ')}`);
        return;
      }
      if (file.size > MAX_BYTES) {
        setValidationError(`File size exceeds server-enforced limit of ${MAX_SIZE_MB} MB.`);
        return;
      }
      if (file.size === 0) {
        setValidationError('Selected media file is empty (0 bytes).');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected, setValidationError],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled || isUploading) return;
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled || isUploading) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndSelect(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndSelect(file);
    }
    e.target.value = '';
  };

  const openFileDialog = () => {
    if (!disabled && !isUploading) {
      inputRef.current?.click();
    }
  };

  return (
    <div
      role="region"
      aria-label="Media upload dropzone"
      tabIndex={disabled || isUploading ? -1 : 0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openFileDialog();
        }
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={openFileDialog}
      className={clsx(
        'w-full min-h-[260px] sm:min-h-[275px] max-h-[300px] rounded-[14px] border border-dashed transition-all duration-200 cursor-pointer select-none flex flex-col items-center justify-center p-6 text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]',
        isDragOver
          ? 'border-[#648CFF] bg-[#172338]/60 shadow-[0_0_24px_rgba(100,140,255,0.12)]'
          : 'border-[#34445C] bg-[#111927] hover:border-[#4B5E7D] hover:bg-[#131B29]',
        (disabled || isUploading) && 'opacity-50 cursor-not-allowed',
        className,
      )}
    >
      <label htmlFor="media-file-input" className="sr-only">
        Choose video file
      </label>
      <input
        ref={inputRef}
        type="file"
        id="media-file-input"
        accept={ACCEPTED_EXTENSIONS.join(',')}
        className="sr-only"
        onChange={handleFileChange}
        disabled={disabled || isUploading}
        tabIndex={-1}
      />

      {/* 1. Compact upload icon container (48-52px, 22-24px icon) */}
      <div className="w-12 h-12 rounded-[12px] bg-[#171F2D] border border-[#253247] flex items-center justify-center text-[#648CFF] mb-3 transition-transform duration-200">
        <Upload className="w-6 h-6" />
      </div>

      {/* 2. Main instruction */}
      <h2 className="text-base sm:text-[17px] font-semibold text-[#E8EDF5] tracking-tight mb-1">
        {isDragOver ? 'Drop your video to upload' : 'Drop your video here'}
      </h2>

      {/* 3. Supporting text */}
      <p className="text-xs sm:text-[13px] text-[#91A0B8] mb-4">
        or browse files from your device
      </p>

      {/* 4. Browse Files button (content-fit, 38-40px height, 13px font) */}
      <button
        type="button"
        disabled={disabled || isUploading}
        onClick={(e) => {
          e.stopPropagation();
          openFileDialog();
        }}
        aria-label="Browse Files — Upload video file for forensic deepfake analysis"
        className="h-[38px] w-fit px-4 rounded-[8px] text-[13px] font-medium bg-[#3B66EB] hover:bg-[#3258D6] active:bg-[#284BBB] text-white inline-flex items-center gap-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] disabled:opacity-50 disabled:pointer-events-none cursor-pointer shadow-xs"
      >
        <FolderOpen className="w-4 h-4" />
        <span>Browse Files</span>
        <span className="sr-only">Upload video file for forensic deepfake analysis</span>
      </button>

      {/* 5. Supported file information (contrast >= 4.5:1 against #111927) */}
      <div className="mt-4 text-[11px] font-mono text-[#91A0B8] tracking-tight">
        MP4, MOV, WEBM, AVI, MKV · Max {MAX_SIZE_MB} MB · Up to 60s
      </div>

      {/* Validation feedback if invalid file was selected */}
      {error && (
        <div
          role="alert"
          aria-live="polite"
          onClick={(e) => e.stopPropagation()}
          className="mt-3.5 inline-flex items-center gap-2 text-xs font-sans text-red-300 bg-red-950/50 border border-red-500/30 px-3.5 py-1.5 rounded-[8px]"
        >
          <AlertCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
