/**
 * Forensic video upload dropzone.
 * Built to instrument-panel standards: viewfinder corners, mono readouts, client-side pre-checks.
 */

import React, { useRef, useState, useCallback } from 'react';
import { Button } from '../ui/Button';

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  isUploading?: boolean;
  disabled?: boolean;
}

const MAX_SIZE_MB = 200;
const MAX_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm', '.mkv'];

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelected,
  isUploading = false,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = useCallback(
    (file: File) => {
      setError(null);
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setError(`Unsupported format. Expected: ${ACCEPTED_EXTENSIONS.join(', ')}`);
        return;
      }
      if (file.size > MAX_BYTES) {
        setError(`File size exceeds ${MAX_SIZE_MB}MB limit.`);
        return;
      }
      if (file.size === 0) {
        setError('Selected file is empty (0 bytes).');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled || isUploading) return;
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
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
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && !isUploading && inputRef.current?.click()}
      className={`relative flex flex-col items-center justify-center p-8 sm:p-12 rounded-lg border transition-colors cursor-pointer ${
        isDragOver
          ? 'border-cobalt-500 bg-black-800/80'
          : 'border-line hover:border-line-strong bg-black-900/60'
      } ${disabled || isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {/* Viewfinder corner crop marks */}
      <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-cobalt-500/60" />
      <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-cobalt-500/60" />
      <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-cobalt-500/60" />
      <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-cobalt-500/60" />

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(',')}
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled || isUploading}
      />

      {/* Sensor icon / viewfinder reticle */}
      <div className="w-12 h-12 rounded-full border border-cobalt-500/40 flex items-center justify-center mb-4 text-cobalt-400">
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      </div>

      <div className="font-mono text-xs uppercase tracking-widest text-cobalt-300 mb-2">
        CH-IN // SENSOR DROPZONE
      </div>

      <p className="font-display text-lg text-text-100 text-center font-medium mb-1">
        {isUploading ? 'Streaming payload to forensic pipeline…' : 'Drop video clip or select local file'}
      </p>

      <div className="font-mono text-[11px] text-text-500 uppercase tracking-wider mb-6 text-center">
        MP4, MOV, WEBM, AVI · MAX {MAX_SIZE_MB}MB · UP TO 60 SEC
      </div>

      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={disabled || isUploading}
        loading={isUploading}
      >
        BROWSE LOCAL DRIVE
      </Button>

      {error && (
        <div
          role="alert"
          className="mt-4 font-mono text-xs text-alert-500 border border-alert-500/30 bg-alert-500/10 px-3 py-1.5 rounded"
        >
          ERR // {error}
        </div>
      )}
    </div>
  );
};
