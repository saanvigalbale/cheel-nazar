import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import {
  UploadCloud,
  Video,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Layers,
  X,
  Loader2,
  Copy,
  Check,
  AlertCircle,
  Film
} from 'lucide-react';
import { API_BASE_URL } from '../api/client.ts';

const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv'];
const MAX_SIZE_MB = 500;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export interface UploadResult {
  job_id: string;
  filename: string;
  status: string;
  message: string;
}

interface VideoUploaderProps {
  onUploadSuccess?: (result: UploadResult) => void;
  activeJobId?: string | null;
}

function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onUploadSuccess,
  activeJobId,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    const nameLower = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => nameLower.endsWith(ext));
    if (!hasValidExt) {
      return `Unsupported file format. Please select: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    if (file.size === 0) {
      return 'The selected file is empty (0 bytes).';
    }
    if (file.size > MAX_SIZE_BYTES) {
      return `File exceeds maximum allowed size of ${MAX_SIZE_MB} MB.`;
    }
    return null;
  };

  const handleFileSelection = (file: File) => {
    setErrorMessage(null);
    setUploadResult(null);
    const error = validateFile(file);
    if (error) {
      setErrorMessage(error);
      setSelectedFile(null);
    } else {
      setSelectedFile(file);
    }
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isUploading) {
      setIsDragging(true);
    }
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isUploading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const triggerSelectFile = () => {
    if (!isUploading && fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleClearSelected = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setErrorMessage(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = () => {
    if (!selectedFile || isUploading) return;

    setErrorMessage(null);
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/api/v1/video/upload`);
    xhr.setRequestHeader('Accept', 'application/json');

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(percent);
      }
    };

    xhr.onload = () => {
      setIsUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const result: UploadResult = JSON.parse(xhr.responseText);
          setUploadResult(result);
          setSelectedFile(null);
          if (onUploadSuccess) {
            onUploadSuccess(result);
          }
        } catch {
          setErrorMessage('Failed to parse backend response.');
        }
      } else {
        try {
          const errorJson = JSON.parse(xhr.responseText);
          setErrorMessage(errorJson.detail || errorJson.message || `Upload failed (HTTP ${xhr.status})`);
        } catch {
          setErrorMessage(`Upload failed with status code ${xhr.status}: ${xhr.statusText || 'Server error'}`);
        }
      }
    };

    xhr.onerror = () => {
      setIsUploading(false);
      setErrorMessage(`Network error: Unable to connect to backend server at ${API_BASE_URL}. Ensure the FastAPI server is running.`);
    };

    xhr.send(formData);
  };

  const handleCopyJobId = (jobId: string) => {
    navigator.clipboard.writeText(jobId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleResetUpload = () => {
    setUploadResult(null);
    setSelectedFile(null);
    setErrorMessage(null);
    setUploadProgress(0);
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Video className="w-5 h-5 text-cyan-400" />
            <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
              Drone Footage Ingestion
            </h2>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
            STAGE 01
          </span>
        </div>

        {/* Hidden File Input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={onFileInputChange}
          accept=".mp4,.mov,.avi,.mkv"
          className="hidden"
          disabled={isUploading}
        />

        {/* Success View */}
        {uploadResult ? (
          <div className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 flex flex-col gap-4 text-left">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-100 font-mono uppercase tracking-wide">
                  {uploadResult.message || 'Video uploaded successfully'}
                </h3>
                <span className="text-xs text-emerald-400 font-mono">STATUS: {uploadResult.status.toUpperCase()}</span>
              </div>
            </div>

            <div className="space-y-2 bg-slate-950/80 border border-slate-800 rounded-lg p-3 text-xs font-mono">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 border-b border-slate-800/60 pb-2">
                <span className="text-slate-400">JOB_ID:</span>
                <div className="flex items-center gap-2">
                  <span className="text-cyan-300 font-semibold break-all">{uploadResult.job_id}</span>
                  <button
                    onClick={() => handleCopyJobId(uploadResult.job_id)}
                    title="Copy Job ID"
                    className="p-1 hover:text-white bg-slate-900 border border-slate-700 rounded text-slate-400 transition-colors"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-1 pt-1">
                <span className="text-slate-400 shrink-0">STORED_FILE:</span>
                <span className="text-slate-300 break-all text-right">{uploadResult.filename}</span>
              </div>
            </div>

            <button
              onClick={handleResetUpload}
              className="mt-1 w-full py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 font-mono text-xs font-semibold uppercase tracking-wider transition-colors"
            >
              Upload Another Video
            </button>
          </div>
        ) : (
          /* Dropzone / File Picker Area */
          <div
            onClick={triggerSelectFile}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={`relative group border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center transition-all cursor-pointer ${
              isDragging
                ? 'border-cyan-400 bg-cyan-950/30 scale-[1.01]'
                : selectedFile
                ? 'border-cyan-500/60 bg-slate-950/70'
                : 'border-slate-700/80 hover:border-cyan-500/50 bg-slate-950/40 hover:bg-slate-950/60'
            }`}
          >
            {/* Center Graphic */}
            <div
              className={`p-3 rounded-full mb-3 transition-transform ${
                selectedFile
                  ? 'bg-cyan-950/80 border border-cyan-400 text-cyan-300'
                  : 'bg-cyan-950/40 border border-cyan-500/30 text-cyan-400 group-hover:scale-105'
              }`}
            >
              {selectedFile ? <Film className="w-8 h-8" /> : <UploadCloud className="w-8 h-8" />}
            </div>

            {selectedFile ? (
              /* Selected File Details */
              <div className="w-full max-w-md">
                <div className="flex items-center justify-between bg-slate-900/90 border border-cyan-500/30 rounded-lg p-3 mb-2">
                  <div className="flex items-center gap-2.5 overflow-hidden text-left">
                    <Video className="w-4 h-4 text-cyan-400 shrink-0" />
                    <div className="overflow-hidden">
                      <p className="text-xs font-semibold text-slate-200 truncate font-mono">{selectedFile.name}</p>
                      <p className="text-[11px] text-slate-400 font-mono">{formatBytes(selectedFile.size)}</p>
                    </div>
                  </div>
                  {!isUploading && (
                    <button
                      onClick={handleClearSelected}
                      title="Clear Selection"
                      className="p-1 hover:text-rose-400 text-slate-400 rounded transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-cyan-400 font-mono">
                  Ready to stream to backend ingestion staging
                </p>
              </div>
            ) : (
              /* Idle Prompt */
              <>
                <p className="text-sm font-medium text-slate-200 mb-1">
                  Click or Drag & Drop Single-Pass Drone Video
                </p>
                <p className="text-xs text-slate-500 max-w-sm mb-4">
                  Supported formats: <span className="font-mono text-slate-400">MP4, MOV, AVI, MKV</span> (up to {MAX_SIZE_MB}MB)
                </p>

                {/* Telemetry hints */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md text-xs font-mono pointer-events-none">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800 text-slate-400">
                    <FileText className="w-3.5 h-3.5 text-cyan-400" />
                    <span>GPS/IMU (.SRT)</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800 text-slate-400">
                    <Layers className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Flight Track (.GPX/.CSV)</span>
                  </div>
                </div>
              </>
            )}

            {/* Upload Progress Bar */}
            {isUploading && (
              <div className="w-full max-w-md mt-4">
                <div className="flex items-center justify-between text-xs font-mono text-slate-300 mb-1">
                  <span className="flex items-center gap-1.5 text-cyan-400">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    UPLOADING VIDEO...
                  </span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-150"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Notification */}
        {errorMessage && (
          <div className="mt-3 p-3 rounded-lg bg-rose-950/40 border border-rose-500/50 flex items-start gap-2.5 text-xs text-rose-300 text-left">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold font-mono block mb-0.5">UPLOAD FAILED</span>
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Action Footer */}
      {!uploadResult && (
        <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-t border-slate-800/80 pt-3">
          <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
            {activeJobId ? (
              <span className="text-cyan-400">ACTIVE JOB: {activeJobId.substring(0, 8)}...</span>
            ) : (
              <span className="flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-slate-500" />
                <span>Select file to begin ingestion</span>
              </span>
            )}
          </div>

          <button
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className={`px-4 py-2 rounded-lg font-mono text-xs font-semibold uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
              !selectedFile || isUploading
                ? 'bg-slate-800 border border-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.3)] active:scale-95'
            }`}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Uploading {uploadProgress}%</span>
              </>
            ) : (
              <>
                <UploadCloud className="w-3.5 h-3.5" />
                <span>Upload Drone Video</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
