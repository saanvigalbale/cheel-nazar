import React from 'react';
import { UploadCloud, Video, FileText, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

export const VideoUploader: React.FC = () => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
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

      {/* Dropzone Placeholder */}
      <div className="relative group border-2 border-dashed border-slate-700/80 hover:border-cyan-500/50 rounded-lg p-6 flex flex-col items-center justify-center text-center transition-all bg-slate-950/40 cursor-not-allowed">
        <div className="p-3 rounded-full bg-cyan-950/40 border border-cyan-500/30 text-cyan-400 mb-3 group-hover:scale-105 transition-transform">
          <UploadCloud className="w-8 h-8" />
        </div>

        <p className="text-sm font-medium text-slate-200 mb-1">
          Select or Drag & Drop Single-Pass Drone Video
        </p>
        <p className="text-xs text-slate-500 max-w-sm mb-4">
          Supported container formats: <span className="font-mono text-slate-400">MP4, MOV, MKV</span> (H.264/H.265, 4K/1080p aerial feed)
        </p>

        {/* Telemetry attachment hint */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-900/90 border border-slate-800 text-slate-400">
            <FileText className="w-3.5 h-3.5 text-cyan-400" />
            <span>GPS/IMU Subtitles (.SRT)</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-900/90 border border-slate-800 text-slate-400">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Flight Log (.CSV/.GPX)</span>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-1.5 text-[11px] text-amber-400/90 font-mono">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>UI Placeholder: Ingestion endpoints will connect in the next phase.</span>
        </div>
      </div>

      {/* Action CTA placeholder */}
      <div className="mt-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <CheckCircle2 className="w-3.5 h-3.5 text-slate-600" />
          <span>Ready for high-bitrate video stream upload</span>
        </div>
        <button
          disabled
          className="px-4 py-2 rounded-lg bg-cyan-600/30 border border-cyan-500/40 text-cyan-300 font-mono text-xs font-semibold uppercase tracking-wider cursor-not-allowed opacity-75"
        >
          Begin Ingestion
        </button>
      </div>
    </div>
  );
};
