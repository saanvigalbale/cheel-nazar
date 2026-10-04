import React from 'react';
import { Activity, Clock, CircleDashed, Terminal } from 'lucide-react';

interface Stage {
  id: string;
  name: string;
  module: string;
  status: 'pending' | 'active' | 'completed';
  description: string;
}

export const ProcessingStatus: React.FC = () => {
  const stages: Stage[] = [
    {
      id: '01',
      name: 'Video & Telemetry Ingestion',
      module: 'Ingestion Pipeline',
      status: 'pending',
      description: 'Validate video codecs, extract embedded GPS/IMU metadata & timestamps',
    },
    {
      id: '02',
      name: 'Keyframe Sampling & Matching',
      module: 'OpenCV / SIFT / LightGlue',
      status: 'pending',
      description: 'Filter motion blur, select optimal parallax keyframes, match feature points',
    },
    {
      id: '03',
      name: 'Sparse & Dense 3D SfM',
      module: 'COLMAP / Open3D',
      status: 'pending',
      description: 'Camera bundle adjustment, camera pose recovery, dense point cloud triangulation',
    },
    {
      id: '04',
      name: 'AI Object & Hazard Analysis',
      module: 'YOLO / Depth Anything',
      status: 'pending',
      description: 'Detect vehicles, damaged structures, road obstacles, and terrain hazards in 3D',
    },
    {
      id: '05',
      name: 'Georeferencing & Export',
      module: 'GIS / WGS84 Projection',
      status: 'pending',
      description: 'Align coordinates to UTM/WGS84, produce georeferenced orthomosaic and 3D tiles',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            Processing Status
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-xs font-mono text-amber-400 bg-amber-950/40 border border-amber-500/30 px-2 py-0.5 rounded">
            <Clock className="w-3 h-3 animate-spin" />
            STANDBY / IDLE
          </span>
        </div>
      </div>

      {/* Stepper Pipeline */}
      <div className="space-y-3">
        {stages.map((stage) => (
          <div
            key={stage.id}
            className="flex items-start gap-3 p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 hover:border-slate-700 transition-colors"
          >
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-slate-900 border border-slate-700 text-slate-400 font-mono text-xs font-bold shrink-0 mt-0.5">
              {stage.id}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1">
                <span className="text-sm font-medium text-slate-200">{stage.name}</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-cyan-400 w-fit">
                  {stage.module}
                </span>
              </div>
              <p className="text-xs text-slate-500">{stage.description}</p>
            </div>

            <div className="shrink-0 flex items-center text-slate-600">
              <CircleDashed className="w-4 h-4" />
            </div>
          </div>
        ))}
      </div>

      {/* Execution Console Log placeholder */}
      <div className="mt-4 p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-400">
        <div className="flex items-center justify-between text-slate-500 text-[11px] pb-1 border-b border-slate-800 mb-2">
          <div className="flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>MISSION EXECUTION LOG</span>
          </div>
          <span>[SYSTEM LOGS]</span>
        </div>
        <p className="text-slate-500">
          [00:00:00] <span className="text-emerald-400">INFO:</span> Cheel Nazar backend pipeline initialized in standby mode.
        </p>
        <p className="text-slate-500">
          [00:00:00] <span className="text-slate-400">WAIT:</span> Awaiting drone video stream ingestion from operator...
        </p>
      </div>
    </div>
  );
};
