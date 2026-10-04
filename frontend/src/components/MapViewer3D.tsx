import React, { useState } from 'react';
import { Box, Compass, Crosshair, Layers, Maximize2, RotateCw, ZoomIn, ZoomOut, Eye } from 'lucide-react';
import { isPlaceholderResults } from '../api/client.ts';
import type { JobResults } from '../types/job.ts';

interface MapViewer3DProps {
  isComplete: boolean;
  results: JobResults | null;
  resultsLoading: boolean;
}

export const MapViewer3D: React.FC<MapViewer3DProps> = ({ isComplete, results, resultsLoading }) => {
  const [renderMode, setRenderMode] = useState<'points' | 'mesh' | 'ortho'>('points');

  // Reconstruction / georeferencing are not implemented; current results are mock.
  const georefUnavailable = isPlaceholderResults(results);
  const geo = georefUnavailable ? undefined : results?.georeference;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm flex flex-col h-full">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Box className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            3D Map & Point Cloud Viewer
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
            THREE.JS READY
          </span>
        </div>
      </div>

      {/* Viewport Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 bg-slate-950/80 border border-slate-800 rounded-lg p-2 text-xs font-mono">
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500 mr-1 text-[11px]">RENDER:</span>
          {(['points', 'mesh', 'ortho'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setRenderMode(mode)}
              className={`px-2.5 py-1 rounded text-xs uppercase transition-all ${
                renderMode === mode
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 font-semibold'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {mode === 'points' ? 'Point Cloud' : mode === 'mesh' ? '3D Mesh' : 'Orthomosaic'}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 text-slate-400">
          <button title="Zoom In" className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800">
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button title="Zoom Out" className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800">
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button title="Reset Orientation" className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800">
            <RotateCw className="w-3.5 h-3.5" />
          </button>
          <button title="Fullscreen" className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800">
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main 3D Viewport Placeholder Canvas */}
      <div className="relative flex-1 min-h-[360px] rounded-lg border border-slate-800 bg-[#070a10] overflow-hidden tactical-grid flex items-center justify-center">
        <div className="radar-sweep" />

        {/* Tactical Crosshair Center */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-30">
          <Crosshair className="w-16 h-16 text-cyan-500" />
        </div>

        {/* Top-Right Compass / Orientation Gizmo Placeholder */}
        <div className="absolute top-3 right-3 p-2 rounded-lg bg-slate-950/80 border border-slate-800 text-cyan-400 flex flex-col items-center shadow-md">
          <Compass className="w-7 h-7 text-cyan-400" />
          <span className="text-[10px] font-mono font-bold mt-1 text-slate-300">N 000°</span>
        </div>

        {/* Top-Left Telemetry Coordinates Overlay */}
        <div className="absolute top-3 left-3 p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 font-mono text-[11px] text-slate-400 space-y-0.5 shadow-md">
          <div className="text-cyan-400 font-semibold text-xs flex items-center gap-1.5 mb-1">
            <Layers className="w-3.5 h-3.5" />
            <span>GEOREFERENCE OVERLAY</span>
          </div>
          <div>CRS: <span className="text-slate-200">EPSG:4326 (WGS 84 / UTM)</span></div>
          <div>LAT: <span className="text-slate-200">{geo ? `${geo.lat.toFixed(6)}° N` : '--.------° N'}</span></div>
          <div>LON: <span className="text-slate-200">{geo ? `${geo.lon.toFixed(6)}° E` : '--.------° E'}</span></div>
          <div>ALT (MSL): <span className="text-slate-200">{geo ? `${geo.altitude_m.toFixed(1)} m` : '---.- m'}</span></div>
          <div>STATUS: <span className="text-amber-400">{georefUnavailable ? 'NOT IMPLEMENTED' : 'AVAILABLE'}</span></div>
        </div>

        {/* Center UI Placeholder Notification */}
        <div className="relative z-10 flex flex-col items-center justify-center text-center p-6 max-w-sm rounded-xl bg-slate-950/90 border border-slate-800/80 shadow-2xl backdrop-blur-md">
          <div className="p-3 rounded-full bg-cyan-950/50 border border-cyan-500/30 text-cyan-400 mb-3">
            <Eye className="w-6 h-6 animate-pulse" />
          </div>
          <h3 className="text-sm font-semibold text-slate-100 font-mono tracking-wide uppercase mb-1">
            3D Viewport Standby
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed mb-3">
            {isComplete
              ? 'Reconstruction is not implemented yet - no point cloud or mesh is available to display.'
              : 'Three.js / WebGL visualization layer initialized. Waiting for reconstructed point cloud or textured mesh input.'}
          </p>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-700 text-[10px] font-mono text-cyan-300">
            <span>
              {resultsLoading
                ? 'LOADING RESULTS'
                : isComplete
                ? 'RECONSTRUCTION NOT IMPLEMENTED'
                : 'AWAITING SFM OUTPUT'}
            </span>
          </div>
        </div>

        {/* Bottom Status Bar */}
        <div className="absolute bottom-2 left-3 right-3 flex items-center justify-between text-[10px] font-mono text-slate-500">
          <span>POINTS: 0 | DENSE VERTICES: 0 | FACES: 0</span>
          <span>FPS: 60 | DRAW_CALLS: 0</span>
        </div>
      </div>
    </div>
  );
};
