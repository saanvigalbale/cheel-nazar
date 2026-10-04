import React from 'react';
import { AlertTriangle, MapPin, Maximize, Mountain, Ruler } from 'lucide-react';
import { isPlaceholderResults } from '../api/client.ts';
import type { JobResults } from '../types/job.ts';

interface MeasurementsProps {
  isComplete: boolean;
  results: JobResults | null;
  resultsLoading: boolean;
  resultsError: string | null;
}

function formatMetric(value: number | undefined, unit: string): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value.toFixed(2)} ${unit}`;
}

export const Measurements: React.FC<MeasurementsProps> = ({
  isComplete,
  results,
  resultsLoading,
  resultsError,
}) => {
  // Backend results are still mock (model_url === null) until reconstruction is
  // implemented, so they must never be presented as real measurements.
  const unavailable = isPlaceholderResults(results);
  const measurements = unavailable ? undefined : results?.measurements;

  const notice = resultsLoading
    ? 'Loading job results...'
    : resultsError
    ? resultsError
    : !isComplete
    ? 'Measurements will be available once the pipeline completes.'
    : 'Measurements unavailable - reconstruction not yet implemented.';

  const tools = [
    {
      name: 'Scene Area',
      icon: Maximize,
      unit: 'm²',
      value: measurements?.area_sq_m,
      desc: 'Footprint area from reconstruction',
    },
    {
      name: 'Scene Width',
      icon: Ruler,
      unit: 'm',
      value: measurements?.scene_width_m,
      desc: 'Reconstructed scene width',
    },
    {
      name: 'Scene Length',
      icon: Ruler,
      unit: 'm',
      value: measurements?.scene_length_m,
      desc: 'Reconstructed scene length',
    },
    {
      name: 'Max Building Height',
      icon: Mountain,
      unit: 'm',
      value: measurements?.max_building_height_m,
      desc: 'Tallest structure height',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Ruler className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            Geospatial & Volumetric Measurements
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
          {unavailable ? 'NOT AVAILABLE' : 'METRIC WGS84'}
        </span>
      </div>

      {/* Availability notice — mock results are never shown as real */}
      {unavailable && (
        <div className="p-3 mb-4 rounded-lg bg-amber-950/30 border border-amber-500/40 flex items-start gap-2.5 text-xs text-amber-300">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <span>{notice}</span>
        </div>
      )}

      {/* Measurement Toolset Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {tools.map((tool) => {
          const Icon = tool.icon;
          return (
            <div
              key={tool.name}
              className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-slate-300">{tool.name}</span>
                <Icon className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="my-2">
                <span className="text-xl font-bold font-mono text-slate-100">
                  {formatMetric(tool.value, tool.unit)}
                </span>
              </div>
              <p className="text-[10px] text-slate-500 leading-tight">{tool.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Interactive Tool Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-400">
          <MapPin className="w-3.5 h-3.5 text-cyan-400" />
          <span>TOOL: <span className="text-slate-200">Point Picker (Disabled)</span></span>
        </div>
        <span className="text-amber-400/90 text-[11px]">
          {unavailable
            ? 'Click-to-measure requires reconstructed geometry (not implemented).'
            : 'Click-to-measure activates once 3D model vertices load.'}
        </span>
      </div>
    </div>
  );
};
