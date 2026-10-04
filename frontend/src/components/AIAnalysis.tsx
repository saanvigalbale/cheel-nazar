import React from 'react';
import { Sparkles, ShieldAlert, Car, Construction, Flame, Filter } from 'lucide-react';

export const AIAnalysis: React.FC = () => {
  const hazardCategories = [
    {
      title: 'Structural Damage',
      icon: Construction,
      color: 'text-rose-400',
      bgColor: 'bg-rose-950/30',
      borderColor: 'border-rose-500/30',
      description: 'Roof collapse, wall breach, facade fractures',
      status: 'AWAITING RECON',
    },
    {
      title: 'Road Blockages & Craters',
      icon: ShieldAlert,
      color: 'text-amber-400',
      bgColor: 'bg-amber-950/30',
      borderColor: 'border-amber-500/30',
      description: 'Fallen trees, cratering, impassable routes',
      status: 'AWAITING RECON',
    },
    {
      title: 'Vehicle & Asset Detection',
      icon: Car,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-950/30',
      borderColor: 'border-cyan-500/30',
      description: 'Military/civilian vehicles, heavy equipment',
      status: 'AWAITING RECON',
    },
    {
      title: 'Thermal & Debris Hazards',
      icon: Flame,
      color: 'text-orange-400',
      bgColor: 'bg-orange-950/30',
      borderColor: 'border-orange-500/30',
      description: 'Unstable rubble heaps, hotspots, spills',
      status: 'AWAITING RECON',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            AI Hazard & Damage Analysis
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
          YOLO & DEPTH READY
        </span>
      </div>

      {/* Filter and Sensitivity Controls Placeholder */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 mb-4 rounded-lg bg-slate-950/70 border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-400">
          <Filter className="w-3.5 h-3.5 text-cyan-400" />
          <span>CONFIDENCE THRESHOLD:</span>
          <span className="text-cyan-300 font-semibold">0.70</span>
        </div>
        <div className="flex items-center gap-2 text-slate-500">
          <span>MODEL: <span className="text-slate-300">YOLOv11-Recon (Standby)</span></span>
        </div>
      </div>

      {/* Hazard Cards Grid Placeholder */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        {hazardCategories.map((cat, idx) => {
          const Icon = cat.icon;
          return (
            <div
              key={idx}
              className={`p-3.5 rounded-lg border ${cat.borderColor} ${cat.bgColor} flex flex-col justify-between`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 ${cat.color}`} />
                  <span className="text-xs font-semibold text-slate-200 uppercase font-mono">
                    {cat.title}
                  </span>
                </div>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                  {cat.status}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
                {cat.description}
              </p>
              <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 font-mono text-[11px] text-slate-500">
                <span>DETECTED: 0</span>
                <span>CRITICALITY: --</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Action Footer */}
      <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 flex items-center justify-between text-xs">
        <span className="text-slate-400 font-mono text-[11px]">
          Target analysis pipeline will execute automatically once 3D frames are triangulated.
        </span>
        <button
          disabled
          className="px-3 py-1.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono text-xs cursor-not-allowed opacity-60"
        >
          Run Inference
        </button>
      </div>
    </div>
  );
};
