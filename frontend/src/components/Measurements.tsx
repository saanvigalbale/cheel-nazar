import React from 'react';
import { Ruler, Maximize, Mountain, Box, MapPin } from 'lucide-react';

export const Measurements: React.FC = () => {
  const tools = [
    {
      name: 'Euclidean Distance',
      icon: Ruler,
      unit: 'm',
      currentValue: '--.--',
      desc: 'Point-to-point 3D measurement',
    },
    {
      name: 'Surface / Perimeter Area',
      icon: Maximize,
      unit: 'm²',
      currentValue: '---.--',
      desc: 'Polygon footprint calculation',
    },
    {
      name: 'Volumetric Rubble / Crater',
      icon: Box,
      unit: 'm³',
      currentValue: '----.--',
      desc: '3D volume estimation via mesh surface',
    },
    {
      name: 'Elevation Profile',
      icon: Mountain,
      unit: 'm MSL',
      currentValue: '---.-',
      desc: 'Terrain cut & fill cross-section',
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
          METRIC WGS84
        </span>
      </div>

      {/* Measurement Toolset Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {tools.map((tool, idx) => {
          const Icon = tool.icon;
          return (
            <div
              key={idx}
              className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-slate-300">{tool.name}</span>
                <Icon className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="my-2">
                <span className="text-xl font-bold font-mono text-slate-100">{tool.currentValue}</span>
                <span className="text-xs font-mono text-slate-500 ml-1">{tool.unit}</span>
              </div>
              <p className="text-[10px] text-slate-500 leading-tight">{tool.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Interactive Tool Selector Bar Placeholder */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-400">
          <MapPin className="w-3.5 h-3.5 text-cyan-400" />
          <span>TOOL: <span className="text-slate-200">Point Picker (Disabled in Standby)</span></span>
        </div>
        <span className="text-amber-400/90 text-[11px]">
          Click-to-measure will activate when 3D model vertices load.
        </span>
      </div>
    </div>
  );
};
