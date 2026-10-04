import React, { useEffect, useState } from 'react';
import { Eye, Shield, Activity, Radio, Cpu, RefreshCw } from 'lucide-react';

interface HeaderProps {
  backendHealthy: boolean | null;
  onRefreshHealth: () => void;
  isCheckingHealth: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  backendHealthy,
  onRefreshHealth,
  isCheckingHealth,
}) => {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        {/* Brand & Project Identity */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-lg bg-cyan-950/60 border border-cyan-500/40 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.25)]">
            <Eye className="w-5 h-5 text-cyan-400 animate-pulse" />
            <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-wider text-slate-100 uppercase font-mono">
                Cheel Nazar
              </h1>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-700/50 text-cyan-400 font-semibold">
                v0.1.0-alpha
              </span>
            </div>
            <p className="text-xs text-slate-400 tracking-wide">
              Single-Pass Drone Reconnaissance • 3D Geospatial Reconstruction • AI Hazard Detection
            </p>
          </div>
        </div>

        {/* Telemetry, Status & Live UTC Clock */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800 text-slate-300">
            <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span>SYS_CLK:</span>
            <span className="text-cyan-400 font-semibold">{time || 'SYNCHRONIZING...'}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">PIPELINE:</span>
            <span className="text-amber-400 font-semibold">STANDBY</span>
          </div>

          {/* Backend Health Badge */}
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-colors ${
              backendHealthy === true
                ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                : backendHealthy === false
                ? 'bg-rose-950/40 border-rose-500/50 text-rose-300'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
          >
            <Activity
              className={`w-3.5 h-3.5 ${
                backendHealthy === true ? 'text-emerald-400 animate-pulse' : 'text-slate-400'
              }`}
            />
            <span className="font-semibold">
              API: {backendHealthy === true ? 'ONLINE' : backendHealthy === false ? 'OFFLINE' : 'CHECKING...'}
            </span>
            <button
              onClick={onRefreshHealth}
              disabled={isCheckingHealth}
              title="Ping Backend /health"
              className="ml-1 p-0.5 hover:text-white transition-transform active:rotate-180 disabled:opacity-50"
            >
              <RefreshCw className={`w-3 h-3 ${isCheckingHealth ? 'animate-spin' : ''}`} />
            </button>
          </div>

          <div className="hidden lg:flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
            <Shield className="w-3 h-3 text-cyan-400" />
            <span>SEC_LVL: 1</span>
          </div>
        </div>
      </div>
    </header>
  );
};
