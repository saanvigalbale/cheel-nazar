import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header.tsx';
import { VideoUploader } from './components/VideoUploader.tsx';
import { ProcessingStatus } from './components/ProcessingStatus.tsx';
import { MapViewer3D } from './components/MapViewer3D.tsx';
import { AIAnalysis } from './components/AIAnalysis.tsx';
import { Measurements } from './components/Measurements.tsx';
import { ConfidenceMetric } from './components/ConfidenceMetric.tsx';
import { Terminal } from 'lucide-react';

export function App() {
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);

  const checkHealth = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      // Connect to FastAPI backend health endpoint
      const response = await fetch('http://127.0.0.1:8000/health', {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (response.ok) {
        setBackendHealthy(true);
      } else {
        setBackendHealthy(false);
      }
    } catch {
      setBackendHealthy(false);
    } finally {
      setIsCheckingHealth(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    // Auto-poll health every 15 seconds
    const timer = setInterval(checkHealth, 15000);
    return () => clearInterval(timer);
  }, [checkHealth]);

  return (
    <div className="min-h-screen bg-[#0a0d14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation & Status Bar */}
      <Header
        backendHealthy={backendHealthy}
        onRefreshHealth={checkHealth}
        isCheckingHealth={isCheckingHealth}
      />

      {/* Main Dashboard Grid */}
      <main className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1600px] w-full mx-auto">
        {/* Pipeline & Ingestion Section */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5">
            <VideoUploader />
          </div>
          <div className="lg:col-span-7">
            <ProcessingStatus />
          </div>
        </section>

        {/* 3D Map Viewport & AI Hazard Detection */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <MapViewer3D />
          </div>
          <div className="lg:col-span-5">
            <AIAnalysis />
          </div>
        </section>

        {/* Tactical Measurements & Uncertainty Analysis */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <Measurements />
          </div>
          <div className="lg:col-span-5">
            <ConfidenceMetric />
          </div>
        </section>
      </main>

      {/* Footer / Telemetry Bar */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 px-4 lg:px-8 py-3 text-xs font-mono text-slate-500 mt-auto">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 max-w-[1600px] mx-auto">
          <div className="flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>CHEEL NAZAR RECONNAISSANCE ENGINE • PROTOTYPE FOUNDATION</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>FRONTEND: REACT 18 + VITE + TS</span>
            <span>BACKEND: FASTAPI + UVICORN</span>
            <span className="text-cyan-400">READY FOR PIPELINE INTEGRATION</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
