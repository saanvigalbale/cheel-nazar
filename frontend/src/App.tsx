import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header.tsx';
import { VideoUploader } from './components/VideoUploader.tsx';
import { ProcessingStatus } from './components/ProcessingStatus.tsx';
import { MapViewer3D } from './components/MapViewer3D.tsx';
import { AIAnalysis } from './components/AIAnalysis.tsx';
import { Measurements } from './components/Measurements.tsx';
import { ConfidenceMetric } from './components/ConfidenceMetric.tsx';
import { Terminal } from 'lucide-react';
import {
  ApiError,
  getJobReconstruction,
  getJobResults,
  getJobStatus,
  healthCheck,
  startProcessing,
} from './api/client.ts';
import type {
  JobReconstruction,
  JobResults,
  JobStatusResponse,
} from './types/job.ts';

/** Polling cadence for job status while the pipeline is running. */
const POLL_INTERVAL_MS = 2500;

const IN_PROGRESS_STATUSES: JobStatusResponse['status'][] = [
  'queued',
  'extracting_frames',
  'detecting_objects',
  'segmenting_scene',
  'estimating_depth',
  'analyzing_hazards',
  'reconstructing',
  'georeferencing',
];

export function App() {
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);
  const [activeJob, setActiveJob] = useState<{ job_id: string; filename: string } | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [resultsLoading, setResultsLoading] = useState<boolean>(false);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [pollError, setPollError] = useState<string | null>(null);
  const [reconstruction, setReconstruction] = useState<JobReconstruction | null>(null);
  const [reconstructionLoading, setReconstructionLoading] = useState<boolean>(false);
  const [reconstructionError, setReconstructionError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      await healthCheck();
      setBackendHealthy(true);
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

  // Poll job status while the pipeline is running.
  useEffect(() => {
    if (!isPolling || !activeJob) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getJobStatus(activeJob.job_id);
        if (cancelled) return;
        setJobStatus(status);
        setPollError(null);
        // Stop polling once the job reaches a terminal state.
        if (status.status === 'done' || status.status === 'failed') {
          setIsPolling(false);
        }
      } catch (err) {
        if (cancelled) return;
        setPollError(
          err instanceof ApiError && err.status === 404
            ? 'Job not found on the backend.'
            : 'Lost connection to the backend while polling job status.',
        );
        setIsPolling(false);
      }
    };

    poll();
    const timer = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [isPolling, activeJob]);

  const jobStatusValue = jobStatus?.status ?? null;

  // Fetch job results once processing has completed.
  useEffect(() => {
    if (!activeJob || jobStatusValue !== 'done') return;

    let cancelled = false;
    setResultsLoading(true);
    setResultsError(null);

    getJobResults(activeJob.job_id)
      .then((data) => {
        if (!cancelled) setResults(data);
      })
      .catch((err) => {
        if (cancelled) return;
        setResults(null);
        setResultsError(
          err instanceof ApiError && err.status === 404
            ? 'Results are not available yet.'
            : 'Failed to load job results.',
        );
      })
      .finally(() => {
        if (!cancelled) setResultsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeJob, jobStatusValue]);

  // Fetch the real 3D reconstruction (Phases 8.1-8.6) once a job is known.
  // Re-fetched while the reconstruct stage is running so the viewer picks the
  // cloud up as soon as it exists.
  useEffect(() => {
    if (!activeJob) {
      setReconstruction(null);
      setReconstructionError(null);
      return;
    }

    let cancelled = false;
    setReconstructionLoading(true);

    getJobReconstruction(activeJob.job_id)
      .then((data) => {
        if (!cancelled) {
          setReconstruction(data);
          setReconstructionError(null);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setReconstruction(null);
        setReconstructionError(
          err instanceof ApiError && err.status === 404
            ? 'No reconstruction data for this job on the backend.'
            : 'Could not reach the reconstruction endpoint on the backend.',
        );
      })
      .finally(() => {
        if (!cancelled) setReconstructionLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeJob, isPolling]);

  // Store the freshly uploaded job and reset any previous run state.
  const handleUploadSuccess = useCallback(
    async (result: { job_id: string; filename: string }) => {
      setActiveJob({ job_id: result.job_id, filename: result.filename });
      setJobStatus(null);
      setResults(null);
      setResultsError(null);
      setStartError(null);
      setPollError(null);
      setIsPolling(false);
      try {
        const status = await getJobStatus(result.job_id);
        setJobStatus(status);
      } catch {
        // Status will be populated once processing starts.
      }
    },
    [],
  );

  // POST /api/v1/jobs/{job_id}/process, then begin polling.
  const handleStartProcessing = useCallback(async () => {
    if (!activeJob || isStarting) return;
    setIsStarting(true);
    setStartError(null);
    setPollError(null);
    try {
      await startProcessing(activeJob.job_id);
      setIsPolling(true);
    } catch (err) {
      // 409 → the job is already running; just resume polling.
      if (err instanceof ApiError && err.status === 409) {
        setIsPolling(true);
      } else {
        setStartError(
          err instanceof ApiError ? err.detail : 'Failed to start processing.',
        );
      }
    } finally {
      setIsStarting(false);
    }
  }, [activeJob, isStarting]);

  const isProcessing = jobStatusValue !== null && IN_PROGRESS_STATUSES.includes(jobStatusValue);
  const isComplete = jobStatusValue === 'done';
  const processingError =
    jobStatusValue === 'failed'
      ? jobStatus?.error ?? 'Processing failed.'
      : startError ?? pollError;

  return (
    <div className="min-h-screen bg-[#0a0d14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation & Status Bar */}
      <Header
        backendHealthy={backendHealthy}
        onRefreshHealth={checkHealth}
        isCheckingHealth={isCheckingHealth}
        pipelineStatus={jobStatusValue}
        progress={jobStatus?.progress ?? null}
      />

      {/* Main Dashboard Grid */}
      <main className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1600px] w-full mx-auto">
        {/* Pipeline & Ingestion Section */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5">
            <VideoUploader
              activeJobId={activeJob?.job_id}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>
          <div className="lg:col-span-7">
            <ProcessingStatus
              jobStatus={jobStatus}
              hasActiveJob={activeJob !== null}
              isStarting={isStarting}
              isPolling={isPolling}
              errorMessage={processingError}
              onStartProcessing={handleStartProcessing}
              reconstruction={reconstruction}
            />
          </div>
        </section>

        {/* 3D Map Viewport & AI Hazard Detection */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <MapViewer3D
              isComplete={isComplete}
              results={results}
              resultsLoading={resultsLoading}
              jobId={activeJob?.job_id ?? null}
              reconstruction={reconstruction}
              reconstructionLoading={reconstructionLoading}
              reconstructionError={reconstructionError}
              isReconstructing={
                jobStatusValue === 'reconstructing' || jobStatusValue === 'georeferencing'
              }
            />
          </div>
          <div className="lg:col-span-5">
            <AIAnalysis
              results={results}
              resultsLoading={resultsLoading}
              resultsError={resultsError}
              isComplete={isComplete}
              isProcessing={isProcessing}
            />
          </div>
        </section>

        {/* Tactical Measurements & Uncertainty Analysis */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <Measurements
              isComplete={isComplete}
              results={results}
              resultsLoading={resultsLoading}
              resultsError={resultsError}
            />
          </div>
          <div className="lg:col-span-5">
            <ConfidenceMetric
              isComplete={isComplete}
              results={results}
              resultsLoading={resultsLoading}
              resultsError={resultsError}
            />
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
            <span className="text-cyan-400">PIPELINE INTEGRATION: PHASE 7</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
