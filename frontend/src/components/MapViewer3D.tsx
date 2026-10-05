import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Layers,
  Maximize2,
  RotateCw,
  ZoomIn,
  ZoomOut,
  Eye,
  AlertTriangle,
  Loader2,
  Ruler,
  Download,
} from 'lucide-react';
import {
  AmbientLight,
  Box3,
  BufferGeometry,
  Color,
  DirectionalLight,
  Float32BufferAttribute,
  GridHelper,
  Line,
  LineBasicMaterial,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { reconstructionArtifactUrl } from '../api/client.ts';
import type { JobReconstruction, JobResults, ReconstructionInfo } from '../types/job.ts';

interface MapViewer3DProps {
  isComplete: boolean;
  results: JobResults | null;
  resultsLoading: boolean;
  /** Active job whose reconstruction should be rendered, if any. */
  jobId: string | null;
  reconstruction: JobReconstruction | null;
  reconstructionLoading: boolean;
  reconstructionError: string | null;
  /** True while the pipeline is inside the reconstruct/georeference stages. */
  isReconstructing: boolean;
}

type ViewState = 'idle' | 'loading' | 'ready' | 'error';

interface SceneApi {
  fitView: () => void;
  zoom: (factor: number) => void;
  pick: (clientX: number, clientY: number) => { x: number; y: number; z: number } | null;
  getOrientation: () => { azimuth: number; elevation: number };
}

const countFormatter = new Intl.NumberFormat('en-US');

function formatCount(value: number | null | undefined): string {
  return typeof value === 'number' ? countFormatter.format(value) : '--';
}



export const MapViewer3D: React.FC<MapViewer3DProps> = ({
  isComplete,
  results,
  resultsLoading,
  jobId,
  reconstruction,
  reconstructionLoading,
  reconstructionError,
  isReconstructing,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneApiRef = useRef<SceneApi | null>(null);
  const measureLineRef = useRef<Line | null>(null);

  const [viewState, setViewState] = useState<ViewState>('idle');
  const [viewError, setViewError] = useState<string | null>(null);
  const [renderedPoints, setRenderedPoints] = useState<number | null>(null);
  const [cloudExtent, setCloudExtent] = useState<[number, number, number] | null>(null);
  const [orientation, setOrientation] = useState({ azimuth: 0, elevation: 0 });
  const [measurePoints, setMeasurePoints] = useState<
    { x: number; y: number; z: number }[]
  >([]);
  const [showHelp, setShowHelp] = useState(false);

  // The dedicated endpoint is authoritative, but results.json carries the same
  // reconstruction section, so it is a safe fallback.
  const activeReconstruction: JobReconstruction | ReconstructionInfo | null =
    reconstruction ?? results?.reconstruction ?? null;

  const artifactUrl = useMemo(() => {
    const url = activeReconstruction?.artifact?.url;
    if (!url) return null;
    return reconstructionArtifactUrl(url);
  }, [activeReconstruction]);

  const hasArtifact = Boolean(artifactUrl);
  const isFailed = activeReconstruction?.status === 'global_reconstruction_failed'
    || activeReconstruction?.status === 'dense_reconstruction_failed';

  // ── Scene lifecycle ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!artifactUrl || !containerRef.current) return;

    let disposed = false;

    const container = containerRef.current;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setViewState('error');
      setViewError(
        'WebGL is unavailable in this browser, so the 3D point cloud cannot be rendered.',
      );
      return;
    }

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';

    const scene = new Scene();
    scene.background = new Color(0x070a10);

    const camera = new PerspectiveCamera(55, width / height, 0.01, 10000);
    camera.position.set(3, 3, 3);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableRotate = true;
    controls.enableZoom = true;
    controls.enablePan = true;

    scene.add(new AmbientLight(0xffffff, 1.4));
    const key = new DirectionalLight(0xffffff, 1.1);
    key.position.set(4, 8, 6);
    scene.add(key);

    // Reference grid in LOCAL reconstruction units — a scale aid, not data.
    const grid = new GridHelper(40, 40, 0x1f6f8b, 0x14293a);
    (grid.material as { transparent: boolean; opacity: number }).transparent = true;
    (grid.material as { transparent: boolean; opacity: number }).opacity = 0.25;
    scene.add(grid);

    const raycaster = new Raycaster();
    const pointer = new Vector2();
    const measureLine = new Line(
      new BufferGeometry().setFromPoints([new Vector3(), new Vector3()]),
      new LineBasicMaterial({ color: 0xfbbf24 }),
    );
    measureLine.visible = false;
    measureLine.frustumCulled = false;
    scene.add(measureLine);
    measureLineRef.current = measureLine;

    const pointObjects: Points[] = [];
    let fitted = false;

    const fitView = () => {
      if (disposed || pointObjects.length === 0) return;
      const bounds = new Box3();
      pointObjects.forEach((points) => bounds.expandByObject(points));
      if (bounds.isEmpty()) return;

      const center = bounds.getCenter(new Vector3());
      const size = bounds.getSize(new Vector3());
      const radius = Math.max(size.length() * 0.5, 1e-6);

      const direction = new Vector3(1, 0.72, 1).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(radius * 2.6));
      camera.near = radius / 500;
      camera.far = radius * 200;
      camera.updateProjectionMatrix();

      controls.target.copy(center);
      controls.minDistance = radius * 0.05;
      controls.maxDistance = radius * 30;
      controls.update();

      grid.scale.setScalar(Math.max(radius, 1e-6) / 20);
      grid.position.set(center.x, bounds.min.y, center.z);

      setCloudExtent([size.x, size.y, size.z]);
    };

    const loader = new GLTFLoader();
    loader.load(
      artifactUrl,
      (gltf) => {
        if (disposed) return;

        let total = 0;
        gltf.scene.traverse((child) => {
          if (!(child as Points).isPoints) return;
          const points = child as Points;
          if (!points.material) {
            points.material = new PointsMaterial({ size: 0.02, sizeAttenuation: true });
          }
          const material = points.material as PointsMaterial;
          material.sizeAttenuation = true;
          material.needsUpdate = true;
          points.frustumCulled = false;
          pointObjects.push(points);
          total += points.geometry.getAttribute('position').count;
        });

        if (disposed) return;
        if (total === 0) {
          setViewState('error');
          setViewError('The artifact loaded but contains no point data to display.');
          return;
        }

        scene.add(gltf.scene);
        fitView();
        fitted = true;
        setRenderedPoints(total);
        setViewState('ready');
        setViewError(null);
      },
      undefined,
      (error) => {
        if (disposed) return;
        setViewState('error');
        const status = (error as { status?: number })?.status;
        setViewError(
          status === 404
            ? 'The reconstruction artifact was not found on the backend (404).'
            : 'Failed to load the reconstruction GLB from the backend.',
        );
      },
    );

    let frame = 0;
    const animate = () => {
      frame = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      if (fitted) {
        const offset = camera.position.clone().sub(controls.target);
        setOrientation({
          azimuth: (Math.atan2(offset.x, offset.z) * 180) / Math.PI,
          elevation: (Math.asin(offset.y / Math.max(offset.length(), 1e-6)) * 180) / Math.PI,
        });
      }
    };
    animate();


    sceneApiRef.current = {
      fitView,
      zoom: (factor) => {
        const offset = camera.position.clone().sub(controls.target);
        const next = Math.min(
          Math.max(offset.length() * factor, controls.minDistance),
          controls.maxDistance,
        );
        camera.position.copy(controls.target).add(offset.setLength(next));
        controls.update();
      },
      pick: (clientX, clientY) => {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(pointObjects, false)[0];
        if (!hit) return null;
        return { x: hit.point.x, y: hit.point.y, z: hit.point.z };
      },
      getOrientation: () => orientation,
    };

    // Keep the measurement polyline in sync with the picked points.
    const resizeObserver = new ResizeObserver(() => {
      if (!renderer.domElement.isConnected) return;
      const nextWidth = Math.max(container.clientWidth, 1);
      const nextHeight = Math.max(container.clientHeight, 1);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight, false);
    });
    resizeObserver.observe(container);
    const onWindowResize = () => fitView();
    window.addEventListener('resize', onWindowResize);

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', onWindowResize);
      resizeObserver.disconnect();
      sceneApiRef.current = null;
      controls.dispose();
      scene.traverse((child) => {
        const object = child as Points;
        object.geometry?.dispose();
        const material = object.material;
        if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
        else material?.dispose();
      });
      grid.geometry.dispose();
      (grid.material as { dispose: () => void }).dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [artifactUrl]);

  // ── Interactions ─────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    sceneApiRef.current?.fitView();
    setMeasurePoints([]);
  }, []);

  const handleZoom = useCallback((factor: number) => {
    sceneApiRef.current?.zoom(factor);
  }, []);

  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!showHelp) return;
    const hit = sceneApiRef.current?.pick(event.clientX, event.clientY);
    if (!hit) return;
    setMeasurePoints((previous) => (previous.length >= 2 ? [hit] : [...previous, hit]));
  }, [showHelp]);

  const relativeDistance = useMemo(() => {
    if (measurePoints.length !== 2) return null;
    const [a, b] = measurePoints;
    return Math.sqrt(
      (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2,
    );
  }, [measurePoints]);

  // Sync the measurement polyline whenever two points have been picked.
  useEffect(() => {
    const line = measureLineRef.current;
    if (!line) return;
    if (measurePoints.length !== 2) {
      line.visible = false;
      return;
    }
    const [a, b] = measurePoints;
    const geometry = new BufferGeometry();
    geometry.setAttribute(
      'position',
      new Float32BufferAttribute([a.x, a.y, a.z, b.x, b.y, b.z], 3),
    );
    line.geometry.dispose();
    line.geometry = geometry;
    line.visible = true;
  }, [measurePoints]);

  // Reset transient view state whenever the job changes.
  useEffect(() => {
    setMeasurePoints([]);
    setRenderedPoints(null);
    setCloudExtent(null);
    setViewError(null);
    setViewState(artifactUrl ? 'loading' : 'idle');
  }, [artifactUrl, jobId]);

  const banner = (() => {
    if (!jobId) {
      return { icon: <Eye className="w-6 h-6" />, title: 'No Active Job', body: 'Upload a video to load its 3D reconstruction.', tone: 'cyan' };
    }
    if (reconstructionLoading) {
      return { icon: <Loader2 className="w-6 h-6 animate-spin" />, title: 'Loading Reconstruction', body: 'Fetching reconstruction metadata for this job.', tone: 'cyan' };
    }
    if (reconstructionError) {
      return { icon: <AlertTriangle className="w-6 h-6" />, title: 'Backend Unavailable', body: reconstructionError, tone: 'red' };
    }
    if (isReconstructing) {
      return { icon: <Loader2 className="w-6 h-6 animate-spin" />, title: 'Reconstruction In Progress', body: 'The pipeline is currently building the point cloud.', tone: 'cyan' };
    }
    if (isFailed) {
      return {
        icon: <AlertTriangle className="w-6 h-6" />,
        title: 'Reconstruction Failed',
        body: 'The reconstruction stage reported a failure for this job. Check the backend logs for the exact reason.',
        tone: 'red',
      };
    }
    if (!hasArtifact) {
      return {
        icon: <Box className="w-6 h-6" />,
        title: 'Reconstruction Unavailable',
        body: 'This job has no reconstructed point cloud yet. Run the reconstruction stage, or process a job that has one.',
        tone: 'amber',
      };
    }
    if (viewState === 'loading') {
      return { icon: <Loader2 className="w-6 h-6 animate-spin" />, title: 'Loading Point Cloud', body: 'Streaming dense_points_global.glb from the backend.', tone: 'cyan' };
    }
    if (viewState === 'error') {
      return { icon: <AlertTriangle className="w-6 h-6" />, title: 'Point Cloud Unavailable', body: viewError ?? 'The 3D artifact could not be displayed.', tone: 'red' };
    }
    return null;
  })();

  const toneClasses: Record<string, string> = {
    cyan: 'bg-cyan-950/50 border-cyan-500/30 text-cyan-400',
    amber: 'bg-amber-950/40 border-amber-500/30 text-amber-400',
    red: 'bg-red-950/40 border-red-500/30 text-red-400',
  };

  // Downloadable artifacts. The dedicated endpoint returns `served` (name -> URL);
  // results.json only carries the primary artifact, so both are supported.
  const downloads = useMemo(() => {
    const served = activeReconstruction?.served;
    const entries: { name: string; url: string; mediaType: string }[] = [];
    if (served) {
      Object.entries(served).forEach(([name, url]) => {
        entries.push({ name, url, mediaType: name.endsWith('.glb') ? 'glTF binary' : name.split('.').pop()?.toUpperCase() ?? '' });
      });
    } else if (activeReconstruction?.artifact?.url) {
      entries.push({
        name: activeReconstruction.artifact.primary ?? 'dense_points_global.glb',
        url: activeReconstruction.artifact.url,
        mediaType: 'glTF binary',
      });
    }
    const seen = new Set<string>();
    return entries.filter((entry) => {
      const key = `${entry.name}|${entry.url}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [activeReconstruction]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm flex flex-col h-full">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Box className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            3D Reconstruction Viewer
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
            {viewState === 'ready' ? 'THREE.JS / GLTF' : 'THREE.JS READY'}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 bg-slate-950/80 border border-slate-800 rounded-lg p-2 text-xs font-mono">
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500 mr-1 text-[11px]">CONTROLS:</span>
          <span className="px-2.5 py-1 rounded text-[11px] uppercase bg-slate-900 text-slate-300 border border-slate-800">
            Drag = Orbit
          </span>
          <span className="px-2.5 py-1 rounded text-[11px] uppercase bg-slate-900 text-slate-300 border border-slate-800">
            Scroll = Zoom
          </span>
          <span className="px-2.5 py-1 rounded text-[11px] uppercase bg-slate-900 text-slate-300 border border-slate-800">
            Right-drag = Pan
          </span>
        </div>

        <div className="flex items-center gap-2 text-slate-400">
          <button
            title="Zoom In"
            onClick={() => handleZoom(0.8)}
            disabled={viewState !== 'ready'}
            className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800 disabled:opacity-40"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            title="Zoom Out"
            onClick={() => handleZoom(1.25)}
            disabled={viewState !== 'ready'}
            className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800 disabled:opacity-40"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            title="Reset / Recenter View"
            onClick={handleReset}
            disabled={viewState !== 'ready'}
            className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800 disabled:opacity-40"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </button>
          <button
            title="Toggle relative measurement"
            onClick={() => setShowHelp((previous) => !previous)}
            disabled={viewState !== 'ready'}
            className={`p-1 rounded border disabled:opacity-40 ${
              showHelp
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                : 'hover:text-cyan-400 bg-slate-900 border-slate-800'
            }`}
          >
            <Ruler className="w-3.5 h-3.5" />
          </button>
          <button
            title="Fullscreen viewport"
            onClick={() => containerRef.current?.parentElement?.requestFullscreen?.()}
            className="p-1 hover:text-cyan-400 bg-slate-900 rounded border border-slate-800"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>


      {/* Viewport */}
      <div
        ref={containerRef}
        onClick={handleCanvasClick}
        className="relative flex-1 min-h-[360px] rounded-lg border border-slate-800 bg-[#070a10] overflow-hidden tactical-grid"
        style={{ cursor: showHelp ? 'crosshair' : 'grab' }}
      >
        {banner && (
          <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
            <div className="flex flex-col items-center justify-center text-center p-6 max-w-sm rounded-xl bg-slate-950/90 border border-slate-800/80 shadow-2xl backdrop-blur-md">
              <div className={`p-3 rounded-full border mb-3 ${toneClasses[banner.tone]}`}>
                {banner.icon}
              </div>
              <h3 className="text-sm font-semibold text-slate-100 font-mono tracking-wide uppercase mb-1">
                {banner.title}
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">{banner.body}</p>
              {!jobId && !resultsLoading && !isComplete && (
                <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-700 text-[10px] font-mono text-cyan-300 mt-3">
                  AWAITING JOB
                </span>
              )}
            </div>
          </div>
        )}

        {/* Real camera orientation readout */}
        {viewState === 'ready' && (
          <div className="absolute top-3 right-3 p-2 rounded-lg bg-slate-950/80 border border-slate-800 font-mono shadow-md">
            <div className="text-[10px] text-slate-500">CAMERA ORIENTATION</div>
            <div className="text-cyan-400 font-semibold text-xs">
              AZ {orientation.azimuth.toFixed(0)}°
            </div>
            <div className="text-cyan-400 font-semibold text-xs">
              EL {orientation.elevation.toFixed(0)}°
            </div>
            <div className="text-[9px] text-slate-600 mt-0.5">LOCAL AXES</div>
          </div>
        )}

        {/* Honest reconstruction facts — no coordinates, no metric units */}
        {activeReconstruction && (
          <div className="absolute top-3 left-3 p-2.5 rounded-lg bg-slate-950/85 border border-slate-800 font-mono text-[11px] text-slate-400 space-y-0.5 shadow-md max-w-[240px]">
            <div className="text-cyan-400 font-semibold text-xs flex items-center gap-1.5 mb-1">
              <Layers className="w-3.5 h-3.5" />
              <span>RECONSTRUCTION</span>
            </div>
            <div>STATUS: <span className={activeReconstruction.status === 'success' ? 'text-emerald-400' : 'text-amber-400'}>{String(activeReconstruction.status).toUpperCase()}</span></div>
            <div>FRAMES: <span className="text-slate-200">{formatCount(activeReconstruction.n_frames_registered)}</span></div>
            <div>SPARSE PTS: <span className="text-slate-200">{formatCount(activeReconstruction.sparse_point_count)}</span></div>
            <div>DENSE PTS: <span className="text-slate-200">{formatCount(activeReconstruction.dense_point_count)}</span></div>
            <div className="pt-1 mt-1 border-t border-slate-800">
              <div>COORD SYS: <span className="text-slate-200">{activeReconstruction.coordinate_system}</span></div>
              <div>SCALE: <span className="text-amber-400">UP-TO-SCALE</span></div>
              <div>GEOREF: <span className="text-amber-400">{activeReconstruction.georeferenced ? 'YES' : 'NO'}</span></div>
              <div>METRIC: <span className="text-amber-400">UNAVAILABLE</span></div>
            </div>
          </div>
        )}


        {/* Relative measurement readout */}
        {showHelp && (
          <div className="absolute bottom-10 left-3 right-3 p-2.5 rounded-lg bg-slate-950/85 border border-amber-500/30 font-mono text-[11px] shadow-md">
            <div className="text-amber-400 font-semibold text-xs mb-1">
              RELATIVE MEASUREMENT (RECONSTRUCTION UNITS)
            </div>
            <div className="text-slate-400">
              {measurePoints.length === 0 && 'Click a point to start measuring.'}
              {measurePoints.length === 1 && 'Click a second point to measure the span.'}
              {relativeDistance !== null && (
                <span className="text-slate-200">
                  SPAN: <span className="text-amber-400 font-semibold">{relativeDistance.toFixed(3)}</span> reconstruction units
                </span>
              )}
            </div>
            <div className="text-[10px] text-slate-600 mt-1">
              Not a real-world distance. The reconstruction is up-to-scale and not
              georeferenced, so no metres are implied.
            </div>
            {measurePoints.length >= 1 && (
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  setMeasurePoints([]);
                }}
                className="mt-1.5 px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-[10px] text-slate-300 hover:text-amber-400"
              >
                CLEAR
              </button>
            )}
          </div>
        )}

        {/* Status bar — real counters only */}
        <div className="absolute bottom-2 left-3 right-3 flex items-center justify-between text-[10px] font-mono text-slate-500 pointer-events-none">
          <span>
            RENDERED PTS: {formatCount(renderedPoints)}
            {cloudExtent
              ? ` | EXTENT: ${cloudExtent.map((value) => value.toFixed(1)).join(' x ')} u`
              : ''}
          </span>
          <span>{viewState === 'ready' ? 'LOCAL UNITS - NOT METRIC' : ''}</span>
        </div>
      </div>

      {viewState === 'ready' && (
        <p className="mt-2 text-[10px] font-mono text-slate-600 leading-relaxed">
          Coordinates are local reconstruction units from an up-to-scale monocular
          reconstruction with heuristic (uncalibrated) intrinsics. They are not
          geographic positions and imply no real-world size.
        </p>
      )}

      {/* Export — real artifacts served by the backend */}
      <div className="mt-3 pt-3 border-t border-slate-800/80">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-mono uppercase tracking-wide text-slate-400">
            Export 3D Model
          </span>
          <span className="text-[10px] font-mono text-slate-600">UP-TO-SCALE</span>
        </div>

        {downloads.length === 0 ? (
          <p className="text-[11px] font-mono text-slate-500">
            No exportable artifact yet.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {downloads.map((entry) => {
              const absolute = reconstructionArtifactUrl(entry.url);
              const isPrimary = entry.name.endsWith('.glb');
              return absolute ? (
                <a
                  key={`${entry.name}-${absolute}`}
                  href={absolute}
                  download={entry.name}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-[11px] font-mono transition-colors ${
                    isPrimary
                      ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40 hover:bg-cyan-500/25'
                      : 'bg-slate-900 text-slate-300 border-slate-800 hover:text-cyan-300'
                  }`}
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>{isPrimary ? 'Download GLB' : `Download ${entry.name.split('.').pop()?.toUpperCase()}`}</span>
                </a>
              ) : null;
            })}
          </div>
        )}
      </div>
    </div>
  );
};
