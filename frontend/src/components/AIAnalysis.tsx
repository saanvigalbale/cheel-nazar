import React from 'react';
import {
  AlertTriangle,
  Building2,
  Droplets,
  Loader2,
  Mountain,
  Route,
  Sparkles,
  Sprout,
  Tag,
  Truck,
  User,
} from 'lucide-react';
import type { JobResults, SemanticCategory } from '../types/job.ts';

interface AIAnalysisProps {
  results: JobResults | null;
  resultsLoading: boolean;
  resultsError: string | null;
  isComplete: boolean;
  isProcessing: boolean;
}

function formatConfidence(value: number | null | undefined): string {
  return value == null ? '--' : value.toFixed(3);
}

interface SemanticCardProps {
  title: string;
  data: SemanticCategory;
  Icon: React.ComponentType<{ className?: string }>;
  accent: string;
  barClass: string;
}

const SemanticCard: React.FC<SemanticCardProps> = ({ title, data, Icon, accent, barClass }) => (
  <div className="p-3.5 rounded-lg border border-slate-800 bg-slate-950/50">
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${accent}`} />
        <span className="text-xs font-semibold text-slate-200 uppercase font-mono">{title}</span>
      </div>
      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
        SEMANTIC
      </span>
    </div>
    <div className="text-lg font-bold font-mono text-slate-100 mb-1.5">
      {data.coverage_pct.toFixed(2)}%
    </div>
    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
      <div
        className={`h-full ${barClass}`}
        style={{ width: `${Math.min(100, Math.max(0, data.coverage_pct))}%` }}
      />
    </div>
  </div>
);

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
      <div className="text-slate-500 text-[10px] leading-tight">{label}</div>
      <div className="text-slate-100 font-semibold mt-0.5">{value}</div>
    </div>
  );
}

export const AIAnalysis: React.FC<AIAnalysisProps> = ({
  results,
  resultsLoading,
  resultsError,
  isComplete,
  isProcessing,
}) => {
  const statusMessage = isProcessing
    ? 'Analysis pipeline running (YOLO11n, SegFormer-B0, Depth Anything V2)...'
    : !isComplete
    ? 'Upload drone footage and start processing to run analysis.'
    : resultsLoading
    ? 'Loading real analysis results...'
    : null;

  const flood = results?.disasters?.flood;

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
          YOLO11N + SEGFORMER
        </span>
      </div>

      {/* Model provenance (from the real results payload) */}
      {results && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-4 font-mono text-[11px]">
          <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
            <div className="text-slate-500 text-[10px]">DETECTOR</div>
            <div className="text-slate-200">{results.models.detector}</div>
          </div>
          <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
            <div className="text-slate-500 text-[10px]">SEGMENTATION</div>
            <div className="text-slate-200">{results.models.segmentation}</div>
          </div>
          <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
            <div className="text-slate-500 text-[10px]">DEPTH</div>
            <div className="text-slate-200">{results.models.depth}</div>
          </div>
        </div>
      )}

      {/* Status / error states */}
      {statusMessage && (
        <div className="p-4 mb-4 rounded-lg bg-slate-950/70 border border-slate-800 flex items-center gap-2.5 text-xs text-slate-400 font-mono">
          {resultsLoading ? (
            <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
          ) : (
            <Tag className="w-4 h-4 text-slate-500" />
          )}
          <span>{statusMessage}</span>
        </div>
      )}

      {!statusMessage && resultsError && (
        <div className="p-3 mb-4 rounded-lg bg-rose-950/40 border border-rose-500/50 flex items-start gap-2.5 text-xs text-rose-300">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <span>{resultsError}</span>
        </div>
      )}

      {!statusMessage && !resultsError && results && (
        <>
          {/* Detected categories (real YOLO11n) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
            <div className="p-3.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-semibold text-slate-200 uppercase font-mono">People</span>
                </div>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                  DETECTED
                </span>
              </div>
              <div className="text-lg font-bold font-mono text-slate-100 mb-1">
                {results.categories.people.count}
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 font-mono text-[11px] text-slate-500">
                <span>
                  AVG CONF: <span className="text-slate-300">{formatConfidence(results.categories.people.avg_confidence)}</span>
                </span>
                <span>
                  FRAMES: <span className="text-slate-300">{results.categories.people.frames_with_detections}</span>
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Truck className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-semibold text-slate-200 uppercase font-mono">Vehicles</span>
                </div>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                  DETECTED
                </span>
              </div>
              <div className="text-lg font-bold font-mono text-slate-100 mb-1">
                {results.categories.vehicles.count}
              </div>
              {Object.keys(results.categories.vehicles.breakdown).length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {Object.entries(results.categories.vehicles.breakdown).map(([label, count]) => (
                    <span
                      key={label}
                      className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300"
                    >
                      {label}: {count}
                    </span>
                  ))}
                </div>
              )}
              <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 font-mono text-[11px] text-slate-500">
                <span>
                  AVG CONF: <span className="text-slate-300">{formatConfidence(results.categories.vehicles.avg_confidence)}</span>
                </span>
                <span>
                  FRAMES: <span className="text-slate-300">{results.categories.vehicles.frames_with_detections}</span>
                </span>
              </div>
            </div>
          </div>

          {/* Semantic scene coverage (real SegFormer) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
            <SemanticCard title="Structure Coverage" data={results.categories.structures} Icon={Building2} accent="text-cyan-400" barClass="bg-cyan-400" />
            <SemanticCard title="Road Coverage" data={results.categories.roads} Icon={Route} accent="text-amber-400" barClass="bg-amber-400" />
            <SemanticCard title="Vegetation Coverage" data={results.categories.vegetation} Icon={Sprout} accent="text-emerald-400" barClass="bg-emerald-400" />
            <SemanticCard title="Terrain Coverage" data={results.categories.terrain} Icon={Mountain} accent="text-orange-400" barClass="bg-orange-400" />
          </div>

          {/* Water proxy (semantic estimate - explicitly not a flood measurement) */}
          <div className="p-3.5 rounded-lg border border-blue-500/40 bg-blue-950/20 mb-4">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <Droplets className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-semibold text-slate-200 uppercase font-mono">
                  Possible Water Extent
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950/60 border border-amber-500/40 text-amber-300">
                ESTIMATE / PROXY
              </span>
            </div>
            <div className="text-lg font-bold font-mono text-slate-100">
              {results.categories.possible_water_extent.coverage_pct.toFixed(2)}%
            </div>
            <p className="text-[10px] text-slate-400 leading-tight mt-1.5">
              {results.categories.possible_water_extent.disclaimer}
            </p>
          </div>

          {/* Relative depth summary (non-metric) */}
          {results.relative_depth && (
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 font-mono text-[11px] text-slate-400 mb-4">
              RELATIVE DEPTH (mean):{' '}
              <span className="text-cyan-300">{results.relative_depth.mean_relative_depth ?? '--'}</span>{' '}
              - 0-255 relative scale (not metric). Maps: {results.relative_depth.maps_generated ?? 0}
            </div>
          )}
        </>
      )}

      {/* ══ FLOOD INDICATORS (Phase 6B.1) — ESTIMATE / PROXY ══ */}
      {flood && flood.aggregate.frames_analyzed > 0 && (
        <div className="p-3.5 rounded-lg border border-blue-500/40 bg-blue-950/10 mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Droplets className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-semibold text-slate-200 uppercase font-mono">
                Flood Indicators
              </span>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950/60 border border-amber-500/40 text-amber-300">
              ESTIMATE / PROXY
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2 font-mono text-[11px]">
            <StatCell label="Water extent" value={`${flood.aggregate.water_extent_pct ?? '--'} %`} />
            <StatCell
              label="Water-present frame ratio"
              value={`${flood.aggregate.water_present_frame_ratio ?? '--'} %`}
            />
            <StatCell
              label="Mean water coverage"
              value={`${flood.aggregate.mean_water_coverage_pct ?? '--'} %`}
            />
            <StatCell
              label="Median water coverage"
              value={`${flood.aggregate.median_water_coverage_pct ?? '--'} %`}
            />
            <StatCell
              label="Max water coverage"
              value={`${flood.aggregate.max_water_coverage_pct ?? '--'} %`}
            />
            <StatCell
              label="Water-present frames"
              value={`${flood.aggregate.water_present_frames ?? '--'} / ${flood.aggregate.frames_analyzed}`}
            />
            <StatCell
              label="People observed in water"
              value={`${flood.aggregate.people_frame_instances_in_water ?? 0} frame instances`}
            />
            <StatCell
              label="People observed near water"
              value={`${flood.aggregate.people_frame_instances_near_water ?? 0} frame instances`}
            />
            <StatCell
              label="Vehicles observed in water"
              value={`${flood.aggregate.vehicles_frame_instances_in_water ?? 0} frame instances`}
            />
            <StatCell
              label="Vehicles observed near water"
              value={`${flood.aggregate.vehicles_frame_instances_near_water ?? 0} frame instances`}
            />
          </div>
          <p className="text-[10px] text-slate-500 leading-tight mb-2">
            Frame instances, not unique objects — {flood.count_semantics}
          </p>
          {/* Possible water near road — spatial-proximity proxy, never a flooded-road claim */}
          <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 font-mono text-[11px] text-slate-400 mb-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-blue-300">POSSIBLE WATER NEAR ROAD</span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-950/60 border border-amber-500/40 text-amber-300">
                PROXY / ESTIMATE
              </span>
            </div>

            {flood.aggregate.possible_water_near_road_pct == null ? (
              <span className="text-amber-300">
                UNAVAILABLE — no road pixels were detected in the analyzed frames.
              </span>
            ) : (
              <div>
                <span className="text-slate-100 font-semibold">
                  {flood.aggregate.possible_water_near_road_pct} %
                </span>{' '}
                of road-classified pixels lie within{' '}
                {flood.parameters.water_near_road_buffer_px} px of water-classified pixels
                {flood.aggregate.road_pixels_near_water != null && (
                  <span className="text-slate-500">
                    {' '}({flood.aggregate.road_pixels_near_water} road pixels in range)
                  </span>
                )}
              </div>
            )}

            <p className="text-slate-500 leading-tight mt-1">
              Spatial proximity only — detected water pixels are close to detected road
              pixels. This does NOT confirm that any road is flooded, and is not a flood
              severity or flood-extent measurement.
            </p>
            {flood.aggregate.water_near_road_semantics && (
              <p className="text-slate-500 leading-tight mt-1">
                {flood.aggregate.water_near_road_semantics}
              </p>
            )}
          </div>

          {/* Relative depth evidence — explicitly not metres */}
          {flood.aggregate.relative_depth_evidence && (
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 font-mono text-[11px] text-slate-400 mb-2">
              <div className="text-cyan-300 mb-1">RELATIVE DEPTH EVIDENCE — NOT METRES</div>
              <div>
                MEAN IN WATER:{' '}
                {flood.aggregate.relative_depth_evidence.mean_relative_depth_in_water ?? '--'}
              </div>
              <div>
                MEAN SCENE DEPTH:{' '}
                {flood.aggregate.relative_depth_evidence.mean_relative_depth_scene ?? '--'}
              </div>
              <div>
                WITHIN-FRAME SEPARATION:{' '}
                {flood.aggregate.relative_depth_evidence.mean_within_frame_depth_separation ?? '--'}
              </div>
              <div className="mt-1">
                METRIC DEPTH AVAILABLE:{' '}
                <span className="text-amber-300">
                  {flood.aggregate.relative_depth_evidence.metric_depth_available ? 'YES' : 'NO'}
                </span>
                {' · '}frames with evidence:{' '}
                {flood.aggregate.relative_depth_evidence.frames_with_water_depth_evidence}
              </div>
            </div>
          )}

          {/* Temporal trend */}
          {flood.aggregate.temporal_trend && (
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 font-mono text-[11px] text-slate-400 mb-2">
              TEMPORAL TREND:{' '}
              <span className="text-cyan-300">
                {flood.aggregate.temporal_trend.label.toUpperCase()}
              </span>
              {flood.aggregate.temporal_trend.frames_compared != null && (
                <span className="text-slate-500">
                  {' · '}frames compared {flood.aggregate.temporal_trend.frames_compared}
                </span>
              )}
              {flood.aggregate.temporal_trend.absolute_change_pct_points != null && (
                <span className="text-slate-500">
                  {' · '}absolute change{' '}
                  {flood.aggregate.temporal_trend.absolute_change_pct_points} pct points
                </span>
              )}
              {flood.aggregate.temporal_trend.note && (
                <p className="text-slate-500 leading-tight mt-1">
                  {flood.aggregate.temporal_trend.note}
                </p>
              )}
            </div>
          )}

          {/* Visible disclaimer (never hidden behind a tooltip) */}
          <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-500/40 text-[10px] text-amber-200 leading-relaxed">
            <span className="font-semibold block mb-1">Semantic water is a proxy</span> and may represent
            rivers, lakes, pools, or other water. It is not a confirmed flood boundary, flooded area
            measurement, or metric water-depth measurement.
            <span className="block mt-1 text-amber-300/80">{flood.disclaimer}</span>
          </div>
        </div>
      )}

      {flood && flood.aggregate.frames_analyzed === 0 && (
        <div className="p-3.5 rounded-lg border border-slate-800 bg-slate-950/60 mb-4 font-mono text-[11px] text-slate-400">
          FLOOD INDICATORS: insufficient data —{' '}
          <span className="text-slate-500">{flood.aggregate.note ?? 'no frames analyzed'}</span>
        </div>
      )}

      {/* Hazard modules actually analyzed vs still pending */}
      {(results?.disaster_modules_available?.length ?? 0) > 0 && (
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 font-mono text-[11px] text-slate-400 mb-4">
          DISASTER MODULES ANALYZED:{' '}
          <span className="text-emerald-300">
            {(results?.disaster_modules_available ?? []).join(', ')}
          </span>
          <div className="text-slate-500 mt-0.5">
            NOT IMPLEMENTED: {(results?.disaster_modules_not_implemented ?? []).join(', ')}
          </div>
        </div>
      )}

      {/* Unavailable-metrics footer (honest: nothing is fabricated) */}
      <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs">
        <div className="flex items-center gap-2 mb-1.5 text-slate-400 font-mono text-[11px]">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          <span>NOT MEASURED (require 3D / georeferencing / fine-tuning):</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(
            results?.not_measured ?? [
              'metric_water_depth',
              'georeferenced_area',
              'structural_damage',
              'debris_extent',
              'landslide_extent',
              'fire_extent',
              'hazard_zones',
              'georeferenced_coordinates',
            ]
          ).map((item) => (
            <span
              key={item}
              className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-500"
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
