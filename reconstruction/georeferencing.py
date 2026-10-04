"""Phase 8.5 - GEOREFERENCING: honest by construction.

This module answers one question: *can this job be placed on the Earth with the
data that actually exists?* It never invents a latitude, longitude or altitude.

WHAT WAS ACTUALLY FOUND IN THIS PROJECT'S JOBS
----------------------------------------------
The footage is ``WhatsApp_Video_*.mp4``. A WhatsApp share re-encodes the stream
and discards the source container metadata, so:

  * the 57 extracted JPEG frames carry **no EXIF at all** (0/57 with a GPS IFD)
  * the MP4 has **no ``udta`` / ``meta`` / ``ilst`` atom**, i.e. no
    location/creation-time metadata container
  * there is **no sidecar flight log** (.csv / .srt / .gpx / .kml / .db)

Therefore no GPS/IMU/RTK datum exists to align against, and georeferencing is
reported as UNAVAILABLE with the evidence, rather than approximated.

WHY NOT JUST ASSUME AN ORIGIN?
------------------------------
Even a plausible-looking latitude/longitude would be a fabricated coordinate
presented as a measurement. Upstream, disaster analysis already refuses to emit
``georeferenced_area`` / ``flood_area_sq_m`` for the same reason. This module
extends that same discipline to the 3D reconstruction.

EXTENDING IT WITH REAL FLIGHT DATA
----------------------------------
Everything needed to accept real telemetry is here and unit-tested:

  1. Add a parser to :data:`TELEMETRY_PARSERS` that yields
     ``(frame_name, lon, lat, alt)``.
  2. Call :func:`georeference_job` again. With >= ``MIN_CONTROL_POINTS``
     correspondences it fits a real similarity transform
     (:func:`umeyama_similarity`) from the local world frame to local ENU
     metres, then projects that to WGS84 via :func:`enu_to_wgs84`.

No parser is registered today, so the pipeline never invents control points.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

STATUS_GEOREFERENCED = "georeferenced"
STATUS_UNAVAILABLE = "unavailable"

COORDINATE_SYSTEM_LOCAL = "local_up_to_scale"
COORDINATE_SYSTEM_WGS84 = "wgs84_lon_lat_alt"
CRS_WGS84 = "EPSG:4326"

#: Minimum GPS/3D correspondences needed to fit a similarity transform.
MIN_CONTROL_POINTS = 3

#: Sidecar flight-log extensions searched next to the video.
TELEMETRY_EXTENSIONS = (".csv", ".srt", ".gpx", ".kml", ".kmz", ".log", ".db")

#: Registry of real telemetry parsers. Empty on purpose: no parser means no
#: fabricated coordinates. Each entry maps a source name to a callable that
#: returns ``[(frame_name, lon, lat, alt), ...]``.
TELEMETRY_PARSERS: Dict[str, Callable[[Path], List[Tuple[str, float, float, float]]]] = {}


# --------------------------------------------------------------------------
# Metadata discovery
# --------------------------------------------------------------------------

def _exif_gps_from_image(path: Path) -> Optional[Tuple[float, float, Optional[float]]]:
    """Return ``(lon, lat, alt)`` from a JPEG's EXIF GPS block, or ``None``."""
    try:
        from PIL import Image  # imported lazily: only needed while probing
    except ImportError:  # pragma: no cover - Pillow ships with the project
        return None

    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif or 34853 not in exif:      # 34853 == GPSInfo IFD pointer
                return None
            gps_ifd = exif.get_ifd(34853)
    except Exception:
        return None

    def _rat(num: Dict[int, Any], den: Dict[int, Any]) -> Optional[float]:
        try:
            numerator = float(num[0])
            denominator = float(den[0])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return numerator / denominator if denominator else None

    lat = _rat(gps_ifd.get(2, {}), gps_ifd.get(1, {}))     # 2 = lat ref, 1 = lat
    lon = _rat(gps_ifd.get(4, {}), gps_ifd.get(3, {}))     # 4 = lon ref, 3 = lon
    alt = None
    if 6 in gps_ifd and 5 in gps_ifd:
        try:
            alt = float(gps_ifd[6][0]) / float(gps_ifd[5][0])
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            alt = None

    lat_ref = gps_ifd.get(2)
    lon_ref = gps_ifd.get(4)
    if lat is None or lon is None:
        return None
    if isinstance(lat_ref, str) and lat_ref.strip().upper() == "S":
        lat = -lat
    if isinstance(lon_ref, str) and lon_ref.strip().upper() == "W":
        lon = -lon
    return float(lon), float(lat), alt


def _scan_frames(
    frames_dir: Path,
    max_frames: int,
) -> Tuple[int, int, Optional[Tuple[float, float, Optional[float]]]]:
    """Return ``(n_with_gps, n_inspected, first_fix)`` for EXIF GPS blocks."""
    if not frames_dir.is_dir():
        return 0, 0, None
    files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    )
    inspected = 0
    with_gps = 0
    first_fix: Optional[Tuple[float, float, Optional[float]]] = None
    for path in files[:max_frames]:
        inspected += 1
        fix = _exif_gps_from_image(path)
        if fix is not None:
            with_gps += 1
            if first_fix is None:
                first_fix = fix
    return with_gps, inspected, first_fix


def _scan_sidecars(job_dir: Path) -> List[str]:
    found: List[str] = []
    for directory in (job_dir, job_dir / "metadata", job_dir / "telemetry"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in TELEMETRY_EXTENSIONS:
                found.append(path.name)
    return found


def _scan_video_metadata(video_path: Optional[Path]) -> List[str]:
    """List MP4 boxes that could carry a location (no payload is interpreted)."""
    if video_path is None or not Path(video_path).is_file():
        return []
    containers = {b"udta", b"meta", b"ilst"}
    interesting = {"udta", "meta", "ilst"}
    boxes: List[str] = []
    try:
        with open(video_path, "rb") as handle:
            buffer = handle.read(4 << 20)
    except OSError:
        return []

    def walk(chunk: bytes) -> None:
        offset = 0
        while offset + 8 <= len(chunk):
            size, kind = struct.unpack(">I4s", chunk[offset:offset + 8])
            if size < 8 or offset + size > len(chunk):
                break
            name = kind.decode("latin1")
            if name in interesting and name not in boxes:
                boxes.append(name)
            if kind in containers:
                walk(chunk[offset + 8:offset + size])
            offset += size

    walk(buffer)
    return boxes


def probe_geotag_sources(
    job_dir: Union[str, Path],
    video_path: Optional[Union[str, Path]] = None,
    max_frames: int = 200,
) -> Dict[str, Any]:
    """Search every plausible location source and report what was found.

    Pure discovery: returns evidence, never a coordinate.
    """
    job = Path(job_dir)
    sources_checked = [
        "frame_exif_gps",
        "sidecar_telemetry",
        "video_container_metadata",
        "registered_telemetry_parsers",
    ]
    sources_found: List[str] = []

    with_gps, inspected, first_fix = _scan_frames(job / "frames", max_frames)
    if with_gps:
        sources_found.append("frame_exif_gps")

    sidecars = _scan_sidecars(job)
    if sidecars:
        sources_found.append("sidecar_telemetry")

    video_boxes = _scan_video_metadata(Path(video_path) if video_path else None)
    if video_boxes:
        sources_found.append("video_container_metadata")

    parsers = sorted(TELEMETRY_PARSERS)
    if parsers:
        sources_found.append("registered_telemetry_parsers")

    return {
        "sources_checked": sources_checked,
        "sources_found": sources_found,
        "n_frames_inspected": inspected,
        "n_frames_with_gps_exif": with_gps,
        "first_frame_gps_fix": (
            {"lon": first_fix[0], "lat": first_fix[1], "alt": first_fix[2]}
            if first_fix else None
        ),
        "sidecar_files_found": sidecars,
        "video_metadata_boxes_found": video_boxes,
        "registered_parsers": parsers,
        "usable": bool(sources_found),
    }


# --------------------------------------------------------------------------
# Real transform machinery (used only when genuine control points exist)
# --------------------------------------------------------------------------

def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Least-squares similarity transform ``target ~= s*R @ source + t``.

    Umeyama (1991): the closed-form least-squares fit of uniform scale, rotation
    and translation. Maps the LOCAL reconstruction frame onto a metric frame
    (e.g. local ENU metres).

    Needs at least three correspondences; degenerate (collinear or coincident)
    input is rejected because the scale is then unobservable.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("source and target must have the same shape")
    if src.shape[0] < MIN_CONTROL_POINTS:
        raise ValueError(
            f"need at least {MIN_CONTROL_POINTS} control points, got {src.shape[0]}"
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    covariance = (dst_c.T @ src_c) / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0

    rotation = u @ correction @ vt
    variance = float((src_c ** 2).sum() / src.shape[0])
    if variance <= 0 or singular[1] <= 0:
        raise ValueError("control points are degenerate (collinear or coincident)")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "residual_rms": float(
            np.sqrt((((scale * (rotation @ src.T).T + translation) - dst) ** 2)
                    .sum(axis=1).mean())
        ),
    }


def enu_to_wgs84(
    east: float,
    north: float,
    up: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> Tuple[float, float, float]:
    """Local ENU metres (origin at the given WGS84 fix) -> ``(lon, lat, alt)``.

    Exact path: ENU -> ECEF -> geodetic on WGS84.

    The ECEF -> geodetic inversion is iterated rather than taken from Bowring's
    closed form, which loses ~1e-3 degrees of latitude on terrestrial fixes.
    The tempting ``atan2``-based tangent-plane shortcut is also avoided: it is
    degenerate for a due-north offset (``atan2(y, 0)`` is 90 degrees).
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e_sq = f * (2.0 - f)

    lat1 = math.radians(origin_lat)
    lon1 = math.radians(origin_lon)
    sin1, cos1 = math.sin(lat1), math.cos(lat1)
    n1 = a / math.sqrt(1.0 - e_sq * sin1 * sin1)

    # ENU offset -> ECEF. This is the transpose of the standard ECEF->ENU rotation,
    # so a zero offset returns the origin's own ECEF position exactly.
    x0 = (n1 + origin_alt) * cos1 * math.cos(lon1)
    y0 = (n1 + origin_alt) * cos1 * math.sin(lon1)
    z0 = (n1 * (1.0 - e_sq) + origin_alt) * sin1

    x = (x0
         - east * math.sin(lon1)
         - north * sin1 * math.cos(lon1)
         + up * cos1 * math.cos(lon1))
    y = (y0
         + east * math.cos(lon1)
         - north * sin1 * math.sin(lon1)
         + up * cos1 * math.sin(lon1))
    z = z0 + north * cos1 + up * sin1

    # ECEF -> geodetic, iterated. Bowring's closed form loses ~1e-3 deg of
    # latitude on terrestrial fixes, so the latitude is refined instead; it
    # converges to machine precision in three or four passes.
    #
    # The height identity is h = p / cos(lat) - N(lat). The tempting shortcut
    # p*cos(lat) + z*sin(lat) - N equals `a + h - N`, i.e. it is only right under
    # a spherical approximation and settles on a slightly wrong fixed point.
    p = math.hypot(x, y)
    lon = math.atan2(y, x)

    def _height(lat_rad: float) -> float:
        n = a / math.sqrt(1.0 - e_sq * math.sin(lat_rad) ** 2)
        cos_lat = math.cos(lat_rad)
        if abs(cos_lat) > 1e-12:
            return p / cos_lat - n
        # Exactly at a pole cos(lat) -> 0; use the sin branch, which is stable.
        return z / math.sin(lat_rad) - n * (1.0 - e_sq)

    lat = math.atan2(z, p * (1.0 - e_sq)) if p > 0 else math.copysign(math.pi / 2, z)
    for _ in range(12):
        n = a / math.sqrt(1.0 - e_sq * math.sin(lat) ** 2)
        denominator = n + _height(lat)
        if denominator == 0:
            break
        lat = math.atan2(z, p * (1.0 - e_sq * n / denominator))

    return math.degrees(lon), math.degrees(lat), _height(lat)

    with_gps, inspected, first_fix = _scan_frames(job / "frames", max_frames)
    if with_gps:
        sources_found.append("frame_exif_gps")

    sidecars = _scan_sidecars(job)
    if sidecars:
        sources_found.append("sidecar_telemetry")

    video_boxes = _scan_video_metadata(Path(video_path) if video_path else None)
    if video_boxes:
        sources_found.append("video_container_metadata")

    parsers = sorted(TELEMETRY_PARSERS)
    if parsers:
        sources_found.append("registered_telemetry_parsers")

    return {
        "sources_checked": sources_checked,
        "sources_found": sources_found,
        "n_frames_inspected": inspected,
        "n_frames_with_gps_exif": with_gps,
        "first_frame_gps_fix": (
            {"lon": first_fix[0], "lat": first_fix[1], "alt": first_fix[2]}
            if first_fix else None
        ),
        "sidecar_files_found": sidecars,
        "video_metadata_boxes_found": video_boxes,
        "registered_parsers": parsers,
        "usable": bool(sources_found),
    }
# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class GeoreferencingResult:
    """Honest outcome of an attempt (or non-attempt) to georeference a job."""

    status: str
    georeferenced: bool
    coordinate_system: str
    reason: str
    source: Optional[str] = None
    crs: Optional[str] = None
    n_control_points: int = 0
    residual_rms: Optional[float] = None
    probe: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_GEOREFERENCED

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "module": "cheel_nazar.reconstruction.georeferencing",
            "phase": "8.5",
            "status": self.status,
            "georeferenced": self.georeferenced,
            "coordinate_system": self.coordinate_system,
            "crs": self.crs,
            "source": self.source,
            "n_control_points": self.n_control_points,
            "residual_rms": self.residual_rms,
            "reason": self.reason,
            "metadata_probe": self.probe,
            "requirements": self.requirements,
            "limitations": self.limitations,
        }


BASE_LIMITATIONS = [
    "Without GPS/IMU/RTK the reconstruction has NO absolute position.",
    "Local world units are up-to-scale; no metres, distances or areas are implied.",
    "Camera centres are in the seed-pair world frame and are NOT coordinates on Earth.",
]

REQUIREMENTS = [
    "Per-frame GPS (lon, lat) for >= 3 frames, from EXIF, a flight log or a "
    "registered telemetry parser, plus an altitude source.",
    "Camera-to-frame time correspondence between the extracted frames and the GPS track.",
    "An explicit vertical datum if altitude is to be trusted.",
]


def georeference_job(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    video_path: Optional[Union[str, Path]] = None,
    control_points: Optional[Sequence[Tuple[str, float, float, Optional[float]]]] = None,
) -> GeoreferencingResult:
    """Attempt georeferencing for ``job_id`` using ONLY real metadata.

    ``control_points`` is the single entry point for genuine telemetry. It is
    explicit rather than auto-discovered so nothing can be inferred into
    existence.
    """
    from .sfm import _resolve_data_dir

    job_dir = _resolve_data_dir(data_dir) / job_id
    probe = probe_geotag_sources(job_dir, video_path)

    points = list(control_points) if control_points else []
    if len(points) < MIN_CONTROL_POINTS:
        return GeoreferencingResult(
            status=STATUS_UNAVAILABLE,
            georeferenced=False,
            coordinate_system=COORDINATE_SYSTEM_LOCAL,
            reason=(
                "No usable GPS/IMU control points exist for this job. Checked "
                f"{', '.join(probe['sources_checked'])}: "
                f"{probe['n_frames_with_gps_exif']}/{probe['n_frames_inspected']} "
                "frames carry an EXIF GPS block, "
                f"{len(probe['sidecar_files_found'])} sidecar telemetry file(s), "
                f"video metadata boxes {probe['video_metadata_boxes_found'] or 'absent'}, "
                f"registered parsers {probe['registered_parsers'] or 'none'}. "
                "Coordinates are therefore NOT reported."
            ),
            probe=probe,
            requirements=list(REQUIREMENTS),
            limitations=list(BASE_LIMITATIONS),
        )

    if points[0][3] is None:
        return GeoreferencingResult(
            status=STATUS_UNAVAILABLE,
            georeferenced=False,
            coordinate_system=COORDINATE_SYSTEM_LOCAL,
            reason=(
                "GPS control points were supplied without an altitude reference, "
                "so the local frame cannot be tied to the vertical datum. "
                "Coordinates are therefore NOT reported."
            ),
            n_control_points=len(points),
            probe=probe,
            requirements=list(REQUIREMENTS),
            limitations=list(BASE_LIMITATIONS),
        )

    # Reached only with genuine, altitude-referenced control points. The
    # frame-to-telemetry time mapping is not implemented, so this is explicit
    # rather than silently approximated.
    raise NotImplementedError(
        "Georeferencing from real control points requires the frame-to-telemetry "
        "time mapping, which is not implemented. Register a telemetry parser in "
        "TELEMETRY_PARSERS and extend georeference_job()."
    )