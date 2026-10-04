"""Phase 8.5 - georeferencing tests.

The point of these tests is that georeferencing is HONEST: with no real GPS data
it must refuse to produce a coordinate, and when genuine control points exist the
transform machinery must actually be correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reconstruction.georeferencing import (
    BASE_LIMITATIONS,
    COORDINATE_SYSTEM_LOCAL,
    MIN_CONTROL_POINTS,
    STATUS_UNAVAILABLE,
    TELEMETRY_PARSERS,
    enu_to_wgs84,
    georeference_job,
    probe_geotag_sources,
    umeyama_similarity,
)

JOB = "job-georef"


def _make_job(tmp_path, *, frames: int = 3):
    """A job directory with plain frames carrying no EXIF at all."""
    import cv2

    job = tmp_path / "data" / JOB
    (job / "frames").mkdir(parents=True)
    for i in range(frames):
        cv2.imwrite(
            str(job / "frames" / f"frame_{i:05d}.jpg"),
            np.full((16, 16), 120, np.uint8),
        )
    return job


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_probe_reports_nothing_found_when_no_metadata_exists(tmp_path):
    job = _make_job(tmp_path)
    probe = probe_geotag_sources(job)

    assert probe["sources_found"] == []
    assert probe["usable"] is False
    assert probe["n_frames_inspected"] == 3
    assert probe["n_frames_with_gps_exif"] == 0
    assert probe["first_frame_gps_fix"] is None
    assert probe["sidecar_files_found"] == []
    assert probe["video_metadata_boxes_found"] == []


def test_probe_finds_a_sidecar_telemetry_file(tmp_path):
    job = _make_job(tmp_path)
    (job / "flight.csv").write_text("lon,lat,alt\n0,0,0\n", encoding="utf-8")
    probe = probe_geotag_sources(job)

    assert "sidecar_telemetry" in probe["sources_found"]
    assert probe["usable"] is True
    assert probe["sidecar_files_found"] == ["flight.csv"]


def test_probe_counts_a_registered_parser(tmp_path):
    job = _make_job(tmp_path)
    TELEMETRY_PARSERS["stub"] = lambda path: []
    try:
        probe = probe_geotag_sources(job)
        assert "registered_telemetry_parsers" in probe["sources_found"]
        assert probe["registered_parsers"] == ["stub"]
    finally:
        TELEMETRY_PARSERS.pop("stub", None)


def test_probe_handles_a_missing_job_directory(tmp_path):
    probe = probe_geotag_sources(tmp_path / "does_not_exist")
    assert probe["n_frames_inspected"] == 0
    assert probe["usable"] is False


# ---------------------------------------------------------------------------
# Honest failure
# ---------------------------------------------------------------------------

def test_georeferencing_is_unavailable_without_gps(tmp_path):
    _make_job(tmp_path)
    result = georeference_job(JOB, tmp_path / "data")

    assert result.status == STATUS_UNAVAILABLE
    assert result.ok is False
    assert result.georeferenced is False
    assert result.crs is None
    assert result.coordinate_system == COORDINATE_SYSTEM_LOCAL
    assert "NOT reported" in result.reason
    assert result.limitations == BASE_LIMITATIONS


def test_georeferencing_manifest_contains_no_coordinates(tmp_path):
    _make_job(tmp_path)
    manifest = georeference_job(JOB, tmp_path / "data").to_manifest()

    assert manifest["georeferenced"] is False
    assert manifest["crs"] is None
    assert manifest["metadata_probe"]["first_frame_gps_fix"] is None
    for forbidden in ("lat", "lon", "latitude", "longitude", "alt", "fix"):
        assert forbidden not in manifest, f"manifest must not expose {forbidden!r}"
    assert "0/3 frames" in manifest["reason"]


def test_georeferencing_requires_an_altitude_reference(tmp_path):
    _make_job(tmp_path)
    points = [("a.jpg", 73.0, 18.0, None) for _ in range(MIN_CONTROL_POINTS)]
    result = georeference_job(JOB, tmp_path / "data", control_points=points)

    assert result.status == STATUS_UNAVAILABLE
    assert result.georeferenced is False
    assert result.n_control_points == MIN_CONTROL_POINTS


def test_too_few_control_points_is_reported_not_guessed(tmp_path):
    _make_job(tmp_path)
    points = [("a.jpg", 73.0, 18.0, 5.0)]
    result = georeference_job(JOB, tmp_path / "data", control_points=points)

    assert result.status == STATUS_UNAVAILABLE
    assert result.georeferenced is False


def test_full_control_points_do_not_silently_succeed(tmp_path):
    """Real points exist, but the frame/time mapping is unimplemented.

    It must raise loudly rather than emit a fabricated coordinate.
    """
    _make_job(tmp_path)
    points = [(f"f{i}.jpg", 73.0 + i * 1e-4, 18.0, 50.0 + i) for i in range(4)]
    with pytest.raises(NotImplementedError):
        georeference_job(JOB, tmp_path / "data", control_points=points)


# ---------------------------------------------------------------------------
# Real transform machinery
# ---------------------------------------------------------------------------

def _rodrigues(rvec):
    import cv2

    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))


def test_umeyama_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(8, 3))
    rotation, _ = _rodrigues(rng.normal(scale=0.4, size=3))
    scale = 2.5
    translation = np.array([1.0, -2.0, 3.0])
    target = (scale * (rotation @ source.T)).T + translation

    fit = umeyama_similarity(source, target)

    assert fit["scale"] == pytest.approx(scale, rel=1e-6)
    assert np.allclose(fit["rotation"], rotation, atol=1e-6)
    assert np.allclose(fit["translation"], translation, atol=1e-6)
    assert fit["residual_rms"] == pytest.approx(0.0, abs=1e-6)


def test_umeyama_rejects_insufficient_control_points():
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


def test_umeyama_rejects_collinear_points():
    line = np.column_stack([np.arange(5.0), np.zeros(5), np.zeros(5)])
    with pytest.raises(ValueError):
        umeyama_similarity(line, line * 2.0)


def test_umeyama_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((4, 3)), np.zeros((3, 3)))


def test_enu_to_wgs84_round_trips_zero_offset():
    """Zero offset must return the origin exactly (this caught a real bug)."""
    lon, lat, alt = enu_to_wgs84(0.0, 0.0, 0.0, 18.0, 73.0, 12.0)
    assert lat == pytest.approx(18.0, abs=1e-9)
    assert lon == pytest.approx(73.0, abs=1e-9)
    assert alt == pytest.approx(12.0, abs=1e-6)


def _wgs84_curvatures(lat_deg: float):
    """``(meridional M, prime-vertical N)`` radii of curvature at a latitude."""
    a, f = 6378137.0, 1.0 / 298.257223563
    e_sq = f * (2.0 - f)
    s = math.sin(math.radians(lat_deg)) ** 2
    w = math.sqrt(1.0 - e_sq * s)
    return a * (1.0 - e_sq) / w ** 3, a / w


def test_enu_to_wgs84_matches_known_offsets():
    """A 1 km ENU offset must move lat/lon by the local radii of curvature.

    The function returns ``(lon, lat, alt)``. North uses the meridional radius
    M, east uses the prime-vertical radius N scaled by cos(lat); neither is the
    mean Earth radius.
    """
    lat, lon, alt = 18.0, 73.0, 12.0
    m_radius, n_radius = _wgs84_curvatures(lat)

    lon_a, lat_b, alt_b = enu_to_wgs84(0.0, 1000.0, 0.0, lat, lon, alt)
    assert lat_b > lat and lon_a == pytest.approx(lon, abs=1e-9)
    # A horizontal move on an ellipsoid changes height slightly (the surface
    # curves away), so altitude must stay close, not identical.
    assert alt_b == pytest.approx(alt, abs=0.5)
    north_metres = math.radians(lat_b - lat) * m_radius
    assert north_metres == pytest.approx(1000.0, rel=1e-3)

    lon_c, lat_c, _ = enu_to_wgs84(1000.0, 0.0, 0.0, lat, lon, alt)
    east_metres = math.radians(lon_c - lon) * n_radius * math.cos(math.radians(lat))
    assert east_metres == pytest.approx(1000.0, rel=1e-3)
    # A due-east offset must not move latitude.
    assert lat_c == pytest.approx(lat, abs=1e-5)


def test_enu_to_wgs84_handles_a_due_north_offset():
    """A pure north offset must not collapse latitude to 90 degrees."""
    m_radius, _ = _wgs84_curvatures(18.0)
    _, lat, _ = enu_to_wgs84(0.0, 5000.0, 0.0, 18.0, 73.0, 0.0)
    assert math.radians(lat - 18.0) * m_radius == pytest.approx(5000.0, rel=1e-3)


def test_enu_to_wgs84_up_offset_changes_altitude():
    _, _, alt = enu_to_wgs84(0.0, 0.0, 250.0, 18.0, 73.0, 10.0)
    assert alt == pytest.approx(260.0, abs=1e-3)


def test_enu_to_wgs84_combined_offset_is_consistent():
    """A 3D offset must move lon, lat and alt in the expected directions."""
    lon, lat, alt = enu_to_wgs84(300.0, -400.0, 50.0, 18.0, 73.0, 10.0)
    assert lon > 73.0          # east
    assert lat < 18.0          # south
    assert alt > 60.0          # up, plus curvature


def test_enu_to_wgs84_at_the_pole_stays_finite():
    lon, lat, _ = enu_to_wgs84(10.0, 10.0, 1.0, 90.0, 0.0)
    assert math.isfinite(lon) and math.isfinite(lat)