"""
pygeovision.insar.slc.sentinel1
================================
Sentinel-1 SLC product metadata parsing and validation.

Handles both the legacy S1A naming convention and the new S1C/S1D products.
All Sentinel-1 SLC products share the same internal structure regardless of
which satellite acquired them (1A, 1B, 1C, 1D).

Product filename convention:
    S1C_IW_SLC__1SDV_20260601T053000_20260601T053027_NNNNNN_XXXXXX_YYYY.SAFE.zip
    ↑   ↑↑  ↑↑↑  ↑↑↑↑  ↑                             ↑       ↑      ↑
    sat mod prod level  start_time                    orbit   dtake  crc

Subswath IW1/IW2/IW3 cover:
    IW1: far  range ~19-26° incidence
    IW2: mid  range ~26-35° incidence  (most common for land applications)
    IW3: near range ~35-44° incidence
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger("pygeovision.insar.slc.sentinel1")

# Sentinel-1 C-band centre frequency and wavelength
S1_CENTRE_FREQ_HZ      = 5.405e9     # 5.405 GHz
S1_WAVELENGTH_M        = 0.05546576  # metres  (c / f)
SENTINEL1_WAVELENGTH_M = S1_WAVELENGTH_M   # public alias used across modules
S1_RANGE_SAMPLING_RATE = 64.345238e6 # samples/s (IW mode)
S1_PRF_IW              = 486.486           # Hz (approximate IW mode PRF)

# IW subswath incidence angle ranges (mid-swath, degrees)
SUBSWATH_INCIDENCE     = {"IW1": 32.9, "IW2": 38.3, "IW3": 43.1}
SUBSWATH_INCIDENCE_DEG = SUBSWATH_INCIDENCE    # public alias

# Valid satellite IDs (including new C/D)
VALID_SATELLITES = {"S1A", "S1B", "S1C", "S1D"}


@dataclass
class S1SLCProduct:
    """
    Parsed Sentinel-1 SLC product descriptor.

    Parameters
    ----------
    path : str
        Path to the .zip or .SAFE directory.
    satellite : str
        Satellite identifier: S1A, S1B, S1C, or S1D.
    start_time : datetime
        Acquisition start time (UTC).
    stop_time : datetime
        Acquisition stop time (UTC).
    absolute_orbit : int
        Absolute orbit number.
    polarisations : List[str]
        Available polarisations, e.g. ['VV', 'VH'].
    subswaths : List[str]
        Available subswaths, e.g. ['IW1', 'IW2', 'IW3'].
    pass_direction : str
        'ASCENDING' or 'DESCENDING'.
    wavelength_m : float
        SAR wavelength in metres.
    """
    path:           str
    satellite:      str
    start_time:     datetime
    stop_time:      datetime
    absolute_orbit: int
    polarisations:  List[str]           = field(default_factory=list)
    subswaths:      List[str]           = field(default_factory=list)
    pass_direction: str                 = "UNKNOWN"
    wavelength_m:   float               = S1_WAVELENGTH_M
    metadata:       Dict[str, str]      = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def date_str(self) -> str:
        return self.start_time.strftime("%Y%m%d")

    @property
    def temporal_baseline_days(self) -> float:
        """Days since start_time (used for pair temporal baseline)."""
        return 0.0

    def incidence_angle(self, subswath: str = "IW2") -> float:
        """Return mid-swath incidence angle for the given subswath."""
        return SUBSWATH_INCIDENCE.get(subswath.upper(), 38.3)

    def is_compatible_with(self, other: "S1SLCProduct") -> bool:
        """
        Check whether this product can form an InSAR pair with `other`.

        Compatibility requires:
        - Same subswath coverage
        - Same polarisations
        - Same pass direction
        - Same relative orbit (same ground track)
        """
        if self.pass_direction != other.pass_direction:
            return False
        if set(self.polarisations) != set(other.polarisations):
            return False
        if self.relative_orbit != other.relative_orbit:
            return False
        return True

    @property
    def relative_orbit(self) -> int:
        """Relative orbit number (1–175 for Sentinel-1)."""
        return ((self.absolute_orbit - 73) % 175) + 1

    def __repr__(self) -> str:
        return (
            f"S1SLCProduct({self.satellite} {self.date_str} "
            f"orbit={self.absolute_orbit} "
            f"pol={'/'.join(self.polarisations)} "
            f"swaths={'/'.join(self.subswaths)} "
            f"{self.pass_direction[:3]})"
        )


# ── Filename parser ──────────────────────────────────────────────────────────

_S1_FILENAME_RE = re.compile(
    r"(?P<sat>S1[ABCD])_"
    r"(?P<mode>IW)_"
    r"(?P<prod>SLC)__"
    r"(?P<level>1S)(?P<pol>DV|DH|SV|SH)_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_"
    r"(?P<orbit>\d{6})_"
    r"(?P<dtake>[0-9A-F]{6})_"
    r"(?P<crc>[0-9A-F]{4})"
)

_POL_MAP = {
    "DV": ["VV", "VH"],
    "DH": ["HH", "HV"],
    "SV": ["VV"],
    "SH": ["HH"],
}


def parse_s1_slc_filename(path: str) -> Optional[S1SLCProduct]:
    """
    Parse a Sentinel-1 SLC product from its filename alone.

    Works for .zip, .SAFE directories, and bare filenames.
    Returns None if the filename does not match the Sentinel-1 SLC convention.
    """
    name = Path(path).stem.replace(".SAFE", "")
    m = _S1_FILENAME_RE.match(name)
    if not m:
        logger.warning("Cannot parse S1 SLC filename: %s", name)
        return None

    g = m.groupdict()
    fmt = "%Y%m%dT%H%M%S"
    return S1SLCProduct(
        path           = str(path),
        satellite      = g["sat"],
        start_time     = datetime.strptime(g["start"], fmt),
        stop_time      = datetime.strptime(g["stop"],  fmt),
        absolute_orbit = int(g["orbit"]),
        polarisations  = _POL_MAP.get(g["pol"], [g["pol"]]),
        subswaths      = ["IW1", "IW2", "IW3"],   # all IW subswaths present
        pass_direction = "UNKNOWN",                # needs manifest for this
    )


def parse_s1_slc_manifest(path: str) -> Optional[S1SLCProduct]:
    """
    Parse a Sentinel-1 SLC product by reading its manifest.safe XML.

    More accurate than filename-only parsing — gives pass direction,
    exact orbit, and actual subswath availability.

    Parameters
    ----------
    path : str
        Path to .zip or .SAFE directory.
    """
    path = Path(path)

    # Get manifest content
    manifest_text = _read_manifest(path)
    if not manifest_text:
        # Fall back to filename parsing
        return parse_s1_slc_filename(str(path))

    root = ET.fromstring(manifest_text)
    ns = {
        "safe": "http://www.esa.int/safe/sentinel-1.0",
        "s1":   "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
        "s1sar":"http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
    }

    # ── satellite / family ────────────────────────────────────────────────
    family_el = root.find(".//safe:familyName", ns)
    number_el  = root.find(".//safe:number", ns)
    satellite  = "S1?"
    if family_el is not None and number_el is not None:
        sat_num = number_el.text.strip()               # e.g. "C"
        satellite = f"S1{sat_num}"

    # ── times ─────────────────────────────────────────────────────────────
    start_el = root.find(".//safe:startTime", ns)
    stop_el  = root.find(".//safe:stopTime",  ns)
    fmt      = "%Y-%m-%dT%H:%M:%S.%f"
    try:
        start_time = datetime.strptime(start_el.text[:26], fmt) if start_el is not None else datetime.utcnow()
        stop_time  = datetime.strptime(stop_el.text[:26],  fmt) if stop_el  is not None else datetime.utcnow()
    except (ValueError, AttributeError):
        start_time = stop_time = datetime.utcnow()

    # ── orbit ─────────────────────────────────────────────────────────────
    orbit_el = root.find(".//safe:orbitNumber[@type='start']", ns)
    abs_orbit = int(orbit_el.text) if orbit_el is not None else 0

    # ── pass direction ────────────────────────────────────────────────────
    pass_el = root.find(".//s1:pass", ns)
    pass_dir = pass_el.text.upper() if pass_el is not None else "UNKNOWN"

    # ── polarisations ─────────────────────────────────────────────────────
    pol_els = root.findall(".//s1sar:transmitterReceiverPolarisation", ns)
    pols = sorted({el.text.strip() for el in pol_els if el.text})
    if not pols:
        pols = ["VV", "VH"]   # default dual-pol assumption

    # ── subswaths ─────────────────────────────────────────────────────────
    swath_els = root.findall(".//safe:swath", ns)
    swaths    = sorted({el.text.strip() for el in swath_els if el.text})
    if not swaths:
        swaths = ["IW1", "IW2", "IW3"]

    # Filename for any remaining fields
    prod_from_file = parse_s1_slc_filename(str(path))

    return S1SLCProduct(
        path           = str(path),
        satellite      = satellite,
        start_time     = start_time,
        stop_time      = stop_time,
        absolute_orbit = abs_orbit or (prod_from_file.absolute_orbit if prod_from_file else 0),
        polarisations  = pols,
        subswaths      = swaths,
        pass_direction = pass_dir,
    )


def _read_manifest(path: Path) -> Optional[str]:
    """Read manifest.safe from a .zip or .SAFE directory."""
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(str(path)) as z:
                # manifest.safe is at the root of the SAFE structure inside the zip
                candidates = [n for n in z.namelist() if n.endswith("manifest.safe")]
                if candidates:
                    return z.read(candidates[0]).decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("Cannot read manifest from zip %s: %s", path, e)
        return None

    # .SAFE directory
    manifest = path / "manifest.safe"
    if not manifest.exists():
        manifest = path.with_suffix("") / "manifest.safe"
    if manifest.exists():
        return manifest.read_text(encoding="utf-8", errors="replace")
    return None


# ── Pair validation ──────────────────────────────────────────────────────────

def validate_slc_pair(
    master: S1SLCProduct,
    slave:  S1SLCProduct,
) -> Dict[str, object]:
    """
    Validate a master/slave SLC pair for InSAR processing.

    Returns a dict with:
        valid (bool)            : True if the pair can form an interferogram
        temporal_baseline_days  : time separation between acquisitions
        spatial_baseline_m      : estimated perpendicular baseline (metres)
        warnings                : list of warning messages
        errors                  : list of blocking error messages
    """
    warnings_: List[str] = []
    errors_:   List[str] = []

    # Temporal baseline
    dt = abs((slave.start_time - master.start_time).total_seconds()) / 86400
    temporal_baseline = dt

    # Checks
    if master.pass_direction != slave.pass_direction:
        errors_.append(
            f"Pass direction mismatch: master={master.pass_direction}, "
            f"slave={slave.pass_direction}. Ascending/descending pairs "
            f"cannot form an interferogram."
        )

    if master.relative_orbit != slave.relative_orbit:
        errors_.append(
            f"Relative orbit mismatch: master={master.relative_orbit}, "
            f"slave={slave.relative_orbit}. Must be same ground track."
        )

    if not set(master.polarisations) & set(slave.polarisations):
        errors_.append(
            f"No common polarisation: master={master.polarisations}, "
            f"slave={slave.polarisations}."
        )

    if temporal_baseline < 0.1:
        errors_.append("Acquisitions appear to be the same date — no displacement to measure.")

    if temporal_baseline > 365:
        warnings_.append(
            f"Temporal baseline of {temporal_baseline:.0f} days is large. "
            f"Temporal decorrelation may reduce coherence significantly. "
            f"Consider using pairs ≤ 48 days for vegetated areas."
        )
    elif temporal_baseline > 48:
        warnings_.append(
            f"Temporal baseline of {temporal_baseline:.0f} days. "
            f"Coherence may be reduced in vegetated or agricultural areas. "
            f"Optimal baseline for Sentinel-1 IW: 6–24 days."
        )

    if master.satellite != slave.satellite:
        warnings_.append(
            f"Cross-satellite pair: {master.satellite}/{slave.satellite}. "
            f"Orbital baseline may differ — verify coherence before interpreting."
        )

    # Estimated perpendicular baseline (rough approximation from orbit numbers)
    # True baseline requires precise orbit files — this is indicative only
    orbit_diff = abs(master.absolute_orbit - slave.absolute_orbit)
    # Sentinel-1 orbit repeat: 175 orbits = 12 days
    # Typical perpendicular baseline: ~150 m/repeat, highly variable
    spatial_baseline_est = (orbit_diff % 175) * 0.8   # very rough, metres

    return {
        "valid":                  len(errors_) == 0,
        "temporal_baseline_days": round(temporal_baseline, 1),
        "spatial_baseline_m_est": round(spatial_baseline_est, 1),
        "common_polarisations":   sorted(set(master.polarisations) & set(slave.polarisations)),
        "warnings":               warnings_,
        "errors":                 errors_,
    }
