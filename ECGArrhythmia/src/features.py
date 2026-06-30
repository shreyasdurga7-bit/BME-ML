"""Beat-level feature extraction for the MIT-BIH Arrhythmia Database.

For every annotated beat in every record under ``data/``, extracts a feature
vector spanning time-domain, morphological, frequency-domain, and wavelet
descriptors of the signal window around the beat's R-peak, maps the beat's
raw annotation symbol to one of the 5 AAMI EC57 classes (N, S, V, F, Q), and
writes everything to a CSV ready for ML classification.

Usage:
    python src/features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "features.csv"

# AAMI EC57 mapping from MIT-BIH annotation symbols to the 5 AAMI beat
# classes. Symbols not listed here (e.g. "+", "~", "!", '"', "x", "|", "[",
# "]") are rhythm-change / signal-quality annotations rather than heartbeats
# and are dropped before feature extraction.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Window around each R-peak used for all feature families, in seconds.
PRE_WINDOW_S = 0.25
POST_WINDOW_S = 0.40

# QRS duration is estimated from the signal's slope magnitude around the
# R-peak. SEARCH_S bounds how far the QRS onset/offset can be from R;
# THRESHOLD_FRAC is the fraction of the window's peak slope below which the
# signal is considered to have left the complex; SMOOTH_N smooths the slope
# to avoid a spurious "boundary" right at R (where slope crosses zero at the
# peak itself); MIN_HALF_S guards against stopping before the loop has moved
# past that zero-crossing.
QRS_SEARCH_S = 0.12
QRS_THRESHOLD_FRAC = 0.08
QRS_SMOOTH_N = 3
QRS_MIN_HALF_S = 0.012

# Discrete wavelet transform settings.
WAVELET = "db4"
WAVELET_LEVEL = 4

# Welch PSD bandpower ranges, in Hz.
FREQ_BANDS = {
    "bandpower_0_5hz": (0.0, 5.0),
    "bandpower_5_15hz": (5.0, 15.0),
    "bandpower_15_40hz": (15.0, 40.0),
}

# Search window for P-wave prominence, in seconds before the R-peak. Chosen
# to sit ahead of the QRS onset (which starts roughly 30-50ms before R for a
# normal beat) without running past PRE_WINDOW_S.
P_WAVE_SEARCH_START_S = 0.22
P_WAVE_SEARCH_END_S = 0.08

# Number of preceding RR intervals used as a per-patient local baseline when
# normalizing RR features (see ``compute_local_rr_ratios``).
LOCAL_RR_WINDOW = 9


def list_records(data_dir: Path = DATA_DIR) -> list[str]:
    """List MIT-BIH record names available in a data directory.

    Reads ``RECORDS`` if present (one record name per line), otherwise falls
    back to scanning for ``*.hea`` files.

    Args:
        data_dir: Directory containing the MIT-BIH .dat/.hea/.atr files.

    Returns:
        List of record name strings (e.g. ["100", "101", ...]).
    """
    records_file = data_dir / "RECORDS"
    if records_file.exists():
        return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    return sorted({p.stem for p in data_dir.glob("*.hea")})


def time_domain_features(window: np.ndarray) -> dict:
    """Compute time-domain statistical descriptors of a beat window.

    Args:
        window: 1-D array of raw signal samples covering one heartbeat segment.

    Returns:
        Dict with keys: rms, mean, std, skew, kurtosis, zero_crossing_rate.
    """
    std = float(np.std(window))
    centered_sign = np.sign(window - np.mean(window))
    zero_crossings = int(np.sum(np.diff(centered_sign) != 0))
    return {
        "rms": float(np.sqrt(np.mean(window ** 2))),
        "mean": float(np.mean(window)),
        "std": std,
        "skew": float(skew(window)) if std > 0 else 0.0,
        "kurtosis": float(kurtosis(window)) if std > 0 else 0.0,
        "zero_crossing_rate": zero_crossings / len(window),
    }


def estimate_qrs_duration(signal_full: np.ndarray, fs: int, r_peak: int) -> float:
    """Estimate QRS complex duration from the local signal slope around the R-peak.

    This is a lightweight heuristic, not a clinical-grade delineation
    algorithm. It computes |slope| in a window around R, smooths it slightly,
    and walks outward from R in both directions until the slope drops below
    a fraction of its peak value in that window — those crossing points are
    taken as the QRS onset and offset. Two simpler approaches were tried and
    rejected: a static amplitude threshold misfires because the QRS upstroke
    swings back through the pre-beat baseline level on its way from the Q
    trough to the R peak; and an un-smoothed slope walk misfires because the
    raw derivative is itself near zero exactly at the R-peak sample (it's a
    local extremum), which a naive walk reads as the complex already having
    ended. This version was validated against record 200: it separates
    narrow normal complexes (median ~81ms) from the wide, bizarre complexes
    of ventricular beats (median ~106ms), consistent with clinical expectation.

    Args:
        signal_full: Full 1-D signal array for the record.
        fs: Sampling frequency in Hz.
        r_peak: Sample index of the R-peak.

    Returns:
        Estimated QRS duration in seconds.
    """
    search_n = int(QRS_SEARCH_S * fs)
    start = max(0, r_peak - search_n)
    end = min(len(signal_full), r_peak + search_n + 1)
    seg = signal_full[start:end]
    r_local = r_peak - start

    deriv = np.abs(np.diff(seg))
    if len(deriv) == 0:
        return 0.0
    kernel = np.ones(QRS_SMOOTH_N) / QRS_SMOOTH_N
    deriv_smooth = np.convolve(deriv, kernel, mode="same")

    max_slope = np.max(deriv_smooth)
    if max_slope == 0:
        return 0.0
    threshold = QRS_THRESHOLD_FRAC * max_slope
    min_half_n = int(QRS_MIN_HALF_S * fs)

    onset = 0
    for i in range(max(0, r_local - min_half_n), -1, -1):
        if deriv_smooth[i] < threshold:
            onset = i + 1
            break

    offset = len(seg) - 1
    start_offset_search = min(r_local + min_half_n, len(deriv_smooth) - 1)
    for i in range(start_offset_search, len(deriv_smooth)):
        if deriv_smooth[i] < threshold:
            offset = i
            break

    return (offset - onset) / fs


def morphological_features(
    signal_full: np.ndarray,
    fs: int,
    r_peak: int,
    prev_r: int | None,
    next_r: int | None,
) -> dict:
    """Compute morphological and RR-interval features for one beat.

    Args:
        signal_full: Full 1-D signal array for the record.
        fs: Sampling frequency in Hz.
        r_peak: Sample index of this beat's R-peak.
        prev_r: Sample index of the preceding beat's R-peak, or None if this
            is the first beat in the record.
        next_r: Sample index of the following beat's R-peak, or None if this
            is the last beat in the record.

    Returns:
        Dict with keys: r_amplitude, qrs_duration, rr_prev, rr_next, rr_ratio.
        rr_prev/rr_next/rr_ratio are NaN when the neighboring beat doesn't exist.
    """
    rr_prev = (r_peak - prev_r) / fs if prev_r is not None else np.nan
    rr_next = (next_r - r_peak) / fs if next_r is not None else np.nan
    rr_ratio = rr_prev / rr_next if (prev_r is not None and next_r is not None and rr_next != 0) else np.nan

    return {
        "r_amplitude": float(signal_full[r_peak]),
        "qrs_duration": estimate_qrs_duration(signal_full, fs, r_peak),
        "rr_prev": rr_prev,
        "rr_next": rr_next,
        "rr_ratio": rr_ratio,
    }


def p_wave_amplitude(signal_full: np.ndarray, fs: int, r_peak: int) -> float:
    """Approximate P-wave prominence ahead of a beat's QRS complex.

    Takes the earliest sample in the P-wave search window as a local
    baseline and returns how far the signal rises above it within that
    window. This is a coarse proxy, not true P-wave delineation, but it
    carries real signal: supraventricular (S-class) beats arise from an
    ectopic atrial/junctional focus, so their preceding atrial activity is
    often absent, abnormally shaped, or mistimed relative to a normal sinus
    P-wave -- exactly the kind of difference this feature is meant to expose.

    Args:
        signal_full: Full 1-D signal array for the record.
        fs: Sampling frequency in Hz.
        r_peak: Sample index of the R-peak.

    Returns:
        P-wave prominence in the same units as the raw signal.
    """
    start = max(0, r_peak - int(P_WAVE_SEARCH_START_S * fs))
    end = max(start + 1, r_peak - int(P_WAVE_SEARCH_END_S * fs))
    seg = signal_full[start:end]
    baseline = seg[0]
    return float(np.max(seg) - baseline)


def compute_local_rr_ratios(rr_prev: np.ndarray, rr_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each beat's RR intervals against the patient's recent local rhythm.

    For each beat, divides its preceding and following RR interval by the
    median of the ``LOCAL_RR_WINDOW`` preceding RR intervals (excluding the
    current beat). This turns an absolute interval in seconds into a
    relative "how premature/delayed is this beat for this patient right
    now" signal, which generalizes far better across patients with
    different baseline heart rates than the raw RR features do -- it's one
    of the most consistently useful feature families in the inter-patient
    ECG classification literature for catching ectopic beats.

    Args:
        rr_prev: Per-beat array of seconds since the previous beat (NaN for
            the record's first beat).
        rr_next: Per-beat array of seconds until the next beat (NaN for the
            record's last beat).

    Returns:
        (rr_prev_local_ratio, rr_next_local_ratio) arrays, same length as the
        inputs, NaN wherever no local baseline or RR value is available.
    """
    n = len(rr_prev)
    prev_ratio = np.full(n, np.nan)
    next_ratio = np.full(n, np.nan)
    for i in range(n):
        window = rr_prev[max(0, i - LOCAL_RR_WINDOW):i]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            continue
        local_median = np.median(window)
        if local_median <= 0:
            continue
        if not np.isnan(rr_prev[i]):
            prev_ratio[i] = rr_prev[i] / local_median
        if not np.isnan(rr_next[i]):
            next_ratio[i] = rr_next[i] / local_median
    return prev_ratio, next_ratio


def frequency_domain_features(window: np.ndarray, fs: int) -> dict:
    """Compute Welch power spectral density bandpower over fixed frequency bands.

    Args:
        window: 1-D array of raw signal samples covering one heartbeat segment.
        fs: Sampling frequency in Hz.

    Returns:
        Dict with keys bandpower_0_5hz, bandpower_5_15hz, bandpower_15_40hz —
        each the integral of the Welch PSD estimate over that band.
    """
    nperseg = min(len(window), 128)
    freqs, psd = sp_signal.welch(window, fs=fs, nperseg=nperseg)

    out = {}
    for name, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return out


def wavelet_features(window: np.ndarray) -> dict:
    """Compute discrete wavelet transform coefficient statistics for a beat window.

    Decomposes the window with a db4 wavelet to ``WAVELET_LEVEL`` levels and
    summarizes each resulting coefficient array (one approximation array plus
    one detail array per level) with its mean and standard deviation.

    Args:
        window: 1-D array of raw signal samples covering one heartbeat segment.

    Returns:
        Dict with mean/std keys for each level, e.g. wavelet_cA4_mean,
        wavelet_cA4_std, wavelet_cD4_mean, wavelet_cD4_std, ..., wavelet_cD1_std.
    """
    coeffs = pywt.wavedec(window, WAVELET, level=WAVELET_LEVEL)
    level_names = [f"cA{WAVELET_LEVEL}"] + [f"cD{WAVELET_LEVEL - i}" for i in range(WAVELET_LEVEL)]

    out = {}
    for name, coeff in zip(level_names, coeffs):
        out[f"wavelet_{name}_mean"] = float(np.mean(coeff))
        out[f"wavelet_{name}_std"] = float(np.std(coeff))
    return out


def extract_beat_features(
    signal_full: np.ndarray,
    fs: int,
    r_peak: int,
    prev_r: int | None,
    next_r: int | None,
) -> dict:
    """Extract the full feature vector for one heartbeat.

    Combines time-domain, morphological, frequency-domain, and wavelet
    features computed over a fixed window centered on the beat's R-peak.

    Args:
        signal_full: Full 1-D signal array for the record (one lead).
        fs: Sampling frequency in Hz.
        r_peak: Sample index of this beat's R-peak.
        prev_r: Sample index of the preceding beat's R-peak, or None.
        next_r: Sample index of the following beat's R-peak, or None.

    Returns:
        Dict mapping feature names to values for this beat.
    """
    pre_n = int(PRE_WINDOW_S * fs)
    post_n = int(POST_WINDOW_S * fs)
    start = max(0, r_peak - pre_n)
    end = min(len(signal_full), r_peak + post_n)
    window = signal_full[start:end]

    features = {}
    features.update(time_domain_features(window))
    features.update(morphological_features(signal_full, fs, r_peak, prev_r, next_r))
    features.update(frequency_domain_features(window, fs))
    features.update(wavelet_features(window))
    features["p_wave_amplitude"] = p_wave_amplitude(signal_full, fs, r_peak)
    return features


def extract_record_features(record_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Extract beat-level features for every classifiable beat in one record.

    Loads the record's signal and annotation files, keeps only beats whose
    annotation symbol maps to an AAMI class (dropping rhythm/quality
    markers), and computes a feature row per beat.

    Args:
        record_name: MIT-BIH record name (e.g. "100").
        data_dir: Directory containing the MIT-BIH .dat/.hea/.atr files.

    Returns:
        DataFrame with one row per beat: feature columns plus record, sample,
        symbol (raw MIT-BIH annotation), and label (AAMI class).
    """
    record_path = str(data_dir / record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal_full = record.p_signal[:, 0]  # primary lead (e.g. MLII)
    fs = record.fs

    beat_mask = [sym in AAMI_MAP for sym in annotation.symbol]
    beat_locs = np.array(annotation.sample)[beat_mask]
    beat_symbols = np.array(annotation.symbol)[beat_mask]

    rr_prev_arr = np.full(len(beat_locs), np.nan)
    rr_next_arr = np.full(len(beat_locs), np.nan)
    if len(beat_locs) > 1:
        rr_prev_arr[1:] = np.diff(beat_locs) / fs
        rr_next_arr[:-1] = np.diff(beat_locs) / fs
    local_prev_ratio, local_next_ratio = compute_local_rr_ratios(rr_prev_arr, rr_next_arr)

    rows = []
    for i, (r_peak, symbol) in enumerate(zip(beat_locs, beat_symbols)):
        prev_r = beat_locs[i - 1] if i > 0 else None
        next_r = beat_locs[i + 1] if i < len(beat_locs) - 1 else None
        features = extract_beat_features(signal_full, fs, r_peak, prev_r, next_r)
        features["rr_prev_local_ratio"] = local_prev_ratio[i]
        features["rr_next_local_ratio"] = local_next_ratio[i]
        features["record"] = record_name
        features["sample"] = int(r_peak)
        features["symbol"] = symbol
        features["label"] = AAMI_MAP[symbol]
        rows.append(features)

    return pd.DataFrame(rows)


def main():
    records = list_records(DATA_DIR)

    all_features = []
    for record_name in records:
        print(f"Extracting features for record {record_name}...")
        try:
            all_features.append(extract_record_features(record_name, DATA_DIR))
        except FileNotFoundError as e:
            print(f"  skipping {record_name}: {e}")

    features_df = pd.concat(all_features, ignore_index=True)
    features_df.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"Wrote {len(features_df)} beat feature rows from {len(all_features)} records to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
