"""Plot ECG signals from the MIT-BIH Arrhythmia Database under ``data/``.

Usage:
    python src/plot_ecg.py

    When run, prompts for a record number (e.g. 100) and plots the first
    10 seconds of the default lead with beat annotations overlaid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

#Used to read ECG signals in .dat format from Physionet Dataset
import wfdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Maps each MIT-BIH beat symbol to its clinical name.
# Used to filter wfdb annotations to heartbeat events only (non-beat annotations are excluded).
BEAT_SYMBOLS = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "B": "Bundle branch block beat (unspecified)",
    "A": "Atrial premature beat",
    "a": "Aberrated atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature beat",
    "V": "Premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "e": "Atrial escape beat",
    "j": "Nodal (junctional) escape beat",
    "n": "Supraventricular escape beat",
    "E": "Ventricular escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal beat",
    "Q": "Unclassifiable beat",
    "?": "Beat not classified during learning",
}


def plot_record(
    record_name: str,
    data_dir: Path,
    lead: str | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    show_annotations: bool = True,
    output: Path | None = None,
):
    record_path = str(data_dir / record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs

    lead_idx = record.sig_name.index(lead) if lead else 0
    signal = record.p_signal[:, lead_idx]

    start_n = int(start_s * fs)
    end_n = len(signal) if duration_s is None else min(len(signal), start_n + int(duration_s * fs))
    t = np.arange(start_n, end_n) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, signal[start_n:end_n], color="black", linewidth=0.8)

    if show_annotations:
        annotation = wfdb.rdann(record_path, "atr")
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in BEAT_SYMBOLS.keys() or not (start_n <= sample < end_n):
                continue
            ax.axvline(sample / fs, color="red", linewidth=0.5, alpha=0.4)
            ax.annotate(symbol, (sample / fs, signal[sample]), color="red", fontsize=8, ha="center", va="bottom",
                        gid=BEAT_SYMBOLS[symbol])

    ax.set_title(f"Record {record_name} — lead {record.sig_name[lead_idx]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    plt.show()
    return fig


def main():
    record = input("Enter record number: ").strip()

    plot_record(
        record_name=record,
        data_dir=DATA_DIR,
        lead=None,
        start_s=0.0,
        duration_s=10.0,
        show_annotations=True,
        output=None,
    )


if __name__ == "__main__":
    main()
