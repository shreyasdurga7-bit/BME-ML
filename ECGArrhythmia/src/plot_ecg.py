"""Plot ECG signals from the MIT-BIH Arrhythmia Database under ``data/``.

Usage:
    python src/plot_ecg.py --record 100
    python src/plot_ecg.py --record 100 --start 0 --duration 10
    python src/plot_ecg.py --record 100 --lead V5 --no-annotations
    python src/plot_ecg.py --record 100 --output ecg_100.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Same beat symbols recognized in features.py, used to mark annotations on the plot.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


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
            if symbol not in BEAT_SYMBOLS or not (start_n <= sample < end_n):
                continue
            ax.axvline(sample / fs, color="red", linewidth=0.5, alpha=0.4)
            ax.annotate(symbol, (sample / fs, signal[sample]), color="red", fontsize=8, ha="center", va="bottom")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, help="Record name to plot (e.g. 100)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing MIT-BIH .dat/.hea/.atr files")
    parser.add_argument("--lead", default=None, help="Lead to plot (default: first lead, e.g. MLII)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds to plot (default: 10; use 0 for full record)")
    parser.add_argument("--no-annotations", action="store_true", help="Don't mark annotated beats")
    parser.add_argument("--output", type=Path, default=None, help="Save plot to this path instead of displaying it")
    args = parser.parse_args()

    plot_record(
        record_name=args.record,
        data_dir=args.data_dir,
        lead=args.lead,
        start_s=args.start,
        duration_s=None if args.duration == 0 else args.duration,
        show_annotations=not args.no_annotations,
        output=args.output,
    )


if __name__ == "__main__":
    main()
