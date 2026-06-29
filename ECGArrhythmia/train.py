"""Entry point for running the full training pipeline (see src/model.py).

Invoking ``python src/model.py`` directly would execute that module as
``__main__``, which makes classes it defines (like ``SoftVotingEnsemble``)
unpicklable from any other context (e.g. the analysis notebook) — Python
records a pickled class's module as whatever ``__name__`` was at definition
time. Importing ``src.model`` normally, as this script does, keeps its
classes addressable as ``src.model.SoftVotingEnsemble`` everywhere, so the
joblib artifacts it writes to ``results/`` can be loaded back from the
notebook.

Usage:
    python train.py
    python train.py --quick
"""

from src.model import main

if __name__ == "__main__":
    main()
