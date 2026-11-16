"""Persist measured exploitability and render a convergence curve without a GUI."""

import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_convergence_plot(series, path, title):
    """Plot actual measurements for one or more agents on the same log scale."""
    figure = Figure(figsize=(9, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for label, history in series.items():
        axis.plot(
            [row["iteration"] for row in history],
            [row["exploitability"] for row in history],
            label=label,
            linewidth=1.5,
            marker="." if len(history) < 100 else None,
        )
    axis.set_yscale("log")
    axis.set_xlabel("DDCFR iteration")
    axis.set_ylabel("Exploitability (lower is stronger)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    if len(series) > 1:
        axis.legend()
    figure.savefig(path, dpi=150)


class ConvergenceRecorder:
    def __init__(self, folder, title):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.folder / "convergence.csv"
        self.plot_path = self.folder / "convergence.png"
        self.title = title
        self.history = []
        # An explicit output directory must not overwrite a previous experiment.
        self._file = self.csv_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "iteration",
                "exploitability",
                "elapsed_seconds",
                "alpha",
                "beta",
                "gamma",
                "tau",
            ],
        )
        self._writer.writeheader()
        self._file.flush()

    def record(self, iteration, exploitability, elapsed_seconds, action=None):
        """Append an actual measurement; tau describes the selected block."""
        row = {
            "iteration": iteration,
            "exploitability": exploitability,
            "elapsed_seconds": elapsed_seconds,
        }
        if action is not None:
            row.update(zip(("alpha", "beta", "gamma", "tau"), action))
        self._writer.writerow(row)
        self._file.flush()
        self.history.append(row)

    def save_plot(self):
        if not self.history:
            return
        save_convergence_plot(
            {"DDCFR+MCTS": self.history}, self.plot_path, self.title,
        )

    def close(self):
        if self._file.closed:
            return
        try:
            self.save_plot()
        finally:
            self._file.close()
