import csv
import unittest
from tempfile import TemporaryDirectory

from matplotlib import image

from ddcfr.utils.convergence import ConvergenceRecorder


class ConvergenceRecorderTest(unittest.TestCase):
    def test_measurements_are_flushed_and_close_renders_a_headless_png(self):
        with TemporaryDirectory() as folder:
            recorder = ConvergenceRecorder(folder, "Test convergence")
            self.addCleanup(recorder.close)
            recorder.record(0, 0.5, 0.0)
            recorder.record(1, 1e-12, 0.1, (1.5, -0.5, 2.0, 1))
            # Data can be read while the run is still active.
            with recorder.csv_path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]["exploitability"]), 1e-12)
            recorder.close()
            pixels = image.imread(recorder.plot_path)
            self.assertGreater(pixels.std(), 0)

    def test_existing_measurements_are_not_overwritten(self):
        with TemporaryDirectory() as folder:
            recorder = ConvergenceRecorder(folder, "First run")
            recorder.record(0, 0.5, 0.0)
            recorder.close()
            original = recorder.csv_path.read_bytes()

            with self.assertRaises(FileExistsError):
                ConvergenceRecorder(folder, "Second run")

            self.assertEqual(recorder.csv_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
