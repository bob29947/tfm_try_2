import unittest
from pathlib import Path

import yaml

from benchmarks.tokenization.cluster.cluster_admin import EXPECTED_VERSIONS


TFM_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_YAML = (
    TFM_ROOT
    / "benchmarks"
    / "tokenization"
    / "cluster"
    / "ray-tokenization-s3-4xl4.yaml"
)


class CloudClusterConfigTest(unittest.TestCase):
    def test_numpy_scipy_abi_pair_is_pinned_and_strictly_inventoried(self):
        config = yaml.safe_load(CLUSTER_YAML.read_text())
        setup = "\n".join(config["setup_commands"])

        self.assertEqual(EXPECTED_VERSIONS["numpy"], "2.2.6")
        self.assertEqual(EXPECTED_VERSIONS["scipy"], "1.15.3")
        self.assertIn('"numpy==2.2.6"', setup)
        self.assertIn('"scipy==1.15.3"', setup)

    def test_setup_exercises_the_cupy_scipy_import_path(self):
        config = yaml.safe_load(CLUSTER_YAML.read_text())
        setup = "\n".join(config["setup_commands"])

        self.assertIn("import cupyx.scipy,scipy", setup)
        self.assertIn("linalg.det(numpy.eye(2))", setup)


if __name__ == "__main__":
    unittest.main()
