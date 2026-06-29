import unittest

from src import ray_common
from src.tokenization.cli import parse_args
from src.tokenization import contract
from src.tokenization.runner import validate_cluster_resources


class _FakeRay:
    def __init__(self, *, cpus: float, gpus: float):
        self._resources = {"CPU": cpus, "GPU": gpus}

    def cluster_resources(self):
        return dict(self._resources)


class TokenizationCliTest(unittest.TestCase):
    def test_ray_common_reexports_the_tokenization_contract(self):
        for name in (
            "MERCHANT_HASH_MODE",
            "MERCHANT_HASH_SIZE",
            "SEQ_LENGTH",
            "SEQ_CHUNK_SIZE",
            "PAD_TOKEN_ID",
            "BOS_TOKEN_ID",
            "EOS_TOKEN_ID",
            "SEP_TOKEN_ID",
            "UNK_TOKEN_ID",
        ):
            self.assertEqual(getattr(ray_common, name), getattr(contract, name))

    def test_normal_default_matches_documented_two_gpu_deployment(self):
        args = parse_args(["unused-splits"])

        self.assertIsNone(args.profile)
        self.assertEqual(args.actors, 2)
        self.assertEqual(args.num_gpus_per_actor, 1.0)

    def test_v3_profile_supplies_audited_four_gpu_defaults(self):
        args = parse_args(["unused-splits", "--profile", "v3-4x-v100"])

        self.assertEqual(args.profile, "v3-4x-v100")
        self.assertEqual(args.actors, 4)
        self.assertEqual(args.num_cpus_per_actor, 16)
        self.assertEqual(args.local_num_cpus, 64)
        self.assertEqual(args.local_num_gpus, 4)
        self.assertEqual(args.output_dtype, "uint16")
        self.assertEqual(args.compression, "none")

    def test_explicit_flags_override_profile(self):
        args = parse_args(
            [
                "unused-splits",
                "--profile",
                "v3-4x-v100",
                "--actors",
                "3",
                "--compression",
                "lz4",
            ]
        )

        self.assertEqual(args.actors, 3)
        self.assertEqual(args.compression, "lz4")

    def test_resource_check_accepts_matching_cluster(self):
        args = parse_args(
            [
                "unused-splits",
                "--profile",
                "v3-4x-v100",
                "--ray-address",
                "local",
            ]
        )

        validate_cluster_resources(_FakeRay(cpus=64, gpus=4), args)

    def test_resource_check_fails_before_unschedulable_actors(self):
        args = parse_args(
            [
                "unused-splits",
                "--profile",
                "v3-4x-v100",
                "--ray-address",
                "local",
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "4 GPUs requested, 2 available; 64 CPUs requested, 32 available",
        ):
            validate_cluster_resources(_FakeRay(cpus=32, gpus=2), args)

    def test_resource_check_does_not_apply_fast_actor_contract_to_legacy(self):
        args = parse_args(
            ["unused-splits", "--engine", "legacy", "--ray-address", "local"]
        )

        validate_cluster_resources(_FakeRay(cpus=1, gpus=0), args)

    def test_resource_check_allows_remote_autoscaling(self):
        args = parse_args(["unused-splits", "--actors", "4", "--ray-address", "auto"])

        validate_cluster_resources(_FakeRay(cpus=1, gpus=0), args)


if __name__ == "__main__":
    unittest.main()
