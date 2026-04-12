import sys
import unittest
from pathlib import Path
from unittest import mock

import scripts.run_tests as run_tests


class RunTestsScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(run_tests.__file__).resolve().parents[1]

    @mock.patch("scripts.run_tests.run")
    def test_main_prepares_env_then_runs_discovery(self, run_mock: mock.Mock) -> None:
        with mock.patch.object(sys, "argv", ["run_tests.py"]):
            self.assertEqual(run_tests.main(), 0)

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(
            run_mock.call_args_list[0],
            mock.call(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "prepare_test_env.py"),
                    "--repo-root",
                    str(self.repo_root),
                ],
                cwd=self.repo_root,
            ),
        )
        self.assertEqual(
            run_mock.call_args_list[1],
            mock.call(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_*.py",
                ],
                cwd=self.repo_root,
            ),
        )

    @mock.patch("scripts.run_tests.run")
    def test_main_can_skip_prepare_env(self, run_mock: mock.Mock) -> None:
        with mock.patch.object(sys, "argv", ["run_tests.py", "--skip-prepare-test-env"]):
            self.assertEqual(run_tests.main(), 0)

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(
            run_mock.call_args,
            mock.call(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_*.py",
                ],
                cwd=self.repo_root,
            ),
        )

    @mock.patch("scripts.run_tests.run")
    def test_main_passes_prepare_env_overrides(self, run_mock: mock.Mock) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_tests.py",
                "--skip-albatross",
                "--albatross-url",
                "https://example.com/albatross.git",
                "--albatross-dest",
                "vendor/Albatross",
                "--albatross-ref",
                "topic",
                "tests.test_run_tests",
            ],
        ):
            self.assertEqual(run_tests.main(), 0)

        self.assertEqual(
            run_mock.call_args_list[0],
            mock.call(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "prepare_test_env.py"),
                    "--repo-root",
                    str(self.repo_root),
                    "--skip-albatross",
                    "--albatross-url",
                    "https://example.com/albatross.git",
                    "--albatross-dest",
                    "vendor/Albatross",
                    "--albatross-ref",
                    "topic",
                ],
                cwd=self.repo_root,
            ),
        )
        self.assertEqual(
            run_mock.call_args_list[1],
            mock.call(
                [sys.executable, "-m", "unittest", "tests.test_run_tests"],
                cwd=self.repo_root,
            ),
        )


if __name__ == "__main__":
    unittest.main()
