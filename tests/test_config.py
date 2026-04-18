import os
import unittest
from unittest.mock import patch

from nanovllm.config import Config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "/tmp/nanovllm-config-test/model.pth"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb"):
            pass
        self.fake_model_config = type("FakeModelConfig", (), {"max_position_embeddings": 8192})()

    def _build_config(self, **kwargs) -> Config:
        with (
            patch("nanovllm.config.resolve_model_pth", return_value=self.model_path),
            patch("nanovllm.config.RWKV7Config.from_pth", return_value=self.fake_model_config),
        ):
            return Config(self.model_path, **kwargs)

    def test_preserves_explicit_gpu_memory_utilization_point_nine(self):
        config = self._build_config(gpu_memory_utilization=0.9)
        self.assertEqual(config.gpu_memory_utilization, 0.9)

    def test_accepts_state_cache_safety_reserve_slots(self):
        config = self._build_config(rwkv_state_cache_safety_reserve_slots=7)
        self.assertEqual(config.rwkv_state_cache_safety_reserve_slots, 7)

    def test_rejects_negative_state_cache_safety_reserve_slots(self):
        with self.assertRaises(AssertionError):
            self._build_config(rwkv_state_cache_safety_reserve_slots=-1)


if __name__ == "__main__":
    unittest.main()
