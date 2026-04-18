from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    penalty_decay: float = 0.996
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or a positive integer"
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        assert 0.0 <= self.penalty_decay <= 1.0, "penalty_decay must be in [0, 1]"
        assert self.max_tokens > 0, "max_tokens must be positive"
