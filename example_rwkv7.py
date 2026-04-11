#!/usr/bin/env python3
import os
from nanovllm import LLM, SamplingParams


def main():
    model_path = "/models/rwkv7-g1e-1.5b-20260309-ctx8192.pth"

    print(f"Loading model: {model_path}")

    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_model_len=2048)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=100)
    prompts_text = [
        "The quick brown fox",
        "In 2025, artificial intelligence",
    ]

    print(f"\nGenerating...")
    outputs = llm.generate(prompts_text, sampling_params)

    for prompt_text, output in zip(prompts_text, outputs):
        print(f"\nPrompt: {prompt_text!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
