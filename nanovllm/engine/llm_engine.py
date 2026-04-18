import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.tokenizers import get_rwkv_tokenizer


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = get_rwkv_tokenizer()
        config.eos = self.tokenizer.eos_token_id
        config.stop_token_seqs = self.tokenizer.get_default_stop_token_seqs()
        self.model_runner.eos = config.eos
        self.model_runner.stop_token_seqs = config.stop_token_seqs
        self.scheduler = Scheduler(config)
        if config.rwkv_state_cache_enable:
            self.scheduler.prefix_index.cache_key_token_rewriter = self.tokenizer.canonicalize_state_cache_token_ids
            self.model_runner.attach_state_cache(self.scheduler.slot_manager, self.scheduler.prefix_index)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.allow_sparse_penalty_state = True
        self.scheduler.add(seq)
        return seq

    def abort(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        prefill_tokens = (
            sum(seq.prefill_step_tokens(self.scheduler.config.rwkv_prefill_chunk_size) for seq in seqs)
            if is_prefill
            else 0
        )
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = prefill_tokens if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
