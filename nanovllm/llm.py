from nanovllm.utils.cuda_runtime import preload_cuda_nvrtc_libs

preload_cuda_nvrtc_libs()

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
