
from contextlib import ContextDecorator
import os
import random
import time
from typing import Callable
import numpy as np
from transformers import PreTrainedModel

import torch


type ModelFn = Callable[[torch.Tensor], torch.Tensor]

def model_fn(model: PreTrainedModel) -> ModelFn:
    def forward(input: torch.Tensor, mask: torch.Tensor=None):
        output = model(input_ids=input, attention_mask=mask, use_cache=False)
        return output.logits
    
    return forward

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Benchmark(ContextDecorator):
    def __init__(self, label):
        self.label = label
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        elapsed_ms = elapsed_time * 1000
        
        print(f"Benchmark <{self.label}>: {elapsed_ms:.0f}ms")
        
        return False
