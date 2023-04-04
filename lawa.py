import copy
import dataclasses
from collections import defaultdict, deque
from typing import Optional

import torch


@dataclasses.dataclass
class LAWAScheduler:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    param_store: deque = deque()
    buffer_store: deque = deque()
    freq: int = 1000
    k: int = 10
    steps: int = 0
    start_steps: int = 1000
    reset_optimizer: bool = True
    enabled: bool = True
    keep_separate_avg_model: bool = True
    avg_model: Optional[torch.nn.Module] = None

    def __post_init__(self) -> None:
        self._initialize_avg_model()

    def __repr__(self) -> str:
        return f"LAWAScheduler(steps={self.steps},freq={self.freq},k={self.k},start_steps={self.start_steps},enabled={self.enabled})"

    def _initialize_avg_model(self) -> None:
        if self.keep_separate_avg_model and self.avg_model is None:
            self.avg_model = copy.deepcopy(self.model)

    @torch.no_grad()
    def transfer_avg_weights_from_store(self) -> None:
        model_to_be_transferred = (
            self.avg_model if self.keep_separate_avg_model else self.model
        )
        # average all parameters
        avg_params = copy.deepcopy(self.param_store[0])
        for params in list(self.param_store)[1:]:
            for avg_param, param in zip(avg_params, params):
                avg_param.add_(param)
        for avg_param in avg_params:
            avg_param.div_(len(self.param_store))
        for param, avg_param in zip(model_to_be_transferred.parameters(), avg_params):
            param.copy_(avg_param.data)

        # average all buffers if available
        if len(self.buffer_store[0]) > 0:
            avg_buffers = copy.deepcopy(self.buffer_store[0])
            for buffers in list(self.buffer_store)[1:]:
                for avg_buffer, buffer in zip(avg_buffers, buffers):
                    avg_buffer.add_(buffer)
            for avg_buffer in avg_buffers:
                avg_buffer.div_(len(self.buffer_store))
            # transfer averaged parameters and buffers to model
            for buffer, avg_buffer in zip(
                model_to_be_transferred.buffers(), avg_buffers
            ):
                buffer.copy_(avg_buffer.data)

    def update_store(self):
        params = [p.detach().cpu() for p in self.model.parameters()]
        buffers = [b.detach().cpu() for b in self.model.buffers()]
        self.param_store.append(params)
        self.buffer_store.append(buffers)
        if len(self.param_store) > self.k:
            self.param_store.popleft()
            self.buffer_store.popleft()

    def step(self):
        if self.enabled:
            self.steps += 1
            if (self.steps % self.freq) == 0:
                self.update_store()
            if self.steps >= self.start_steps and self.steps % self.freq == 0:
                self.transfer_avg_weights_from_store()
                if self.reset_optimizer:
                    # reset optimizer state
                    self.optimizer.state = defaultdict(dict)
