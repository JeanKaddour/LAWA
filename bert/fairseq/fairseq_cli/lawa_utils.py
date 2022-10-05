"""

Adopted from https://github.com/lucidrains/ema-pytorch

"""

import copy
import os

import torch
from omegaconf import DictConfig


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def average_model(cfg: DictConfig, model: torch.nn.Module):

    avg_model = None
    param_or_buffer_names_no_ema = set()
    num_avg_models = 0

    for epoch in range(
        cfg.avg.avg_start_idx,
        cfg.avg.avg_end_idx + 1,
    ):
        model.load_state_dict(
            torch.load(os.path.join(cfg.avg.avg_dir, f"checkpoint{epoch}.pt"))["model"]
        )
        num_avg_models += 1
        if num_avg_models == 1:
            avg_model = copy.deepcopy(model)
            avg_model.requires_grad_(False)
        if cfg.avg.avg_method == "swa":
            num_avg_models += 1
            swa_update(model, avg_model, param_or_buffer_names_no_ema, num_avg_models)
        elif cfg.avg.avg_method == "ema":
            ema_update(
                model,
                avg_model,
                param_or_buffer_names_no_ema,
                decay=cfg.avg.avg_ema_decay,
            )

    return avg_model


@torch.no_grad()
def ema_update(
    model,
    avg_model,
    param_or_buffer_names_no_ema,
    decay: float = 0.9,
):

    for (name, current_params), (_, ma_params) in zip(
        list(model.named_parameters()), list(avg_model.named_parameters())
    ):
        if not is_float_dtype(current_params.dtype):
            continue

        if name in param_or_buffer_names_no_ema:
            ma_params.data.copy_(current_params.data)
            continue

        difference = ma_params.data - current_params.data
        difference.mul_(1.0 - decay)
        ma_params.sub_(difference)

    for (name, current_buffer), (_, ma_buffer) in zip(
        list(model.named_buffers()), list(avg_model.named_buffers())
    ):
        if not is_float_dtype(current_buffer.dtype):
            continue

        if name in param_or_buffer_names_no_ema:
            ma_buffer.data.copy_(current_buffer.data)
            continue

        difference = ma_buffer - current_buffer
        difference.mul_(1.0 - decay)
        ma_buffer.sub_(difference)


@torch.no_grad()
def swa_update(model, avg_model, param_or_buffer_names_no_ema, num_avg_models):
    def avg_fn(averaged_model_parameter, model_parameter, num_avg_models):
        return averaged_model_parameter + (
            model_parameter - averaged_model_parameter
        ) / (num_avg_models + 1)

    for (name, current_params), (_, ma_params) in zip(
        list(model.named_parameters()), list(avg_model.named_parameters())
    ):
        if not is_float_dtype(current_params.dtype):
            continue

        if name in param_or_buffer_names_no_ema:
            ma_params.data.copy_(current_params.data)
            continue

        ma_params.data.copy_(
            avg_fn(ma_params.data, current_params.data, num_avg_models)
        )

    for (name, current_buffer), (_, ma_buffer) in zip(
        list(model.named_buffers()), list(avg_model.named_buffers())
    ):
        if not is_float_dtype(current_buffer.dtype):
            continue

        if name in param_or_buffer_names_no_ema:
            ma_buffer.data.copy_(current_buffer.data)
            continue

        ma_buffer.data.copy_(
            avg_fn(ma_buffer.data, current_buffer.data, num_avg_models)
        )
