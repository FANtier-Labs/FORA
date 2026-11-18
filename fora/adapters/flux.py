"""FORA adapter for the Diffusers FLUX.1 text-to-image pipeline."""

from contextlib import ExitStack, contextmanager
from functools import wraps
from threading import Lock

import torch

from ..core import CacheManager
from ..integration.patch import patch_forward
from .base import BaseAdapter


_patch_lock = Lock()


@contextmanager
def _claim_transformer(transformer):
    with _patch_lock:
        if getattr(transformer, "_fora_active", False):
            raise ValueError("FLUX transformer already has an active FORA patch")
        transformer._fora_active = True
    try:
        yield
    finally:
        with _patch_lock:
            del transformer._fora_active


def _tensor_signature(*tensors):
    return tuple((tensor.shape, tensor.device, tensor.dtype) for tensor in tensors)


class FluxAdapter(BaseAdapter):
    @contextmanager
    def patch(self, pipeline, manager: CacheManager):
        """Cache pre-gate features for the duration of one FluxPipeline call."""
        transformer = pipeline.transformer
        with manager._request(), _claim_transformer(transformer), ExitStack() as stack:
            if manager.config.interval == 1:
                yield
                return

            if transformer.training or getattr(transformer, "is_cache_enabled", False):
                raise ValueError("FORA requires eval mode and disabled Diffusers cache")
            if hasattr(transformer, "_orig_mod") or transformer._compiled_call_impl is not None:
                raise ValueError("FORA requires eager execution")

            timestep = None
            step = -1
            branch = 0
            context = None
            original = transformer.forward

            @wraps(original)
            def forward(*args, **kwargs):
                nonlocal timestep, step, branch, context
                if torch.is_grad_enabled():
                    raise ValueError("FORA requires no_grad or inference_mode")
                if torch.compiler.is_compiling():
                    raise ValueError("FORA requires eager execution")
                current = pipeline.current_timestep
                if current is None:
                    raise RuntimeError("FORA requires an active FluxPipeline denoising loop")
                # FluxPipeline assigns one timestep object to both CFG calls in a step.
                if current is not timestep:
                    timestep, step, branch = current, step + 1, 0
                else:
                    branch += 1
                if branch > 1:
                    raise ValueError("expected at most two FLUX transformer calls per step")
                context = ("cond", "uncond")[branch]
                try:
                    return original(*args, **kwargs)
                finally:
                    context = None

            def cached(key, compute, *inputs):
                if context is None:
                    return compute()
                return manager.wrap(
                    key, step, compute, context, input_signature=_tensor_signature(*inputs)
                )

            stack.enter_context(patch_forward(transformer, forward))
            for index, block in enumerate(transformer.transformer_blocks):
                for name in ("attn", "ff", "ff_context"):
                    module = getattr(block, name)
                    wrapped = self._double_forward(
                        module, f"double.{index}.{name}", cached, has_context=name == "attn"
                    )
                    stack.enter_context(patch_forward(module, wrapped))
            for index, block in enumerate(transformer.single_transformer_blocks):
                wrapped = self._single_forward(block, f"single.{index}.proj_out", cached)
                stack.enter_context(patch_forward(block, wrapped))
            yield

    @staticmethod
    def _double_forward(module, key, cached, *, has_context):
        original = module.forward

        @wraps(original)
        def forward(hidden_states, *args, **kwargs):
            inputs = (hidden_states,)
            if has_context:
                encoder_hidden_states = args[0] if args else kwargs.get("encoder_hidden_states")
                inputs += (encoder_hidden_states,)
            return cached(key, lambda: original(hidden_states, *args, **kwargs), *inputs)

        return forward

    @staticmethod
    def _single_forward(block, key, cached):
        @wraps(block.forward)
        def forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            text_length = encoder_hidden_states.shape[1]
            residual = torch.cat((encoder_hidden_states, hidden_states), dim=1)
            normalized, gate = block.norm(residual, emb=temb)

            def compute():
                mlp = block.act_mlp(block.proj_mlp(normalized))
                attention = block.attn(
                    hidden_states=normalized,
                    image_rotary_emb=image_rotary_emb,
                    **(joint_attention_kwargs or {}),
                )
                return block.proj_out(torch.cat((attention, mlp), dim=-1))

            feature = cached(key, compute, hidden_states, encoder_hidden_states, normalized)
            output = residual + gate.unsqueeze(1) * feature
            if output.dtype == torch.float16:
                output = output.clip(-65504, 65504)
            return output[:, :text_length], output[:, text_length:]

        return forward
