# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from types import MethodType
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed
import transformers
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

try:
    from transformers.models.mistral.modeling_mistral import (
        MISTRAL_INPUTS_DOCSTRING,
        _CONFIG_FOR_DOC,
    )

    _MISTRAL_MODELING_LEGACY = True
except ImportError:
    from transformers.models.mistral.configuration_mistral import MistralConfig

    _CONFIG_FOR_DOC = MistralConfig
    MISTRAL_INPUTS_DOCSTRING = ""
    _MISTRAL_MODELING_LEGACY = False

from .utils import PatchOptions, TransformersModelT, apply_lce

_PATCH_OPTS: PatchOptions | None = None


@add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Any,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            Calculate logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, MistralForCausalLM

    >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if "num_logits_to_keep" in kwargs:
        logits_to_keep = kwargs.pop("num_logits_to_keep")

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    _model_kw: dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )
    if _MISTRAL_MODELING_LEGACY and return_dict is not None:
        _model_kw["return_dict"] = return_dict
    outputs = self.model(**_model_kw, **kwargs)

    hidden_states = outputs[0]
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(hidden_states, self.lm_head.weight, labels, _PATCH_OPTS, **kwargs)
    else:
        _slice = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, _slice, :])

        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_mistral(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.mistral import modeling_mistral

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral.MistralForCausalLM
        ), f"Expected a MistralForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model
    else:
        modeling_mistral.MistralForCausalLM.forward = cce_forward
