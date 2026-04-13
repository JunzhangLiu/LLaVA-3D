#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast,BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention,repeat_kv,apply_rotary_pos_emb
import math

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig,prune_ratio=None):
        super(LlavaLlamaModel, self).__init__(config)
        self.prune_ratio=prune_ratio
        if self.prune_ratio is None:
            self.prune_ratio = {}
    # def forward(self,*args,**kwargs):
        # return super(LlavaLlamaModel, self).forward(*args,**kwargs)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        img_start_end=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # elif self._use_sdpa and not output_attentions:
        #     # output_attentions=True can not be supported when using SDPA, and we fall back on
        #     # the manual implementation that requires a 4D causal mask in all cases.
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #     )
        # else:
            # 4d mask is passed through the layers
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = ()
        next_decoder_cache = None
        # self.prune_ratio
        # prune_layer = {}
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if layer_idx in self.prune_ratio:
                output_attentions = True
            else:
                output_attentions = False

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask if output_attentions else None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions and hidden_states.shape[1]>1:
                all_self_attns += (layer_outputs[1],)
            elif output_attentions and hidden_states.shape[1]==1:
                position_ids = torch.tensor([[past_key_values.key_cache[layer_idx+1].shape[2]]*position_ids.shape[0]],device=position_ids.device,dtype=position_ids.dtype)
                key_len = past_key_values.key_cache[layer_idx+1].shape[2]

                causal_mask = _prepare_4d_causal_attention_mask(
                    attention_mask[:,:position_ids[0,0]+1], (batch_size, seq_length), hidden_states, key_len
                )
                continue
            else:
                continue
            attn_weights = layer_outputs[1]
            pruned_hidden_states = []
            pruned_position_ids = []
            
            for i in range(batch_size):
                attn_weight,img_start,img_end = attn_weights[i],img_start_end[0][i],img_start_end[1][i]
                attn_mask = causal_mask[i]
                attn_weight = attn_weight.mean(0)
                attn_importance = attn_weight.sum(0)/(attn_mask[0]==0).sum(0)
                img_attn_importance=attn_importance[img_start:img_end]
                quantile_attn = torch.quantile(img_attn_importance.float(),self.prune_ratio[layer_idx]).to(dtype=img_attn_importance.dtype)
                img_attn_prune_mask = quantile_attn<img_attn_importance
                pruned_hidden_state = torch.cat([hidden_states[i,:img_start],hidden_states[i,img_start:img_end][img_attn_prune_mask],hidden_states[i,img_end:]],dim=0)
                pruned_position_id = torch.arange(0,pruned_hidden_state.shape[0],1,dtype=int,device=pruned_hidden_state.device)
                pruned_hidden_states.append(pruned_hidden_state)
                pruned_position_ids.append(pruned_position_id)
            max_len = max([len(x) for x in pruned_hidden_states])
            pruned_hidden_states = [
                torch.cat([x.new_zeros((max_len-len(x),x.shape[1])),x],dim=0)
                for x in pruned_hidden_states
                ]
            hidden_states = torch.stack(pruned_hidden_states,dim=0)

            pruned_position_ids=[
                torch.cat([x.new_zeros(max_len-len(x)),x],dim=0)
                for x in pruned_position_ids
                ]
            position_ids = torch.stack(pruned_position_ids,dim=0)
            batch_size, seq_length = hidden_states.shape[:2]
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask[:,:seq_length], (batch_size, seq_length), hidden_states, past_key_values_length
            )
            

            

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config,prune_ratio=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config,prune_ratio=prune_ratio)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.FloatTensor] = None,
        clicks: Optional[List[List[float]]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        img_start_end=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                depths,
                poses,
                intrinsics,
                lengths,
                clicks,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            img_start_end=img_start_end
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.FloatTensor] = None,
        clicks: Optional[List[List[float]]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                img_start_end
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                depths,
                poses,
                intrinsics,
                lengths,
                clicks,
                image_sizes=image_sizes
            )
            kwargs['img_start_end'] = img_start_end
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs['img_start_end'] = kwargs['img_start_end']
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    def _validate_model_kwargs(self,model_kwargs):
        pass
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
