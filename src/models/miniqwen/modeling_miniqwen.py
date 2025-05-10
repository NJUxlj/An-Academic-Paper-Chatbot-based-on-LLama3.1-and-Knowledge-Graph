
"""Mini Qwen model implementation based on Transformer architecture."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_miniqwen import MiniQwenConfig


logger = logging.get_logger(__name__)





class RMSNorm(nn.Module):
    """
    RMSNorm归一化层
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


def rotate_half(x):
    """将tensor的后半部分旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """应用旋转位置编码 (RoPE)"""
    # 获取查询和键的形状
    batch_size, seq_length, head_dim = q.shape[0], q.shape[1], q.shape[3]
    
    # 根据position_ids获取对应的cos和sin值
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    
    # 应用旋转位置编码
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class MiniQwenAttention(nn.Module):
    """
    Group Query Attention (GQA) 实现
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_grouping = self.num_heads // self.num_kv_heads  # 每个KV头对应多少个Q头
        
        self.max_position_embeddings = config.max_position_embeddings
        
        # 计算注意力投影矩阵
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 初始化RoPE位置编码
        self.rotary_emb = self._init_rope()

    def _init_rope(self):
        """
        初始化RoPE（Rotary Position Embeddings）
        """
        # 计算RoPE的cos和sin表
        head_dim = self.head_dim
        theta = self.config.rope_theta
        max_position_embeddings = self.max_position_embeddings
        
        # 计算每个维度的频率
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # 创建位置索引
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
        
        # 计算cos和sin值
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(max_position_embeddings, -1)  # [seq_len, dim]
        sin = emb.sin().view(max_position_embeddings, -1)
        
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 获取输入的形状
        batch_size, seq_length, _ = hidden_states.shape
        
        # 计算查询、键和值投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑形状以准备多头注意力计算
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 处理KV缓存（用于生成加速）
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        # 添加过去的KV缓存
        if past_key_value is not None:
            past_key = past_key_value[0]
            past_value = past_key_value[1]
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # 创建当前KV缓存
        past_key_value = (key_states, value_states) if use_cache else None
        
        # 应用旋转位置编码 (RoPE)
        cos, sin = self.rotary_emb
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
        
        # 应用RoPE到查询和键
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # 处理GQA分组
        # 如果num_kv_heads小于num_heads，则扩展key和value
        if self.num_kv_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.q_grouping, dim=1)
            value_states = value_states.repeat_interleave(self.q_grouping, dim=1)
        
        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 确保掩码适合注意力权重的形状
            if attention_mask.dim() == 2:  # [batch_size, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_length]
            
            # 掩码应用
            attn_weights = attn_weights + attention_mask
        
        # 使用Softmax归一化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # 最终输出投影
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output, past_key_value)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs


class MiniQwenMLP(nn.Module):
    """
    MLP实现，使用SwiGLU激活函数
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU激活
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        # SwiGLU: GLU(x) = gate * up
        return self.down_proj(gate * up)


class MiniQwenLayer(nn.Module):
    """
    Transformer层实现
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 注意力前归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 多头注意力层
        self.self_attn = MiniQwenAttention(config)
        
        # 前馈网络前归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络
        self.mlp = MiniQwenMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 应用LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 自注意力
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        attn_output = attn_outputs[0]
        # 残差连接
        hidden_states = residual + attn_output
        
        # 应用第二个LayerNorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP层
        mlp_output = self.mlp(hidden_states)
        
        # 第二个残差连接
        hidden_states = residual + mlp_output
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (attn_outputs[1],)  # past_key_value
        if output_attentions:
            outputs = outputs + (attn_outputs[-1],)  # attention weights
            
        return outputs


class MiniQwenPreTrainedModel(PreTrainedModel):
    """
    预训练模型基类
    """
    config_class = MiniQwenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniQwenLayer"]
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class MiniQwenModel(MiniQwenPreTrainedModel):
    """
    Transformer模型主体
    """
    def __init__(self, config: MiniQwenConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([MiniQwenLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 最终的LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 设置默认参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取输入的嵌入表示
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # 处理past_key_values，用于生成加速
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        # 生成位置ID（如果需要）
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).view(-1, seq_length)
        
        # 生成注意力掩码（如果需要）
        if attention_mask is not None:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # 将填充位置的注意力分数设置为负无穷
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        # 初始化hidden_states和all_hidden_states
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # 通过每个Transformer层
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # 应用层函数
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[1],)
                
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[-1],)
        
        # 应用最终的LayerNorm
        hidden_states = self.norm(hidden_states)
        
        # 添加到final_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # 缓存更新
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniQwenForCausalLM(MiniQwenPreTrainedModel):
    """
    用于因果语言建模的模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniQwenModel(config)
        self.vocab_size = config.vocab_size
        
        # 语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 设置默认参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 通过Transformer模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取最后的隐藏状态
        hidden_states = outputs[0]
        
        # 应用语言模型头部
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 将标签移位作为预测目标
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算损失函数
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """准备生成文本的输入"""
        # 仅使用最后一个输入ID进行有效生成
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        # 如果使用输入嵌入，则调整相应大小
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
            
        # 添加past_key_values和attention_mask
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """在束搜索过程中重新排序past_key_values"""
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )
