"""
65M Decoder-only LLM
基于 MiniMind 架构，简化为 65M 参数配置
支持部分冻结策略
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


@dataclass
class LLMConfig:
    """LLM 配置"""
    vocab_size: int = 32000  # 词表大小
    hidden_size: int = 512   # 隐藏维度
    num_layers: int = 6      # 层数
    num_heads: int = 8       # 注意力头数
    intermediate_size: int = 2048  # FFN 中间维度
    max_position_embeddings: int = 2048  # 最大序列长度
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5


class RotaryEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        # 计算旋转频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成旋转位置编码"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)

        # 生成 cos 和 sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转一半维度"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # QKV 投影
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] 1=valid, 0=padding

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # QKV 投影
        qkv = self.qkv(x)  # [batch, seq_len, 3 * hidden_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 旋转位置编码
        cos, sin = self.rotary_emb(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 计算注意力
        # q,k,v 当前: [batch, num_heads, seq_len, head_dim]
        # 需要 k 转置为 [batch, num_heads, head_dim, seq_len] 以计算注意力
        k = k.transpose(-2, -1)  # [batch, num_heads, head_dim, seq_len]
        # scores: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k) / math.sqrt(self.head_dim)

        # 因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # 处理 padding 掩码
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1=valid, 0=padding
            if attention_mask.dim() == 2:
                # 创建 key_padding_mask: [batch, seq_len]
                # 扩展到 [batch, num_heads, seq_len, seq_len]
                key_padding_mask = (attention_mask == 0)  # [batch, seq_len]
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                # 扩展到所有注意力头
                key_padding_mask = key_padding_mask.repeat(1, self.num_heads, seq_len, 1)  # [batch, num_heads, seq_len, seq_len]
                scores = scores.masked_fill(key_padding_mask, float('-inf'))

        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        # v: [batch, num_heads, seq_len, head_dim]
        # attn_weights: [batch, num_heads, seq_len, seq_len]
        # output: [batch, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)
        output = output.reshape(batch_size, seq_len, hidden_dim)

        # 输出投影
        output = self.o_proj(output)

        return output


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN 结构
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, attention_mask)
        x = residual + x

        # Feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x

    def _forward_checkpoint(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """用于 gradient checkpointing 的前向传播"""
        def _forward(*args):
            x, attention_mask = args[0], args[1]
            return self.forward(x, attention_mask)

        return checkpoint(_forward, x, attention_mask, use_reentrant=False)


class MiniLLM(nn.Module):
    """
    65M 参数的 Mini LLM
    decoder-only 架构
    """

    def __init__(self, config: LLMConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        # Embedding 层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化
        self.post_init()

    def enable_gradient_checkpointing(self, enable: bool = True):
        """启用/禁用 gradient checkpointing"""
        self.gradient_checkpointing = enable

    def post_init(self):
        """初始化权重"""
        # Embedding
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        # LM Head (与 embedding 共享权重)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer 层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final LayerNorm
        hidden_states = self.ln_f(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits

    def forward_with_embeddings(self,
                                 embeddings: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        直接使用 pre-computed embeddings 进行前向传播
        跳过 embedding 层，直接输入到 Transformer 层

        这对于 VLM 很重要 - 允许在 embedding 层注入图像特征

        Args:
            embeddings: [batch, seq_len, hidden_size] pre-computed embeddings
            attention_mask: [batch, seq_len] 注意力掩码

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # 跳过 embed_tokens，直接使用提供的 embeddings
        hidden_states = embeddings

        # Transformer 层
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = layer._forward_checkpoint(hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)

        # Final LayerNorm
        hidden_states = self.ln_f(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits

    def freeze_layers(self, unfrozen_indices: Optional[List[int]] = None):
        """
        冻结/解冻层

        Args:
            unfrozen_indices: 解冻的层索引列表，包括:
                - None: 全部冻结
                - [0]: 只解冻 embedding
                - [-1]: 只解冻最后一层
                - [0, -1]: 解冻 embedding 和最后一层
        """
        # 默认全部冻结
        for param in self.parameters():
            param.requires_grad = False

        if unfrozen_indices is None:
            return

        # 解冻 embedding
        if 0 in unfrozen_indices:
            for param in self.embed_tokens.parameters():
                param.requires_grad = True

        # 解冻最后一层
        if -1 in unfrozen_indices:
            for param in self.layers[-1].parameters():
                param.requires_grad = True
            for param in self.ln_f.parameters():
                param.requires_grad = True
            for param in self.lm_head.parameters():
                param.requires_grad = True

    @property
    def num_parameters(self) -> int:
        """返回总参数量"""
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试 LLM
    print('Testing MiniLLM (65M)...')

    config = LLMConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
    )

    llm = MiniLLM(config)
    print(f'Total parameters: {llm.num_parameters:,}')
    print(f'Expected ~65M parameters')

    # 测试前向传播
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    logits = llm(input_ids, attention_mask)
    print(f'Input shape: {input_ids.shape}')
    print(f'Output shape: {logits.shape}')

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print('LLM test passed!')

    # 测试冻结
    llm.freeze_layers([0, -1])
    print(f'Trainable parameters after freeze: {llm.trainable_parameters:,}')
