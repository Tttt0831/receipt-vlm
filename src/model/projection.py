"""
MLP Projection 层
将视觉特征映射到 LLM 隐藏空间
2层 MLP + GELU (LLaVA 风格)
"""
import torch
import torch.nn as nn
from typing import Optional


class MLPProjection(nn.Module):
    """
    2层 MLP 投影层
    将视觉编码器的输出投影到 LLM 的隐藏空间

    结构: vision_dim -> intermediate_dim -> llm_hidden_dim
    """

    def __init__(self,
                 vision_dim: int,
                 llm_hidden_dim: int,
                 intermediate_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = llm_hidden_dim * 2

        self.vision_dim = vision_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.intermediate_dim = intermediate_dim

        # 2层 MLP
        self.fc1 = nn.Linear(vision_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, llm_hidden_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            vision_features: [batch, num_patches, vision_dim] 视觉特征

        Returns:
            [batch, num_patches, llm_hidden_dim] 投影后的特征
        """
        x = self.fc1(vision_features)  # [batch, num_patches, intermediate_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, num_patches, llm_hidden_dim]

        return x

    @property
    def num_parameters(self) -> int:
        """返回参数数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # 测试投影层
    print('Testing MLP Projection...')

    vision_dim = 768
    llm_hidden_dim = 512

    projection = MLPProjection(vision_dim, llm_hidden_dim)
    print(f'Projection parameters: {projection.num_parameters:,}')

    # 测试前向传播
    batch_size = 2
    num_patches = 576
    vision_features = torch.randn(batch_size, num_patches, vision_dim)

    output = projection(vision_features)
    print(f'Input shape: {vision_features.shape}')
    print(f'Output shape: {output.shape}')

    assert output.shape == (batch_size, num_patches, llm_hidden_dim)
    print('Projection test passed!')
