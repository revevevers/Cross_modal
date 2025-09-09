"""LSTM基准模型 - 用于与算子映射模型对比"""

import torch
import torch.nn as nn

class LSTMMapper(nn.Module):
    """优化版LSTM映射器 - 注重训练速度和效率"""
    
    def __init__(self, signal_length=512, hidden_size=128, num_layers=2, 
                 feature_dim=256, dropout=0.1, bidirectional=True):
        super().__init__()
        self.signal_length = signal_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.feature_dim = feature_dim
        
        # LSTM层 - 处理时序信号
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 简化的特征提取层
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, feature_dim),
            nn.ReLU(),  # 使用ReLU替代GELU，计算更快
            nn.Dropout(dropout)
        )
        
        # 全局池化 - 将序列信息聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 简化的解码器：适配32x32输出
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 4*4*64),  # 更小的起始尺寸
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 32x32
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """高效的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, signals):
        """优化的前向传播"""
        # 确保输入在正确设备上
        signals = signals.to(next(self.parameters()).device)
        
        # 重塑为LSTM输入格式
        x = signals.unsqueeze(-1)  # [B, seq_len, 1]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_size*directions]
        
        # 特征提取
        features = self.feature_extractor(lstm_out)  # [B, seq_len, feature_dim]
        
        # 全局池化 - 将时序信息聚合为单一特征向量
        features = features.transpose(1, 2)  # [B, feature_dim, seq_len]
        pooled_features = self.global_pool(features).squeeze(-1)  # [B, feature_dim]
        
        # 解码生成图像
        out = self.decoder(pooled_features)  # [B, 1, 32, 32]
        
        return out.squeeze(1), pooled_features
    
    def get_model_complexity(self):
        """计算模型复杂度指标"""
        total_params = sum(p.numel() for p in self.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        return {
            'total_params': total_params,
            'lstm_params': lstm_params,
            'decoder_params': total_params - lstm_params
        }
    
    def get_device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device
