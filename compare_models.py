"""模型对比分析脚本 - 比较算子映射模型和LSTM模型"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data_generator import ToyUSDataset, batch_spectrogram
from config import Config
from models import OperatorMapper
from lstm_baseline.config_lstm import LSTMConfig
from lstm_baseline.models_lstm import LSTMMapper

def load_models_and_results():
    """加载训练好的模型和结果"""
    # 配置
    config = Config()
    lstm_config = LSTMConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载算子映射模型
    sample_sig = torch.randn(1, config.SIGNAL_LENGTH)
    with torch.no_grad():
        S = batch_spectrogram(sample_sig, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    spec_shape = (S.shape[1], S.shape[2])
    
    operator_model = OperatorMapper(
        spec_shape=spec_shape,
        rank=config.RANK,
        latent_dim=config.LATENT_DIM
    )
    
    # 加载LSTM模型
    lstm_model = LSTMMapper(
        signal_length=lstm_config.SIGNAL_LENGTH,
        hidden_size=lstm_config.LSTM_HIDDEN_SIZE,
        num_layers=lstm_config.LSTM_NUM_LAYERS,
        feature_dim=lstm_config.FEATURE_DIM,
        dropout=lstm_config.LSTM_DROPOUT,
        bidirectional=lstm_config.BIDIRECTIONAL
    )
    
    try:
        operator_model.load_state_dict(torch.load('cross_modal_model.pth', map_location=device))
        lstm_model.load_state_dict(torch.load('lstm_baseline/lstm_final_model.pth', map_location=device))
        print("模型加载成功!")
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}")
        return None, None, None, None
    
    # 加载训练结果
    try:
        lstm_results = torch.load('lstm_baseline/lstm_training_results.pth', map_location=device)
    except FileNotFoundError:
        lstm_results = None
        print("LSTM训练结果未找到")
    
    return operator_model, lstm_model, config, lstm_results

def compare_model_complexity():
    """比较模型复杂度"""
    operator_model, lstm_model, config, _ = load_models_and_results()
    if operator_model is None:
        return
    
    # 算子映射模型参数
    op_params = sum(p.numel() for p in operator_model.parameters())
    op_rank = operator_model.rank
    op_in_dim = operator_model.in_dim
    op_latent_dim = operator_model.latent_dim
    
    # LSTM模型参数
    lstm_complexity = lstm_model.get_model_complexity()
    
    print("=== 模型复杂度对比 ===")
    print(f"算子映射模型:")
    print(f"  总参数数: {op_params:,}")
    print(f"  低秩维度: {op_rank}")
    print(f"  输入维度: {op_in_dim}")
    print(f"  潜在维度: {op_latent_dim}")
    print(f"  参数效率: {op_params / (op_in_dim * op_latent_dim):.3f}")
    
    print(f"\nLSTM模型:")
    print(f"  总参数数: {lstm_complexity['total_params']:,}")
    print(f"  LSTM参数数: {lstm_complexity['lstm_params']:,}")
    print(f"  解码器参数数: {lstm_complexity['decoder_params']:,}")
    
    # 可视化参数对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 总参数对比
    models = ['Operator Mapping', 'LSTM']
    params = [op_params, lstm_complexity['total_params']]
    colors = ['skyblue', 'lightcoral']
    
    bars1 = ax1.bar(models, params, color=colors)
    ax1.set_ylabel('参数数量')
    ax1.set_title('模型参数总数对比')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 添加数值标签
    for bar, param in zip(bars1, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom')
    
    # 参数组成对比
    op_components = ['Low-rank A', 'Low-rank B', 'Decoder']
    op_values = [
        operator_model.A.numel(),
        operator_model.B.numel(),
        sum(p.numel() for p in operator_model.decoder.parameters())
    ]
    
    lstm_components = ['LSTM', 'Feature Extractor', 'Decoder']
    lstm_values = [
        lstm_complexity['lstm_params'],
        lstm_complexity['decoder_params'] - sum(p.numel() for p in lstm_model.decoder.parameters()),
        sum(p.numel() for p in lstm_model.decoder.parameters())
    ]
    
    x = np.arange(len(op_components))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, op_values, width, label='Operator', color='skyblue')
    bars3 = ax2.bar(x + width/2, lstm_values, width, label='LSTM', color='lightcoral')
    
    ax2.set_ylabel('参数数量')
    ax2.set_title('模型组件参数对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(op_components)
    ax2.legend()
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def compare_predictions():
    """比较两个模型的预测效果"""
    operator_model, lstm_model, config, _ = load_models_and_results()
    if operator_model is None:
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    operator_model.to(device).eval()
    lstm_model.to(device).eval()
    
    # 创建测试数据
    test_dataset = ToyUSDataset(
        n_samples=8,
        sig_len=config.SIGNAL_LENGTH,
        img_size=config.IMAGE_SIZE,
        max_defects=config.MAX_DEFECTS,
        noise_std=config.NOISE_STD
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    sample_batch = next(iter(test_loader))
    sigs, imgs = sample_batch
    
    with torch.no_grad():
        # 算子映射模型预测
        spec = batch_spectrogram(sigs, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        op_preds, _ = operator_model(spec.to(device))
        op_preds = torch.sigmoid(op_preds).cpu().numpy()
        
        # LSTM模型预测
        lstm_preds, _ = lstm_model(sigs.to(device))
        lstm_preds = torch.sigmoid(lstm_preds).cpu().numpy()
    
    # 可视化对比
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    
    for i in range(4):
        # 原始信号
        axes[i, 0].plot(sigs[i].numpy())
        axes[i, 0].set_title(f'Sample {i+1}: 1D Signal')
        axes[i, 0].grid(True)
        
        # 真实标签
        im1 = axes[i, 1].imshow(imgs[i], cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # 算子映射预测
        im2 = axes[i, 2].imshow(op_preds[i], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Operator Mapping')
        axes[i, 2].axis('off')
        
        # LSTM预测
        im3 = axes[i, 3].imshow(lstm_preds[i], cmap='hot', vmin=0, vmax=1)
        axes[i, 3].set_title('LSTM')
        axes[i, 3].axis('off')
    
    plt.suptitle('模型预测效果对比', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 计算量化指标
    calculate_metrics_comparison(imgs.numpy(), op_preds, lstm_preds)

def calculate_metrics_comparison(gt, op_preds, lstm_preds):
    """计算并比较评估指标"""
    def dice_score(pred, gt, threshold=0.5):
        pred_binary = (pred > threshold).astype(float)
        intersection = (pred_binary * gt).sum()
        union = pred_binary.sum() + gt.sum()
        return (2.0 * intersection) / (union + 1e-8)
    
    def mse_score(pred, gt):
        return ((pred - gt) ** 2).mean()
    
    # 计算指标
    op_dice_scores = [dice_score(op_preds[i], gt[i]) for i in range(len(gt))]
    lstm_dice_scores = [dice_score(lstm_preds[i], gt[i]) for i in range(len(gt))]
    
    op_mse_scores = [mse_score(op_preds[i], gt[i]) for i in range(len(gt))]
    lstm_mse_scores = [mse_score(lstm_preds[i], gt[i]) for i in range(len(gt))]
    
    print("\n=== 预测性能对比 ===")
    print(f"Dice Score:")
    print(f"  算子映射: {np.mean(op_dice_scores):.4f} ± {np.std(op_dice_scores):.4f}")
    print(f"  LSTM:     {np.mean(lstm_dice_scores):.4f} ± {np.std(lstm_dice_scores):.4f}")
    
    print(f"\nMSE:")
    print(f"  算子映射: {np.mean(op_mse_scores):.6f} ± {np.std(op_mse_scores):.6f}")
    print(f"  LSTM:     {np.mean(lstm_mse_scores):.6f} ± {np.std(lstm_mse_scores):.6f}")
    
    # 可视化指标对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice分数对比
    dice_data = [op_dice_scores, lstm_dice_scores]
    ax1.boxplot(dice_data, labels=['Operator Mapping', 'LSTM'])
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # MSE对比
    mse_data = [op_mse_scores, lstm_mse_scores]
    ax2.boxplot(mse_data, labels=['Operator Mapping', 'LSTM'])
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主对比函数"""
    print("=== 模型对比分析 ===")
    
    # 模型复杂度对比
    compare_model_complexity()
    
    # 预测效果对比
    compare_predictions()

if __name__ == "__main__":
    main()
