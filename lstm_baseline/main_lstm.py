"""LSTM基准模型主程序"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import ToyUSDataset
from config_lstm import LSTMConfig
from models_lstm import LSTMMapper
from training_lstm import LSTMTrainer

def plot_training_comparison(lstm_trainer):
    """绘制LSTM训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0,0].plot(lstm_trainer.train_losses, label='Train Loss', color='blue')
    axes[0,0].plot(lstm_trainer.val_losses, label='Val Loss', color='red')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('LSTM Training History')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 学习率变化
    axes[0,1].plot(lstm_trainer.learning_rates, color='green')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Learning Rate')
    axes[0,1].set_title('Learning Rate Schedule')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True)
    
    # 训练收敛性分析
    if len(lstm_trainer.train_losses) > 10:
        # 计算损失的移动平均
        window = 5
        train_ma = []
        val_ma = []
        for i in range(window-1, len(lstm_trainer.train_losses)):
            train_ma.append(sum(lstm_trainer.train_losses[i-window+1:i+1])/window)
            val_ma.append(sum(lstm_trainer.val_losses[i-window+1:i+1])/window)
        
        axes[1,0].plot(range(window-1, len(lstm_trainer.train_losses)), train_ma, 
                      label='Train MA', color='blue')
        axes[1,0].plot(range(window-1, len(lstm_trainer.val_losses)), val_ma, 
                      label='Val MA', color='red')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss (Moving Average)')
        axes[1,0].set_title('Smoothed Loss Curves')
        axes[1,0].legend()
        axes[1,0].grid(True)
    
    # 过拟合分析
    overfitting_gap = [v - t for t, v in zip(lstm_trainer.train_losses, lstm_trainer.val_losses)]
    axes[1,1].plot(overfitting_gap, color='purple')
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Val Loss - Train Loss')
    axes[1,1].set_title('Overfitting Analysis')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def get_model_info(model, config):
    """获取LSTM模型信息"""
    complexity = model.get_model_complexity()
    
    print("\n=== LSTM模型信息 ===")
    print(f"信号长度: {config.SIGNAL_LENGTH}")
    print(f"LSTM隐藏层大小: {config.LSTM_HIDDEN_SIZE}")
    print(f"LSTM层数: {config.LSTM_NUM_LAYERS}")
    print(f"双向LSTM: {config.BIDIRECTIONAL}")
    print(f"特征维度: {config.FEATURE_DIM}")
    print(f"总参数数: {complexity['total_params']:,}")
    print(f"LSTM参数数: {complexity['lstm_params']:,}")
    print(f"解码器参数数: {complexity['decoder_params']:,}")

def main():
    """主函数"""
    print("=== 快速LSTM基准模型训练 ===")
    
    # 配置
    config = LSTMConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # GPU信息
    if device == 'cuda':
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # 数据集 - 使用优化的参数
    print("\n创建优化的数据集...")
    print(f"训练样本: {config.N_SAMPLES_TRAIN}, 验证样本: {config.N_SAMPLES_VAL}")
    print(f"信号长度: {config.SIGNAL_LENGTH}, 图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    
    train_dataset = ToyUSDataset(
        n_samples=config.N_SAMPLES_TRAIN,
        sig_len=config.SIGNAL_LENGTH,
        img_size=config.IMAGE_SIZE,
        max_defects=config.MAX_DEFECTS,
        noise_std=config.NOISE_STD
    )
    
    val_dataset = ToyUSDataset(
        n_samples=config.N_SAMPLES_VAL,
        sig_len=config.SIGNAL_LENGTH,
        img_size=config.IMAGE_SIZE,
        max_defects=config.MAX_DEFECTS,
        noise_std=config.NOISE_STD
    )
    
    # 模型
    print("\n创建优化的LSTM模型...")
    model = LSTMMapper(
        signal_length=config.SIGNAL_LENGTH,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        feature_dim=config.FEATURE_DIM,
        dropout=config.LSTM_DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    ).to(device)
    
    get_model_info(model, config)
    
    # 检查模型是否在GPU上
    if device == 'cuda':
        print(f"模型设备: {next(model.parameters()).device}")
    
    # 训练器
    trainer = LSTMTrainer(model, train_dataset, val_dataset, config, device)
    
    # 训练
    trained_model, final_metrics = trainer.train(config.NUM_EPOCHS)
    
    # 绘制训练历史
    plot_training_comparison(trainer)
    
    # 保存模型和结果
    torch.save(trained_model.state_dict(), 'lstm_final_model.pth')
    
    # 保存训练结果
    results = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'learning_rates': trainer.learning_rates,
        'final_metrics': final_metrics,
        'model_complexity': model.get_model_complexity()
    }
    
    torch.save(results, 'lstm_training_results.pth')
    print(f"\n模型和结果已保存")
    print(f"最佳模型: lstm_best_model.pth")
    print(f"最终模型: lstm_final_model.pth")
    print(f"训练结果: lstm_training_results.pth")
    
    return trained_model, trainer, final_metrics

if __name__ == "__main__":
    model, trainer, metrics = main()
