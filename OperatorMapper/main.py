"""主程序 - 集成所有模块并运行完整流程"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data_generator import ToyUSDataset, batch_spectrogram
from models import OperatorMapper
from training import Trainer

def plot_training_history(trainer):
    """绘制训练历史曲线"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = range(1, len(trainer.train_losses) + 1)
    plt.plot(epochs, trainer.train_losses, 'b-', label='Train')
    plt.plot(epochs, trainer.val_losses, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_model_info(model, sample_spec_shape):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== 模型信息 ===")
    print(f"频谱形状: {sample_spec_shape}")
    print(f"输入维度: {model.in_dim}")
    print(f"低秩维度: {model.rank}")
    print(f"潜在维度: {model.latent_dim}")
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

def main():
    """主函数"""
    print("=== 跨模态神经网络 - 谱系数算子映射 ===")
    
    # 配置
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数据集
    print("\n创建数据集...")
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
    
    # 确定频谱形状
    sample_sig, sample_img = train_dataset[0]
    sample_sig_t = torch.from_numpy(sample_sig).unsqueeze(0)
    with torch.no_grad():
        S = batch_spectrogram(sample_sig_t, n_fft=config.N_FFT, 
                            hop_length=config.HOP_LENGTH)
    spec_shape = (S.shape[1], S.shape[2])
    
    # 模型
    print("\n创建模型...")
    model = OperatorMapper(
        spec_shape=spec_shape,
        rank=config.RANK,
        latent_dim=config.LATENT_DIM
    )
    
    get_model_info(model, spec_shape)
    
    # 训练器
    trainer = Trainer(model, train_dataset, val_dataset, config, device)
    
    # 训练
    trained_model = trainer.train(config.NUM_EPOCHS)
    
    # 绘制训练历史
    plot_training_history(trainer)
    
    # 保存模型
    save_path = os.path.join(os.path.dirname(__file__), 'cross_modal_model.pth')
    torch.save(trained_model.state_dict(), save_path)
    print("\n模型已保存到 'cross_modal_model.pth'")
    
    return trained_model, trainer

if __name__ == "__main__":
    model, trainer = main()
