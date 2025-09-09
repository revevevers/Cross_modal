"""训练模块 - 包含训练循环和验证逻辑"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_generator import batch_spectrogram

class Trainer:
    """训练器类 - 封装训练和验证逻辑"""
    
    def __init__(self, model, train_dataset, val_dataset, config, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch in self.train_loader:
            sigs_np, imgs_np = batch
            sigs = sigs_np.to(self.device)
            imgs = imgs_np.to(self.device)
            
            # 计算频谱图
            spec = batch_spectrogram(
                sigs, n_fft=self.config.N_FFT, 
                hop_length=self.config.HOP_LENGTH
            ).to(self.device)
            
            # 前向传播
            preds_logits, latent = self.model(spec)
            
            # 计算损失 - 修复维度不匹配问题
            recon_loss = self.criterion(
                preds_logits, imgs.to(preds_logits.dtype)
            )
            op_reg = self.model.get_operator_regularization()
            loss = recon_loss + self.config.OPERATOR_REG_WEIGHT * op_reg
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * sigs.shape[0]
        
        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sigs_np, imgs_np = batch
                sigs = sigs_np.to(self.device)
                imgs = imgs_np.to(self.device)
                
                spec = batch_spectrogram(
                    sigs, n_fft=self.config.N_FFT,
                    hop_length=self.config.HOP_LENGTH
                ).to(self.device)
                
                preds_logits, _ = self.model(spec)
                
                recon_loss = self.criterion(
                    preds_logits, imgs.to(preds_logits.dtype)
                )
                op_reg = self.model.get_operator_regularization()
                loss = recon_loss + self.config.OPERATOR_REG_WEIGHT * op_reg
                
                val_loss += loss.item() * sigs.shape[0]
        
        return val_loss / len(self.val_loader.dataset)

    def visualize_predictions(self, epoch):
        """可视化预测结果"""
        self.model.eval()
        sample_batch = next(iter(self.train_loader))
        sample_sigs, sample_imgs = sample_batch
        sample_sigs = sample_sigs[:self.config.NUM_VISUALIZE_SAMPLES]
        sample_imgs = sample_imgs[:self.config.NUM_VISUALIZE_SAMPLES]
        
        with torch.no_grad():
            spec = batch_spectrogram(
                sample_sigs.to(self.device),
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH
            ).to(self.device)
            
            preds_logits, _ = self.model(spec)
            preds = torch.sigmoid(preds_logits).cpu().numpy()
        
        fig, axs = plt.subplots(
            self.config.NUM_VISUALIZE_SAMPLES, 3, figsize=(12, 8)
        )
        
        for i in range(self.config.NUM_VISUALIZE_SAMPLES):
            # 1D信号
            axs[i,0].plot(sample_sigs[i].numpy())
            axs[i,0].set_title("1D Signal")
            axs[i,0].grid(True)
            
            # 真实缺陷图
            axs[i,1].imshow(sample_imgs[i], cmap='hot')
            axs[i,1].set_title("Ground Truth")
            axs[i,1].axis('off')
            
            # 预测缺陷图
            axs[i,2].imshow(preds[i], cmap='hot', vmin=0, vmax=1)
            axs[i,2].set_title("Prediction")
            axs[i,2].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Predictions vs Ground Truth')
        plt.tight_layout()
        plt.show()

    def train(self, num_epochs):
        """完整训练流程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch:02d} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # 可视化
            if epoch % self.config.VISUALIZE_EVERY == 0 or epoch == 1:
                self.visualize_predictions(epoch)
        
        print("训练完成!")
        return self.model
