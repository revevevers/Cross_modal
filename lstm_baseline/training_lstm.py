"""LSTM模型的训练模块"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class LSTMTrainer:
    """优化版LSTM训练器 - 注重训练速度"""
    
    def __init__(self, model, train_dataset, val_dataset, config, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 优化的数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2,  # 减少worker数量避免内存问题
            pin_memory=True if device == 'cuda' else False
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE * 2,  # 验证时可以用更大batch
            shuffle=False,
            num_workers=1,
            pin_memory=True if device == 'cuda' else False
        )
        
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        # 简化的损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 更激进的学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self):
        """优化的训练epoch"""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            sigs_np, imgs_np = batch
            sigs = sigs_np.to(self.device, dtype=torch.float32, non_blocking=True)
            imgs = imgs_np.to(self.device, dtype=torch.float32, non_blocking=True)
            
            # 前向传播
            preds_logits, _ = self.model(sigs)
            
            # 计算损失
            loss = self.criterion(preds_logits, imgs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 轻量级梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # 减少打印频率
            if batch_idx % (max(1, num_batches // 2)) == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return running_loss / len(self.train_loader)

    def validate(self):
        """快速验证"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sigs_np, imgs_np = batch
                sigs = sigs_np.to(self.device, dtype=torch.float32, non_blocking=True)
                imgs = imgs_np.to(self.device, dtype=torch.float32, non_blocking=True)
                
                preds_logits, _ = self.model(sigs)
                loss = self.criterion(preds_logits, imgs)
                
                val_loss += loss.item() * sigs.shape[0]
        
        return val_loss / len(self.val_loader.dataset)

    def calculate_metrics(self):
        """计算评估指标"""
        self.model.eval()
        dice_scores = []
        mse_scores = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                sigs_np, imgs_np = batch
                sigs = sigs_np.to(self.device, dtype=torch.float32)
                imgs = imgs_np.to(self.device, dtype=torch.float32)
                
                preds_logits, _ = self.model(sigs)
                preds = torch.sigmoid(preds_logits)
                
                # 计算Dice系数
                preds_binary = (preds > 0.5).float()
                intersection = (preds_binary * imgs).sum(dim=[1,2])
                union = preds_binary.sum(dim=[1,2]) + imgs.sum(dim=[1,2])
                dice = (2.0 * intersection) / (union + 1e-8)
                dice_scores.extend(dice.cpu().numpy())
                
                # 计算MSE
                mse = ((preds - imgs)**2).mean(dim=[1,2])
                mse_scores.extend(mse.cpu().numpy())
        
        return {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores)
        }

    def visualize_predictions(self, epoch):
        """可视化预测结果"""
        self.model.eval()
        sample_batch = next(iter(self.train_loader))
        sample_sigs, sample_imgs = sample_batch
        sample_sigs = sample_sigs[:self.config.NUM_VISUALIZE_SAMPLES]
        sample_imgs = sample_imgs[:self.config.NUM_VISUALIZE_SAMPLES]
        
        with torch.no_grad():
            sample_sigs_gpu = sample_sigs.to(self.device, dtype=torch.float32)
            preds_logits, _ = self.model(sample_sigs_gpu)
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
            axs[i,2].set_title("LSTM Prediction")
            axs[i,2].axis('off')
        
        plt.suptitle(f'LSTM Model - Epoch {epoch}')
        plt.tight_layout()
        plt.show()

    def train(self, num_epochs):
        """优化的训练流程"""
        print("开始快速LSTM模型训练...")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"图像尺寸: {self.config.IMAGE_SIZE}x{self.config.IMAGE_SIZE}")
        
        # 模型复杂度信息
        complexity = self.model.get_model_complexity()
        print(f"模型参数总数: {complexity['total_params']:,}")
        
        best_val_loss = float('inf')
        patience = 10  # 增加耐心值
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证（每2个epoch验证一次以节省时间）
            if epoch % 2 == 0 or epoch == 1:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
            else:
                val_loss = self.val_losses[-1] if self.val_losses else train_loss
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            print(f"Epoch {epoch:02d} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # 早停检查
            if epoch % 2 == 0:  # 只在验证的epoch检查早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'lstm_best_model.pth')
                    print(f"  保存最佳模型，验证损失: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停：连续{patience}次验证损失未改善")
                        break
            
            # 减少可视化频率
            if epoch % (self.config.VISUALIZE_EVERY * 2) == 0 or epoch == 1:
                self.visualize_predictions(epoch)
        
        # 计算最终指标
        final_metrics = self.calculate_metrics()
        print(f"\n最终评估指标:")
        print(f"Dice系数: {final_metrics['dice_mean']:.4f} ± {final_metrics['dice_std']:.4f}")
        print(f"MSE: {final_metrics['mse_mean']:.6f} ± {final_metrics['mse_std']:.6f}")
        
        print("快速LSTM训练完成!")
        return self.model, final_metrics
