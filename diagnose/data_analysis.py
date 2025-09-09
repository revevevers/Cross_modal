"""数据分布诊断工具"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import ToyUSDataset
from lstm_baseline.config_lstm import LSTMConfig

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, config):
        self.config = config
        
        # 创建数据集
        self.train_dataset = ToyUSDataset(
            n_samples=min(500, config.N_SAMPLES_TRAIN),  # 限制样本数量用于分析
            sig_len=config.SIGNAL_LENGTH,
            img_size=config.IMAGE_SIZE,
            max_defects=config.MAX_DEFECTS,
            noise_std=config.NOISE_STD
        )
        
        self.val_dataset = ToyUSDataset(
            n_samples=min(100, config.N_SAMPLES_VAL),
            sig_len=config.SIGNAL_LENGTH,
            img_size=config.IMAGE_SIZE,
            max_defects=config.MAX_DEFECTS,
            noise_std=config.NOISE_STD
        )
        
        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=50, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=50, shuffle=False)
        
    def analyze_signal_distribution(self):
        """分析信号分布"""
        print("=" * 50)
        print("Signal Distribution Analysis")
        print("=" * 50)
        
        signals = []
        for batch_idx, (sigs, _) in enumerate(self.train_loader):
            signals.append(sigs.numpy())
            if batch_idx >= 5:  # 只分析前几个batch
                break
                
        signals = np.concatenate(signals, axis=0)
        
        # 基本统计信息
        print(f"Signal shape: {signals.shape}")
        print(f"Signal mean: {signals.mean():.6f}")
        print(f"Signal std: {signals.std():.6f}")
        print(f"Signal min: {signals.min():.6f}")
        print(f"Signal max: {signals.max():.6f}")
        print(f"Signal median: {np.median(signals):.6f}")
        
        # 可视化信号分布
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 整体分布直方图
        axes[0,0].hist(signals.flatten(), bins=100, alpha=0.7, density=True)
        axes[0,0].set_title('Signal Value Distribution')
        axes[0,0].set_xlabel('Signal Value')
        axes[0,0].set_ylabel('Density')
        axes[0,0].grid(True)
        
        # 几个示例信号
        for i in range(5):
            axes[0,1].plot(signals[i], alpha=0.7, label=f'Signal {i+1}')
        axes[0,1].set_title('Example Signal Waveforms')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Amplitude')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 信号统计量分布
        signal_means = signals.mean(axis=1)
        signal_stds = signals.std(axis=1)
        
        axes[0,2].hist(signal_means, bins=30, alpha=0.7)
        axes[0,2].set_title('Signal Mean Distribution')
        axes[0,2].set_xlabel('Mean Value')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True)
        
        axes[1,0].hist(signal_stds, bins=30, alpha=0.7)
        axes[1,0].set_title('Signal Std Distribution')
        axes[1,0].set_xlabel('Standard Deviation')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True)
        
        # 频谱分析
        signal_fft = np.fft.fft(signals[0])
        freqs = np.fft.fftfreq(len(signal_fft))
        axes[1,1].plot(freqs[:len(freqs)//2], np.abs(signal_fft[:len(freqs)//2]))
        axes[1,1].set_title('Example Signal Spectrum')
        axes[1,1].set_xlabel('Frequency')
        axes[1,1].set_ylabel('Magnitude')
        axes[1,1].grid(True)
        
        # 自相关分析
        autocorr = np.correlate(signals[0], signals[0], mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        axes[1,2].plot(autocorr[:100])
        axes[1,2].set_title('Example Signal Autocorrelation')
        axes[1,2].set_xlabel('Lag')
        axes[1,2].set_ylabel('Correlation Coefficient')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig('diagnose/signal_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return signals
        
    def analyze_image_distribution(self):
        """分析图像分布"""
        print("\n" + "=" * 50)
        print("Image Distribution Analysis")
        print("=" * 50)
        
        images = []
        for batch_idx, (_, imgs) in enumerate(self.train_loader):
            images.append(imgs.numpy())
            if batch_idx >= 5:
                break
                
        images = np.concatenate(images, axis=0)
        
        # 基本统计信息
        print(f"Image shape: {images.shape}")
        print(f"Image mean: {images.mean():.6f}")
        print(f"Image std: {images.std():.6f}")
        print(f"Image min: {images.min():.6f}")
        print(f"Image max: {images.max():.6f}")
        
        # 缺陷统计
        defect_pixels = (images > 0.5).sum(axis=(1,2))
        non_zero_images = (defect_pixels > 0).sum()
        
        print(f"Images with defects: {non_zero_images} / {len(images)}")
        print(f"Average defect pixels: {defect_pixels.mean():.2f}")
        print(f"Defect pixel ratio: {defect_pixels.mean() / (images.shape[1] * images.shape[2]) * 100:.2f}%")
        
        # 可视化图像分布
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 像素值分布
        axes[0,0].hist(images.flatten(), bins=100, alpha=0.7)
        axes[0,0].set_title('Pixel Value Distribution')
        axes[0,0].set_xlabel('Pixel Value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True)
        
        # 缺陷像素数分布
        axes[0,1].hist(defect_pixels, bins=30, alpha=0.7)
        axes[0,1].set_title('Defect Pixels Per Image')
        axes[0,1].set_xlabel('Number of Defect Pixels')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True)
        
        # 缺陷位置热力图
        defect_heatmap = (images > 0.5).mean(axis=0)
        im = axes[0,2].imshow(defect_heatmap, cmap='hot')
        axes[0,2].set_title('Defect Location Heatmap')
        plt.colorbar(im, ax=axes[0,2])
        
        # 图像均值分布
        image_means = images.mean(axis=(1,2))
        axes[0,3].hist(image_means, bins=30, alpha=0.7)
        axes[0,3].set_title('Image Mean Distribution')
        axes[0,3].set_xlabel('Mean Value')
        axes[0,3].set_ylabel('Frequency')
        axes[0,3].grid(True)
        
        # 示例图像
        for i in range(8):
            row = (i // 4) + 1
            col = i % 4
            axes[row, col].imshow(images[i], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f'Sample {i+1} (Defect pixels: {defect_pixels[i]})')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('diagnose/image_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return images
        
    def analyze_signal_image_correlation(self):
        """分析信号与图像的相关性"""
        print("\n" + "=" * 50)
        print("Signal-Image Correlation Analysis")
        print("=" * 50)
        
        signals = []
        images = []
        
        for batch_idx, (sigs, imgs) in enumerate(self.train_loader):
            signals.append(sigs.numpy())
            images.append(imgs.numpy())
            if batch_idx >= 3:
                break
                
        signals = np.concatenate(signals, axis=0)
        images = np.concatenate(images, axis=0)
        
        # 计算信号特征
        signal_features = {
            'mean': signals.mean(axis=1),
            'std': signals.std(axis=1),
            'max': signals.max(axis=1),
            'min': signals.min(axis=1),
            'energy': (signals ** 2).sum(axis=1),
            'peak_count': np.array([len(self._find_peaks(sig)) for sig in signals])
        }
        
        # 计算图像特征
        image_features = {
            'defect_pixels': (images > 0.5).sum(axis=(1,2)),
            'defect_area': (images > 0.1).sum(axis=(1,2)),
            'max_intensity': images.max(axis=(1,2)),
            'mean_intensity': images.mean(axis=(1,2)),
            'center_x': np.array([self._get_center_of_mass(img)[1] for img in images]),
            'center_y': np.array([self._get_center_of_mass(img)[0] for img in images])
        }
        
        # 计算相关系数
        correlations = {}
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for sig_feat_name, sig_feat in signal_features.items():
            for img_feat_name, img_feat in image_features.items():
                # 过滤掉无效值
                valid_mask = ~(np.isnan(sig_feat) | np.isnan(img_feat))
                if valid_mask.sum() > 10:
                    corr = np.corrcoef(sig_feat[valid_mask], img_feat[valid_mask])[0, 1]
                    correlations[f"{sig_feat_name}_vs_{img_feat_name}"] = corr
                    
                    # 绘制前6个最高相关性的散点图
                    if plot_idx < 6:
                        axes[plot_idx].scatter(sig_feat[valid_mask], img_feat[valid_mask], alpha=0.6)
                        axes[plot_idx].set_xlabel(f'Signal {sig_feat_name}')
                        axes[plot_idx].set_ylabel(f'Image {img_feat_name}')
                        axes[plot_idx].set_title(f'Correlation: {corr:.3f}')
                        axes[plot_idx].grid(True)
                        plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('diagnose/signal_image_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印相关系数排序
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nStrongest correlations (sorted by absolute value):")
        for name, corr in sorted_corr[:10]:
            print(f"{name}: {corr:.4f}")
            
        return correlations
        
    def _find_peaks(self, signal, threshold=0.1):
        """简单的峰值检测"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
                peaks.append(i)
        return peaks
        
    def _get_center_of_mass(self, image):
        """计算缺陷质心"""
        binary_img = image > 0.5
        if binary_img.sum() == 0:
            return (np.nan, np.nan)
        
        y_indices, x_indices = np.where(binary_img)
        center_y = y_indices.mean()
        center_x = x_indices.mean()
        return (center_y, center_x)
        
    def generate_report(self):
        """生成完整的数据分析报告"""
        print("Starting data distribution diagnosis...")
        
        # 创建保存目录
        os.makedirs('diagnose', exist_ok=True)
        
        # 分析各个方面
        signals = self.analyze_signal_distribution()
        images = self.analyze_image_distribution()
        correlations = self.analyze_signal_image_correlation()
        
        # 生成文本报告
        with open('diagnose/data_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("Data Distribution Diagnosis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"Signal length: {self.config.SIGNAL_LENGTH}\n")
            f.write(f"Image size: {self.config.IMAGE_SIZE}x{self.config.IMAGE_SIZE}\n")
            f.write(f"Max defects: {self.config.MAX_DEFECTS}\n")
            f.write(f"Noise std: {self.config.NOISE_STD}\n\n")
            
            f.write("Signal Statistics:\n")
            f.write(f"Mean: {signals.mean():.6f}\n")
            f.write(f"Std: {signals.std():.6f}\n")
            f.write(f"Range: [{signals.min():.6f}, {signals.max():.6f}]\n\n")
            
            f.write("Image Statistics:\n")
            f.write(f"Mean: {images.mean():.6f}\n")
            f.write(f"Std: {images.std():.6f}\n")
            f.write(f"Defect pixel ratio: {((images > 0.5).sum() / images.size * 100):.2f}%\n\n")
            
            f.write("Main Correlations:\n")
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, corr in sorted_corr[:5]:
                f.write(f"{name}: {corr:.4f}\n")
        
        print(f"\nDiagnosis completed! Results saved in diagnose/ directory")
        print(f"- Signal distribution: diagnose/signal_distribution.png")
        print(f"- Image distribution: diagnose/image_distribution.png") 
        print(f"- Correlation analysis: diagnose/signal_image_correlation.png")
        print(f"- Text report: diagnose/data_analysis_report.txt")

def main():
    """主函数"""
    config = LSTMConfig()
    analyzer = DataAnalyzer(config)
    analyzer.generate_report()

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
