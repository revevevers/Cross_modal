"""数据生成模块 - 生成玩具超声数据和对应的缺陷图"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset

class ToyUSDataset(Dataset):
    """玩具超声数据集 - 生成1D超声信号和对应的2D缺陷图"""
    
    def __init__(self, n_samples=1000, sig_len=1024, img_size=64, 
                 max_defects=3, noise_std=0.02):
        self.n = n_samples
        self.sig_len = sig_len
        self.img_size = img_size
        self.max_defects = max_defects
        self.noise_std = noise_std
        self.samples = [self._gen_sample() for _ in range(n_samples)]

    def _gen_sample(self):
        """生成单个样本：1D信号和2D缺陷图"""
        # 随机生成缺陷数量和位置
        k = random.randint(1, self.max_defects)
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        defect_positions = []
        
        # 生成每个缺陷
        for _ in range(k):
            x = random.uniform(0.1, 0.9)  # 归一化坐标
            y = random.uniform(0.1, 0.9)
            px = int(x * (self.img_size - 1))
            py = int(y * (self.img_size - 1))
            defect_positions.append((px, py))
            
            # 在缺陷位置添加高斯斑点
            sigma = random.uniform(1.5, 4.0)
            xv, yv = np.meshgrid(np.arange(self.img_size), 
                               np.arange(self.img_size), indexing='xy')
            g = np.exp(-((xv-px)**2 + (yv-py)**2) / (2 * sigma**2))
            img += g.astype(np.float32)
        
        # 归一化图像到[0,1]
        img = img / (img.max() + 1e-8)

        # 根据缺陷位置生成1D超声信号
        t = np.linspace(0, 1, self.sig_len, dtype=np.float32)
        sig = np.zeros_like(t)
        
        for (px, py) in defect_positions:
            # 将2D位置映射为1D延迟
            delay = (px + py) / (2 * (self.img_size - 1))
            center = int(delay * (self.sig_len - 1))
            amp = 0.8 + 0.4 * random.random()
            
            # 生成高斯调制的正弦脉冲
            width = int(6 + 6 * random.random())
            idx = np.arange(max(center-4*width,0), 
                          min(center+4*width, self.sig_len))
            pulse = (amp * np.exp(-((idx-center)**2) / (2 * (width**2))) * 
                    np.sin(2*np.pi*30*(idx-center)/self.sig_len))
            sig[idx] += pulse
        
        # 添加噪声
        sig += np.random.normal(scale=self.noise_std, size=sig.shape).astype(np.float32)
        
        # 归一化信号
        smax = np.max(np.abs(sig)) + 1e-8
        sig = sig / smax
        
        return sig.astype(np.float32), img.astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx]

def batch_spectrogram(signals, n_fft=256, hop_length=8):
    """批量计算频谱图"""
    if isinstance(signals, np.ndarray):
        signals = torch.from_numpy(signals)
    
    window = torch.hann_window(n_fft, device=signals.device)
    spec = torch.stft(signals, n_fft=n_fft, hop_length=hop_length, 
                      window=window, return_complex=True, 
                      normalized=False, onesided=True)
    
    # 取幅值并进行对数压缩
    mag = spec.abs()
    mag = torch.log1p(mag)
    return mag
