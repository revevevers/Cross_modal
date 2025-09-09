"""配置文件 - 包含所有超参数和配置选项"""

class Config:
    # 数据参数
    N_SAMPLES_TRAIN = 800
    N_SAMPLES_VAL = 200
    SIGNAL_LENGTH = 1024
    IMAGE_SIZE = 64
    MAX_DEFECTS = 3
    NOISE_STD = 0.03
    
    # 频谱参数
    N_FFT = 256
    HOP_LENGTH = 8
    
    # 模型参数
    RANK = 64
    LATENT_DIM = 1024
    
    # 训练参数
    NUM_EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    OPERATOR_REG_WEIGHT = 1e-6
    
    # 可视化参数
    VISUALIZE_EVERY = 10
    NUM_VISUALIZE_SAMPLES = 4
