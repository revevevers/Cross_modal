"""LSTM模型配置"""

class LSTMConfig:
    """LSTM模型配置类 - 优化版本，注重训练速度"""
    
    # 数据配置 - 大幅减少数据量提高训练速度
    SIGNAL_LENGTH = 512      # 从1024减少到512，减少50%计算量
    IMAGE_SIZE = 32          # 从64减少到32，减少75%像素数
    N_SAMPLES_TRAIN = 1000   # 从默认减少到1000，快速验证
    N_SAMPLES_VAL = 200      # 相应减少验证集
    MAX_DEFECTS = 2          # 从3减少到2，简化问题
    NOISE_STD = 0.02
    
    # 模型配置 - 降低复杂度
    LSTM_HIDDEN_SIZE = 128   # 从256减少到128
    LSTM_NUM_LAYERS = 2      # 保持2层
    FEATURE_DIM = 256        # 从512减少到256
    LSTM_DROPOUT = 0.1
    BIDIRECTIONAL = True
    
    # 训练配置 - 优化训练效率
    BATCH_SIZE = 32          # 适中的batch size
    NUM_EPOCHS = 30          # 减少训练轮数
    LEARNING_RATE = 1e-3     # 稍微提高学习率加快收敛
    WEIGHT_DECAY = 1e-4
    
    # 可视化配置
    VISUALIZE_EVERY = 10
    NUM_VISUALIZE_SAMPLES = 3
