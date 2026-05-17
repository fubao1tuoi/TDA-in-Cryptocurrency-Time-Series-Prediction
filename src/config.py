import torch
import random
import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & REPRODUCIBILITY
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    """Khóa tính ngẫu nhiên để đảm bảo khả năng tái tạo kết quả (Reproducibility)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Đã set Seed: {seed} | Thiết bị: {DEVICE}")

# ==========================================
# 2. CẤU HÌNH DỮ LIỆU & ĐƯỜNG DẪN
# ==========================================
# Đường dẫn file dữ liệu (Nhớ đổi lại nếu mày lưu tên khác)
DATA_PATH = "btc_full.csv" 
TDA_CACHE_PATH = "tda_cache_final.pickle"

# ==========================================
# 3. CẤU HÌNH PIPELINE CHUNG (TIME SERIES)
# ==========================================
# Các mốc thời gian cần dự báo (Tính bằng Ngày)
HORIZONS = [1, 7, 14]

# Setup cho Walk-Forward Validation
N_SPLITS = 5
TEST_SIZE_PERCENT = 0.15

# ==========================================
# 4. CẤU HÌNH ĐẶC TRƯNG (FEATURE ENGINEERING)
# ==========================================
# Danh sách cột đưa vào Topo (TDA)
TDA_TARGET_COLS = ['return', 'quote_asset_volume', 'funding_rate']
TDA_WINDOW_SIZE = 30
TDA_DIMENSION = 4

# ==========================================
# 5. CẤU HÌNH CHI TIẾT THEO TỪNG HORIZON
# ==========================================
# Mày có thể tinh chỉnh riêng cho từng mốc để đạt RMSE thấp nhất
HORIZON_SETTINGS = {
    1: {
        "time_steps": 14,      # Dự báo 1 ngày: nhìn lại 2 tuần
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    },
    7: {
        "time_steps": 30,      # Dự báo 7 ngày: nhìn lại 1 tháng
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 120
    },
    14: {
        "time_steps": 90,      # Dự báo 14 ngày: nhìn lại 2 tháng
        "learning_rate": 0.0001,
        "batch_size": 64,
        "epochs": 120
    }
}

# Các tham số chung cho kiến trúc mạng
DL_HIDDEN_DIM = 32
DL_LAYER_DIM = 2
DL_DROPOUT = 0.2
DL_PATIENCE = 10