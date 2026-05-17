import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from src.augmentation import TimeSeriesAugmentDataset

def create_sliding_window(X, y, time_steps):
    """
    Cắt dữ liệu 2D thành các khối 3D (Batch, Time_Steps, Features)
    """
    X_seq, y_seq = [], []
    
    # Đảm bảo đầu vào là Numpy Array để cắt cho lẹ
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y
    
    for i in range(len(X_arr) - time_steps):
        X_seq.append(X_arr[i : i + time_steps])
        y_seq.append(y_arr[i + time_steps])
        
    return np.array(X_seq), np.array(y_seq)

def prepare_dataloaders(X_train_raw, y_train_raw, X_val_raw, y_val_raw, 
                        time_steps, batch_size, noise_level=0.0):
    """
    Quy trình một chạm: Scale -> Window -> Tensor -> Augment -> DataLoader
    """
    # 1. SCALE DỮ LIỆU (Chống Leakage: Chỉ fit trên Train, transform trên Val)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    
    # 2. TẠO SLIDING WINDOWS (Cắt thành 3D)
    X_train_seq, y_train_seq = create_sliding_window(X_train_scaled, y_train_raw, time_steps)
    X_val_seq, y_val_seq = create_sliding_window(X_val_scaled, y_val_raw, time_steps)
    
    # 3. CHUYỂN SANG TENSOR
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
    
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1)
    
    # 4. TẠO DATASET VỚI AUGMENTATION (On-the-fly)
    # Train bơm nhiễu (is_training=True)
    train_dataset = TimeSeriesAugmentDataset(X_train_tensor, y_train_tensor, 
                                             noise_level=noise_level, is_training=True)
    # Val ko bơm nhiễu (is_training=False)
    val_dataset = TimeSeriesAugmentDataset(X_val_tensor, y_val_tensor, 
                                           noise_level=0.0, is_training=False)
    
    # 5. TẠO DATALOADER
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Trả về kèm scaler để sau này nếu cần inverse_transform thì dùng
    return train_loader, val_loader, scaler_X, X_train_tensor.shape[2]