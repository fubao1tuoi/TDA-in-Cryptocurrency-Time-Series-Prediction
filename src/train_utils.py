import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy

# ==============================================================================
# 1. EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==============================================================================
# 2. CUSTOM LOSS: CHỐNG ĐI NGANG (FLATLINE)
# ==============================================================================
class CryptoLoss(nn.Module):
    """
    Hàm Loss tùy chỉnh để ép mô hình dự báo có độ biến động và đúng hướng.
    """
    def __init__(self, direction_weight=2.0, variance_weight=1.0):
        super().__init__()
        self.hub = nn.HuberLoss()
        self.direction_weight = direction_weight
        self.variance_weight = variance_weight

    def forward(self, y_pred, y_true):
        hub_loss = self.hub(y_pred, y_true)
        
        # Phạt nếu dự đoán sai hướng (cùng dương hoặc cùng âm thì mới tốt)
        # relu(-y_pred * y_true) sẽ > 0 nếu hai thằng khác dấu
        dir_penalty = torch.mean(torch.relu(-y_pred * y_true))
        
        # Phạt nếu độ biến động của dự đoán (Std) quá thấp so với thực tế
        std_penalty = torch.abs(torch.std(y_pred) - torch.std(y_true))
        
        return hub_loss + (self.direction_weight * dir_penalty) + (self.variance_weight * std_penalty)

# ==============================================================================
# 3. VÒNG LẶP HUẤN LUYỆN
# ==============================================================================
def train_model(model, train_loader, val_loader, epochs, lr, device, patience=20):
    model = model.to(device)
    # Dùng CryptoLoss để bắt mô hình phải nhấp nhô
    criterion = CryptoLoss(direction_weight=3, variance_weight=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    early_stopping = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # PHA TRAIN
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Có thể cắt bớt gradient nếu muốn
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
            
        # PHA VALIDATION
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                if isinstance(outputs, tuple): outputs = outputs[0]
                val_loss = criterion(outputs, y_batch)
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"🛑 Dừng sớm tại epoch {epoch+1}")
            break
            
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
        
    return model, history

# ==============================================================================
# 4. HÀM ĐÁNH GIÁ (EVALUATION)
# ==============================================================================
def evaluate_dl_model(model, data_loader, device='cpu'):
    """
    Chạy model trên tập dữ liệu để lấy dự báo dạng Scaled.
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.numpy())
            
    y_pred = np.vstack(predictions).flatten()
    y_true = np.vstack(actuals).flatten()
    
    # Tính toán Metrics nhanh (trên scale hiện tại)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred