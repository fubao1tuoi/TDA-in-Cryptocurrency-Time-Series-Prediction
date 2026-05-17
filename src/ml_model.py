import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1. DATA PREPARATION (LOOKBACK FLATTEN)
# =========================
def create_lookback_dataset(df, feature_cols, target_col, lookback=7):
    """
    Biến đổi dữ liệu Tabular thông thường thành dạng có Lookback.
    Ví dụ: Lookback = 7 -> X tại t sẽ chứa feature của t, t-1, ..., t-6 (phẳng ra 1D array).
    Cách này giúp XGBoost bắt được chuỗi thời gian xịn như LSTM.
    """
    print(f"🛠 Creating Lookback Data: Horizon target '{target_col}' | Lookback = {lookback} days")
    
    # Lấy numpy array để xử lý cho lẹ
    data_values = df[feature_cols].values
    target_values = df[target_col].values
    dates = df.index.values
    
    X, y, valid_dates = [], [], []
    
    # Trượt window
    for i in range(lookback - 1, len(df)):
        # Lấy lookback ngày (flatten ra 1D)
        window_features = data_values[i - lookback + 1 : i + 1].flatten()
        X.append(window_features)
        y.append(target_values[i])
        valid_dates.append(dates[i])
        
    X = np.array(X)
    y = np.array(y)
    
    # Tạo tên cột mới cho dataframe flatten
    new_col_names = []
    for lag in range(lookback - 1, -1, -1):
        for col in feature_cols:
            suffix = f"_t-{lag}" if lag > 0 else "_t"
            new_col_names.append(f"{col}{suffix}")
            
    df_lookback = pd.DataFrame(X, columns=new_col_names, index=valid_dates)
    df_lookback[target_col] = y
    
    # Bỏ những dòng target bị NaN (do hàm shift của target)
    df_lookback = df_lookback.dropna(subset=[target_col])
    
    return df_lookback, new_col_names

# =========================
# 2. MODEL
# =========================
def get_xgboost_model(task="regression"):
    """
    Cấu hình XGBoost tối ưu cho Financial Time Series.
    Sử dụng colsample_bytree và subsample để tránh overfit,
    learning_rate thấp để học mượt.
    """
    if task == "regression":
        return xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective='reg:squarederror',
            tree_method='hist', # Dùng hist cho data lớn (như lookback 120d)
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1
        )
    else:
        return xgb.XGBClassifier(
            # Config tương tự cho classification nếu cần
            n_estimators=1000, learning_rate=0.01, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', tree_method='hist',
            early_stopping_rounds=50, random_state=42, n_jobs=-1
        )

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}