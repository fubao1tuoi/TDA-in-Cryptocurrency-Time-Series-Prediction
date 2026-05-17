import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from gtda.time_series import SlidingWindow, TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude
import src.config

# =========================
# 1. TDA FEATURES
# =========================
def add_tda_features(df, target_cols=['return', 'funding_rate', 'quote_asset_volume'], 
                     window_size=30, time_delay=1, dimension=4, cache_path=None):
    """
    Hàm tính toán TDA cho nhiều cột dữ liệu cùng lúc an toàn và tối ưu.
    """
    if cache_path and os.path.exists(cache_path):
        print("📥 Load Multi-TDA features từ cache...")
        return pd.read_pickle(cache_path)

    print(f"🚀 Bắt đầu tính toán TDA cho {len(target_cols)} cột: {target_cols}")
    df_tda = df.copy()

    # --- Lớp bảo vệ: Dọn dẹp dữ liệu Infinity trước khi tính TDA ---
    df_tda = df_tda.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Tính trước các biến dùng cho Interaction
    trend_30 = np.sign(df_tda['return'].rolling(30).sum()) if 'return' in df_tda.columns else 1
    vol_zscore = (df_tda['quote_asset_volume'] - df_tda['quote_asset_volume'].rolling(30).mean()) / (df_tda['quote_asset_volume'].rolling(30).std() + 1e-8) if 'quote_asset_volume' in df_tda.columns else 1

    for col in target_cols:
        print(f"\n--- Đang xử lý không gian Hình thái học cho cột: [{col}] ---")
        
        try:
            # --- 1. TÍNH TOÁN TDA CƠ BẢN ---
            X = df_tda[col].values.reshape(1, -1)
            TE = TakensEmbedding(time_delay=time_delay, dimension=dimension)
            X_embedded_2d = TE.fit_transform(X)[0]

            SW = SlidingWindow(size=window_size, stride=1)
            X_windows = SW.fit_transform(X_embedded_2d) 

            VR = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
            diagrams = VR.fit_transform(X_windows)

            PE = PersistenceEntropy()
            features_entropy = PE.fit_transform(diagrams)

            AM = Amplitude(metric='wasserstein')
            features_amplitude = AM.fit_transform(diagrams)

            # Padding
            pad_len = max(0, len(df_tda) - len(features_entropy))
            df_tda[f'{col}_tda_entropy_H0'] = np.pad(features_entropy[:, 0], (pad_len, 0), constant_values=np.nan)
            df_tda[f'{col}_tda_entropy_H1'] = np.pad(features_entropy[:, 1], (pad_len, 0), constant_values=np.nan)
            df_tda[f'{col}_tda_amplitude_H0'] = np.pad(features_amplitude[:, 0], (pad_len, 0), constant_values=np.nan)
            df_tda[f'{col}_tda_amplitude_H1'] = np.pad(features_amplitude[:, 1], (pad_len, 0), constant_values=np.nan)

            # ==========================================
            # --- 2. BỔ SUNG: TDA DERIVATIVES & INTERACTIONS ---
            # ==========================================
            print(f"   -> Đang tính các biến phái sinh cho {col}...")

            # Nhóm 1: TDA Momentum (Đã thêm f'{col}_' vào TẤT CẢ các biến)
            for h in [7, 14, 30]:
                df_tda[f'{col}_tda_amp_H1_delta_{h}'] = df_tda[f'{col}_tda_amplitude_H1'] - df_tda[f'{col}_tda_amplitude_H1'].shift(h)
                df_tda[f'{col}_tda_entropy_H0_delta_{h}'] = df_tda[f'{col}_tda_entropy_H0'] - df_tda[f'{col}_tda_entropy_H0'].shift(h)
                df_tda[f'{col}_tda_amp_H1_roc_{h}'] = (df_tda[f'{col}_tda_amplitude_H1'] / (df_tda[f'{col}_tda_amplitude_H1'].shift(h) + 1e-8) - 1).fillna(0)

            # Nhóm 2: TDA Rolling Statistics
            for window in [14, 30]:
                df_tda[f'{col}_tda_entropy_H0_sma_{window}'] = df_tda[f'{col}_tda_entropy_H0'].rolling(window).mean()
                df_tda[f'{col}_tda_amplitude_H1_sma_{window}'] = df_tda[f'{col}_tda_amplitude_H1'].rolling(window).mean()
                df_tda[f'{col}_tda_entropy_H0_std_{window}'] = df_tda[f'{col}_tda_entropy_H0'].rolling(window).std()

            # Nhóm 3: TDA Interactions
            if 'return' in df_tda.columns:
                df_tda[f'{col}_tda_trend_interaction'] = df_tda[f'{col}_tda_amplitude_H1'] * trend_30
            if 'quote_asset_volume' in df_tda.columns:
                df_tda[f'{col}_tda_vol_weighted'] = df_tda[f'{col}_tda_entropy_H0'] * vol_zscore

            print(f"✅ Tính toán thành công cho: {col}")

        except Exception as e:
            print(f"❌ LỖI NGHIÊM TRỌNG tại cột {col}: {e}")
            continue
            
        # Giải phóng RAM sau mỗi cột
        gc.collect()

    # KHỐI NÀY ĐÃ ĐƯỢC ĐẨY RA NGOÀI VÒNG LẶP FOR
    if cache_path:
        df_tda.to_pickle(cache_path)
        print(f"\n💾 Đã lưu cache thành công tại {cache_path}")

    return df_tda


# =========================
# 2. FEATURE ENGINEERING
# =========================
def create_features(btc_data):
    df = btc_data.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    if 'return' not in df.columns:
        df['return'] = np.log(df['close'] / df['close'].shift(1)) #

    # Gộp chung tính toán SMA Distance (Log & Pct) và Volatility
    for window in [7, 30]:
        rolling_mean = df['close'].rolling(window).mean()
        
        # 1. Khoảng cách tới đường MA (dạng Log)
        df[f'sma_{window}_dist_log'] = np.log(df['close'] / rolling_mean)
        
        # 2. Khoảng cách tới đường MA (dạng Phần trăm)
        df[f'sma_{window}_dist_pct'] = (df['close'] / rolling_mean) - 1
        
        # 3. Tính Volatility (Chỉ cần tính 1 lần)
        df[f'volatility_{window}'] = df['return'].rolling(window).std()

    df['candle_body'] = (df['close'] - df['open']) / df['open'] #
    df['candle_body_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) #
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open'] #
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open'] #

    df['high_low_spread'] = (df['high'] - df['low']) / df['open']

    df['parkinson_vol_14'] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(df['high'] / df['low'])**2).rolling(14).mean()
    ) #

    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    df['garman_klass_vol'] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(14).mean()

    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - prev_close)
    tr3 = np.abs(df['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14_pct'] = true_range.rolling(14).mean() / df['close']

    for window in [14, 30]:
        df[f'skew_{window}'] = df['return'].rolling(window).skew() #
        df[f'kurt_{window}'] = df['return'].rolling(window).kurt() #

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-8)))) #

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_pct'] = (ema_12 - ema_26) / df['close'] #
    df['macd_signal_pct'] = df['macd_pct'].ewm(span=9, adjust=False).mean()

    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = (2 * std20 * 2) / ma20 #

    for window in [14, 30]:
        rolling_max = df['high'].rolling(window).max()
        rolling_min = df['low'].rolling(window).min()
        
        df[f'dist_to_max_{window}'] = (df['close'] - rolling_max) / rolling_max
        df[f'dist_to_min_{window}'] = (df['close'] - rolling_min) / rolling_min
        df[f'donchian_pos_{window}'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8) #

    df['taker_buy_ratio'] = df.get('taker_buy_base', 0) / (df.get('volume', 0) + 1e-8)
    df['volume_pct_change'] = df['volume'].pct_change()
    df['log_volume'] = np.log1p(df.get('volume', 0))
    df['avg_trade_size'] = df['volume'] / (df.get('number_of_trades', 0) + 1e-8)

    if 'quote_asset_volume' in df.columns:
        df['approx_vwap'] = df['quote_asset_volume'] / (df['volume'] + 1e-8) #
        df['vwap_close_dist'] = (df['close'] - df['approx_vwap']) / df['approx_vwap'] #

    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_pct_change'] = obv.pct_change() #

    df['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(30).mean()) /
        (df['volume'].rolling(30).std() + 1e-8)
    )

    df['trend_strength_7'] = df['close'].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x / x[0], 1)[0], raw=True
    ) #

    df['volatility_regime'] = (
        df['volatility_30'] > df['volatility_30'].rolling(100).mean()
    ).astype(int)

    df['return_volatility_interaction'] = df['return'] * df['volatility_7']
    df['rsi_volatility'] = df['rsi_14'] * df['volatility_7']

    for col in ['rsi_14', 'macd_pct', 'volatility_7']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    return df

def add_time_cyclical_features(df):
    """
    Mã hóa chu kỳ thời gian thành Sin/Cos để mô hình hiểu tính liên tục.
    """
    print("⏳ Đang tạo các đặc trưng Chu kỳ thời gian (Cyclical Time)...")
    df_time = df.copy()
    
    if not isinstance(df_time.index, pd.DatetimeIndex):
        df_time.index = pd.to_datetime(df_time.index)
        
    day_of_week = df_time.index.dayofweek
    month_of_year = df_time.index.month
    
    # Chu kỳ tuần (7 ngày)
    df_time['dow_sin'] = np.sin(day_of_week * (2 * np.pi / 7))
    df_time['dow_cos'] = np.cos(day_of_week * (2 * np.pi / 7))
    
    # Chu kỳ năm (12 tháng)
    df_time['month_sin'] = np.sin((month_of_year - 1) * (2 * np.pi / 12))
    df_time['month_cos'] = np.cos((month_of_year - 1) * (2 * np.pi / 12))
    
    return df_time

def add_lag_and_rolling_features(df):
    print("📈 Đang tạo các đặc trưng Lags và Rolling ...")
    df_lag = df.copy()
    
    # 1. Trí nhớ ngắn hạn (Lags) cho các biến Stationary
    stationary_cols = ['return', 'quote_asset_volume', 'funding_rate']
    for col in stationary_cols:
        if col in df_lag.columns:
            for i in [1, 3, 7]:
                df_lag[f'{col}_lag_{i}d'] = df_lag[col].shift(i)
                
    # 2. Độ biến động (Realized Volatility) và Động lượng (Momentum) của Return
    if 'return' in df_lag.columns:
        # Realized Volatility (Đo lường rủi ro/giông bão)
        df_lag['return_realized_vol_7d'] = df_lag['return'].shift(1).rolling(window=7).std()
        df_lag['return_realized_vol_30d'] = df_lag['return'].shift(1).rolling(window=30).std()
        
        # Momentum (Xu hướng trung bình của Return)
        df_lag['return_sma_7d'] = df_lag['return'].shift(1).rolling(window=7).mean()
        df_lag['return_sma_30d'] = df_lag['return'].shift(1).rolling(window=30).mean()
    
    return df_lag

# =========================
# 3. MULTI-HORIZON TARGETS
# =========================
def add_multi_targets(df, horizons=[1, 7, 30], task="regression"):
    df = df.copy()
    for h in horizons:
        future_log_return = np.log(df['close'].shift(-h) / df['close'])
        if task == "classification":
            df[f'target_{h}d'] = (future_log_return > 0).astype(int)
        else:
            df[f'target_{h}d'] = future_log_return
    return df


# =========================
# 4. FINAL PIPELINE
# =========================
def build_dataset(df, task="regression"):
    """
    Quy trình Feature Engineering tổng lực: Gốc + Chu kỳ + TDA + Target
    """
    print("🛠 Đang khởi tạo Pipeline Feature Engineering...")
    
    # 1. Feature gốc của mày
    df = create_features(df)
    
    # 2. Bơm thêm công lực (Chu kỳ thời gian & Trí nhớ ngắn hạn)
    df = add_time_cyclical_features(df)
    df = add_lag_and_rolling_features(df)
    
    # 3. Chạy TDA (Dùng cấu hình từ config.py)
    df = add_tda_features(
        df, 
        target_cols=src.config.TDA_TARGET_COLS,
        window_size=src.config.TDA_WINDOW_SIZE,
        dimension=src.config.TDA_DIMENSION,
        cache_path=src.config.TDA_CACHE_PATH
    )
    
    # 4. Tạo Target
    df = add_multi_targets(df, horizons=src.config.HORIZONS)
    
    # 5. Dọn dẹp NA
    df = df.dropna()
    print(f"🏁 Hoàn tất! Dataset sẵn sàng với {df.shape[1]} đặc trưng.")
    
    return df

# =========================
# 5. PLOTTING UTILS
# =========================
def plot_feature_importance_analysis(full_results_dict, horizons_list):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(horizons_list), figsize=(6 * len(horizons_list), 6))
    fig.suptitle('Analyzing Key Features (Top 10 Feature Importances)', 
                 fontsize=18, fontweight='bold', y=1.05)
    
    if len(horizons_list) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_list):
        if h not in full_results_dict: continue
        
        top_features = full_results_dict[h]['top_features'][:10]
        try:
            importances = [full_results_dict[h]['feature_importances'][feat] for feat in top_features]
        except (KeyError, TypeError):
            importances = list(np.linspace(100, 10, len(top_features)))

        top_features.reverse()
        importances.reverse()

        colors = ['#e74c3c' if any(kw in feat.lower() for kw in ['tda', 'betti', 'persistence']) 
                  else '#3498db' for feat in top_features]

        ax = axes[i]
        ax.barh(top_features, importances, color=colors, alpha=0.8)
        ax.set_title(f'Horizon: {h} Day(s)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=10)
        
        for j, val in enumerate(importances):
            ax.text(val + (max(importances)*0.01), j, f'{val:.1f}', va='center', fontsize=9)

    plt.figtext(0.5, -0.05, '* Red: TDA (Topology) Features | Blue: Finance Features', 
                wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold', color='#c0392b')
    plt.tight_layout()
    plt.show()
