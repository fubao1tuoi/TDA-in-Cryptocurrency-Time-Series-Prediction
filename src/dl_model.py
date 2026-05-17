import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. CUSTOM LAYERS (LỚP TÙY CHỈNH CHỐNG OVERFIT)
# ==============================================================================
class VariationalDropout(nn.Module):
    """
    Variational Dropout: Khác với Dropout thường, kỹ thuật này giữ nguyên
    mask dropout trong suốt các bước thời gian (time steps), giúp LSTM
    giữ được trí nhớ dài hạn tốt hơn mà vẫn chống được Overfitting.
    """
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p


    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # x shape: (Batch, Dim)
        mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask


# ==============================================================================
# 2. KIẾN TRÚC 1: VANILLA LSTM MODEL (CƠ BẢN)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1, dropout=0.2):
        """
        Mô hình LSTM thuần túy.
        - input_dim: Số lượng features (VD: 30)
        - hidden_dim: Số lượng nơ-ron trong hidden state (VD: 64, 128)
        - layer_dim: Số lớp LSTM xếp chồng lên nhau (VD: 1 hoặc 2)
        - output_dim: 1 (Vì bài toán của ta là Regression - Dự báo 1 giá trị)
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
       
        # Lớp LSTM cốt lõi (batch_first=True để nhận input dạng: Batch, Seq, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,
                            dropout=dropout if layer_dim > 1 else 0)


        self.dropout = VariationalDropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        # x shape: (Batch_size, Lookback_days, Features)
        out, (hn, cn) = self.lstm(x)
       
        # Chỉ lấy Hidden State của ngày cuối cùng (tại time step cuối)
        last_out = out[:, -1, :]
       
        # Áp dụng Variational Dropout
        last_out = self.dropout(last_out)
       
        # Qua lớp Fully Connected để ra giá trị dự báo
        pred = self.fc(last_out)
        return pred

