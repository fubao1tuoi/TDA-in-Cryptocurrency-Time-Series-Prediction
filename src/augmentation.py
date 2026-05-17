import torch
from torch.utils.data import Dataset

class TimeSeriesAugmentDataset(Dataset):
    """
    Dataset tùy chỉnh cho PyTorch hỗ trợ On-the-fly Gaussian Noise Augmentation.
    """
    def __init__(self, X_tensor, y_tensor, noise_level=0.01, is_training=True):
        """
        :param X_tensor: Dữ liệu 3D tensor (Batch, Time_steps, Features)
        :param y_tensor: Nhãn 1D/2D tensor
        :param noise_level: Độ lệch chuẩn của nhiễu Gaussian
        :param is_training: Chỉ thêm nhiễu khi đang Train. Validation/Test giữ nguyên.
        """
        self.X = X_tensor
        self.y = y_tensor
        self.noise_level = noise_level
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Lấy ra 1 chuỗi sequence (Time_steps, Features)
        x_seq = self.X[idx].clone() # Clone để không làm hỏng dữ liệu gốc
        y_val = self.y[idx]

        # Chỉ tiêm nhiễu nếu đang ở pha Training và noise_level > 0
        if self.is_training and self.noise_level > 0:
            # Tạo ma trận nhiễu cùng kích thước với x_seq
            noise = torch.randn_like(x_seq) * self.noise_level
            x_seq = x_seq + noise

        return x_seq, y_val