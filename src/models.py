import torch
import torch.nn as nn
import torch.nn.functional as F

# This CRNN matches the architecture used in train_final.py (SimpleCRNN)
# which corresponds to the trained crnn_best.pth checkpoint.

class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, rnn_hidden=256):
        super().__init__()
        # SimpleCRNN architecture from train_final.py
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
        )
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = self.pool_h(f)
        f = f.squeeze(2).permute(0, 2, 1)  # (B, W, C)
        rnn_out, _ = self.rnn(f)
        logits = self.fc(rnn_out)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        return log_probs
