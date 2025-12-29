import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, mode, K_IMP, ratio=1):
        super(SEBlock, self).__init__()
        self.K_IMP = K_IMP  # Store K_IMP as attribute
        self.mode = mode
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()
        
        # Initialize with default K_IMP, will be dynamically adjusted
        self.avg_pooling = nn.AdaptiveAvgPool2d((K_IMP, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((K_IMP, 1))
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling

        # Initialize FC layer with default size
        self.fc = nn.Sequential(
            nn.Linear(K_IMP, K_IMP // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(K_IMP // ratio, K_IMP, bias=False),
        )

    def _create_dynamic_fc(self, K):
        """Create a new FC layer for the given K dimension"""
        return nn.Sequential(
            nn.Linear(K, K // self.ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(K // self.ratio, K, bias=False),
        ).to(next(self.parameters()).device)

    def forward(self, x):
        B, N, K, T = x.shape  # (B,N,K,T)
        
        # Handle dimension mismatch by creating dynamic pooling and FC
        if K != self.K_IMP:
            # Create new pooling layers for the current K dimension
            if self.mode == "max":
                current_pooling = nn.AdaptiveMaxPool2d((K, 1)).to(x.device)
            else:  # avg
                current_pooling = nn.AdaptiveAvgPool2d((K, 1)).to(x.device)
            
            # Create new FC layer for the current K dimension
            current_fc = self._create_dynamic_fc(K)
            
            # Apply pooling and FC
            v = current_pooling(x).view(B, N, K)  # squeeze
            v = current_fc(v).view(B, N, K, 1)
            v = self.sigmoid(v)
            return x * v  # (B,N,K,T)
        else:
            # Use the original FC layer when dimensions match
            v = self.global_pooling(x).view(B, N, K)  # squeeze
            v = self.fc(v).view(B, N, K, 1)
            v = self.sigmoid(v)
            return x * v  # (B,N,K,T)
