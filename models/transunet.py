import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class TransUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=6):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # CNN Encoder (Reduced ResNet-like)
        self.conv1 = ConvBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Feature size after 3 pools: img_size // 8
        feature_size = img_size // 8
        self.feature_size = feature_size
        
        # Patch Embedding / Linear Projection to Transformer Dimension
        # Instead of generic patches, we use the CNN output as "patches"
        self.patch_to_embed = nn.Linear(256, embed_dim)
        
        # Transformer Encoder
        transformer_layers = []
        for _ in range(num_layers):
            transformer_layers.append(TransformerBlock(embed_dim, num_heads, embed_dim * 4))
        self.transformer = nn.Sequential(*transformer_layers)
        
        # Projection back to CNN dimension
        self.embed_to_patch = nn.Linear(embed_dim, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(128 + 256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(64 + 128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(32 + 64, 32) # Cat with x1 instead of x
        
        self.out = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x) # [B, 64, H, W]
        p1 = self.pool1(x1) # [B, 64, H/2, W/2]
        
        x2 = self.conv2(p1) # [B, 128, H/2, W/2]
        p2 = self.pool2(x2) # [B, 128, H/4, W/4]
        
        x3 = self.conv3(p2) # [B, 256, H/4, W/4]
        p3 = self.pool3(x3) # [B, 256, H/8, W/8]
        
        # Transformer
        B, C, H, W = p3.shape
        # Reshape to sequence: [B, H*W, C]
        feat = rearrange(p3, 'b c h w -> b (h w) c')
        feat = self.patch_to_embed(feat) # [B, N, D]
        # Transformer expects [N, B, D]
        feat = feat.permute(1, 0, 2)
        feat = self.transformer(feat)
        # Back to [B, N, D] then [B, N, C]
        feat = feat.permute(1, 0, 2)
        feat = self.embed_to_patch(feat)
        # Back to [B, C, H, W]
        feat = rearrange(feat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Decoder with skip connections
        d3 = self.up3(feat) # [B, 128, H/4, W/4]
        d3 = torch.cat([d3, x3], dim=1) # skip from x3
        d3 = self.up_conv3(d3)
        
        d2 = self.up2(d3) # [B, 64, H/2, W/2]
        d2 = torch.cat([d2, x2], dim=1) # skip from x2
        d2 = self.up_conv2(d2)
        
        d1 = self.up1(d2) # [B, 32, H, W]
        d1 = torch.cat([d1, x1], dim=1) # skip from x1
        d1 = self.up_conv1(d1)
        
        return self.out(d1)
