""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AdditiveAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, return_attention=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        if return_attention:
            return x * psi, psi
        else:
            return x * psi


class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_size=768, num_layers=1, num_heads=12):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Reduce channels to hidden size
        self.channel_reduction = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        
        # Create multi-head self-attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            )
            self.attention_layers.append(nn.LayerNorm(hidden_size))
            self.attention_layers.append(nn.Linear(hidden_size, hidden_size * 4))
            self.attention_layers.append(nn.GELU())
            self.attention_layers.append(nn.Linear(hidden_size * 4, hidden_size))
            self.attention_layers.append(nn.LayerNorm(hidden_size))
        
        # Project back to original channels
        self.channel_expansion = nn.Conv2d(hidden_size, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reduce channels
        x_reduced = self.channel_reduction(x)
        
        # Reshape for attention: (batch, height*width, hidden_size)
        x_flat = x_reduced.flatten(2).transpose(1, 2)
        
        # Pass through attention layers
        hidden_states = x_flat
        for i in range(0, len(self.attention_layers), 6):
            # Self-attention
            attn_output, _ = self.attention_layers[i](hidden_states, hidden_states, hidden_states)
            hidden_states = self.attention_layers[i+1](hidden_states + attn_output)
            
            # Feed-forward
            ff_output = self.attention_layers[i+2](hidden_states)
            ff_output = self.attention_layers[i+3](ff_output)
            ff_output = self.attention_layers[i+4](ff_output)
            hidden_states = self.attention_layers[i+5](hidden_states + ff_output)
        
        # Reshape back to spatial
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, self.hidden_size, height, width)
        
        # Expand channels back
        output = self.channel_expansion(hidden_states)
        
        # Residual connection
        return output + x


class UpWithAttention(nn.Module):
    """Upscaling then double conv with Attention"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        # Add attention gate if enabled
        if use_attention:
            # Calculate attention gate dimensions
            # F_g = in_channels // 2 (from upsampled path)
            # F_l = in_channels // 2 (from skip connection)  
            # F_int = in_channels // 4 (intermediate channels)
            self.att = AdditiveAttentionGate(
                F_g=in_channels // 2, 
                F_l=in_channels // 2, 
                F_int=in_channels // 4
            )
        else:
            self.att = None

    def forward(self, x1, x2, return_attention=False):
        x1 = self.up(x1)
        
        attention_mask = None
        # Apply Attention before concat if enabled
        if self.use_attention and self.att is not None:
            if return_attention:
                x2, attention_mask = self.att(g=x1, x=x2, return_attention=True)
            else:
                x2 = self.att(g=x1, x=x2)
        
        # Original concat and conv logic
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        if return_attention:
            return self.conv(x), attention_mask
        else:
            return self.conv(x)
