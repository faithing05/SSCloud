""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, return_attention_maps=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply transformer bottleneck if enabled
        if self.use_transformer and self.transformer_bottleneck is not None:
            x5 = self.transformer_bottleneck(x5)
        
        attention_maps = []
        
        if return_attention_maps and self.use_attention:
            x, att1 = self.up1(x5, x4, return_attention=True)
            x, att2 = self.up2(x, x3, return_attention=True)
            x, att3 = self.up3(x, x2, return_attention=True)
            x, att4 = self.up4(x, x1, return_attention=True)
            attention_maps = [att1, att2, att3, att4]
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        if return_attention_maps and self.use_attention:
            return logits, attention_maps
        else:
            return logits




class HybridSSCloudUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_transformer=True, use_attention=True):
        super(HybridSSCloudUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_transformer = use_transformer
        self.use_attention = use_attention

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Transformer bottleneck between down4 and up1
        if use_transformer:
            self.transformer_bottleneck = TransformerBottleneck(
                in_channels=1024 // factor,
                hidden_size=768,
                num_layers=1,
                num_heads=12
            )
        else:
            self.transformer_bottleneck = None
        
        # Up blocks with attention gates
        self.up1 = UpWithAttention(1024, 512 // factor, bilinear, use_attention)
        self.up2 = UpWithAttention(512, 256 // factor, bilinear, use_attention)
        self.up3 = UpWithAttention(256, 128 // factor, bilinear, use_attention)
        self.up4 = UpWithAttention(128, 64, bilinear, use_attention)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, return_attention_maps=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply transformer bottleneck if enabled
        if self.use_transformer and self.transformer_bottleneck is not None:
            x5 = self.transformer_bottleneck(x5)
        
        attention_maps = []
        
        if return_attention_maps and self.use_attention:
            x, att1 = self.up1(x5, x4, return_attention=True)
            x, att2 = self.up2(x, x3, return_attention=True)
            x, att3 = self.up3(x, x2, return_attention=True)
            x, att4 = self.up4(x, x1, return_attention=True)
            attention_maps = [att1, att2, att3, att4]
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        if return_attention_maps and self.use_attention:
            return logits, attention_maps
        else:
            return logits