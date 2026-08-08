import torch
import torch.nn as nn
import timm

class ViT_UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()

        self.n_patches = 14
        self.hidden_dim = 768

        # This is the EXACT decoder from your notebook file.
        # It ends with Conv2d, not ConvTranspose2d.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed.to(x.device)
        x = self.encoder.pos_drop(x)

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.norm(x)
        x = x[:, 1:, :]

        x = x.transpose(1, 2).contiguous().view(B, self.hidden_dim, self.n_patches, self.n_patches)

        # The output of this decoder is 56x56
        out = self.decoder(x)

        return out