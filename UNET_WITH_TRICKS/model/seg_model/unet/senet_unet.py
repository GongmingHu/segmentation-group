from model.seg_model.unet.unet_parts import *
from model.encoder.resnet import *
from model.seg_model.unet.ppm import PPM
from model.seg_model.unet.self_attention import attentionHead


class adoptedUNet(nn.Module):
    def __init__(self, layer=34, use_ppm=True, use_attention=True,
                 up_way='bilinear', num_classes=1, pretrained=True,
                 criterion=nn.BCEWithLogitsLoss()):
        super(adoptedUNet, self).__init__()
        assert layer in [18, 34]
        assert up_way in ['bilinear', 'dconv']
        self.use_ppm = use_ppm
        self.use_attention = use_attention
        self.criterion = criterion

        # Encoder
        if layer == 34:
            model = resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = nn.Sequential(model.maxpool, model.layer1)
        self.layer2, self.layer3, self.layer4 = model.layer2, model.layer3, model.layer4

        if use_ppm:
            bins = (1, 2, 3, 6)
            self.ppm = PPM(512, int(512 / len(bins)), bins)

        if use_attention:
            self.attention = attentionHead(512)

        # Decoder
        self.up1 = Up(1024, 256, up_way)
        self.up2 = Up(512, 128, up_way)
        self.up3 = Up(256, 64, up_way)
        self.up4 = Up(128, 64, up_way)
        self.up5 = Up(128, 64, up_way)
        self.outc = adoptedOutConv(64, num_classes)

    def forward(self, x, y=None):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        if self.use_attention:
            x2 = self.attention(x2)
        x3 = self.layer3(x2)
        if self.use_attention:
            x3 = self.attention(x3)
        x4 = self.layer4(x3)
        if self.use_attention:
            x4 = self.attention(x4)

        if self.use_ppm:
            x4 = self.ppm(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        logits = self.outc(x)
        out = torch.sigmoid(logits)

        if self.training:
            main_loss = self.criterion(logits, y)
            return logits, out, main_loss
        else:
            return out
