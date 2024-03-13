import torch
import timm
from torchvision.transforms import v2

class EfficientNet:
    def __init__(self, include_fc: bool = False, device: str = 'cpu', ):
        self.device = device
        self.model = timm.create_model('tf_efficientnet_b5', pretrained=True, num_classes=0).to(self.device)
        self.include_fc = include_fc
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).to(self.device)
            x = self.transforms(x)
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.blocks[0](x)
            x = self.model.blocks[1](x)
            x = self.model.blocks[2](x)
            x = self.model.blocks[3](x)
            y1 = self.model.blocks[4](x)
            x = self.model.blocks[5](y1)
            y2 = self.model.blocks[6](x)
            if not self.include_fc:
                return [y1.cpu().numpy(),y2.cpu().numpy()]
            x = self.model.conv_head(y2)
            x = self.model.bn2(x)
            x = self.model.global_pool(x)
        return [y1.cpu().numpy(),y2.cpu().numpy(),x.cpu().numpy()]