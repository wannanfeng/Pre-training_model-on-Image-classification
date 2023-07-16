from torchvision import models
import torch.nn as nn
import torch
# model = models.vgg16()
# print(model)  Parts: features, avgpool, classifier

class VCG(nn.Module):
    def __init__(self, ):
        super(VCG, self).__init__()
        # model = models.vgg16(weights='imagenet')
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():  # 冻住特征提取的参数
            param.requires_grad = False
        self.features = model.features

        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x  # (batch, 2)
