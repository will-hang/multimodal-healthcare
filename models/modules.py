import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision

class AttributeNet(nn.Module):
    def __init__(self, config):
        super(AttributeNet, self).__init__()
        self.net = ResNetFE(config)
        self.fc_1 = nn.Linear(1000, 500)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_3 = nn.Linear(250, config.attrib_size)
        self.fc_class_1 = nn.Linear(250 + config.attrib_size, config.num_class)
        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, images):
        # try to reconstruct attributes from 1000-dim feature vector
        image_feats = self.net(images)
        hidden_1 = self.fc_1(image_feats)
        hidden_2 = self.fc_2(hidden_1)
        reconstruction = self.fc_3(hidden_2)

        cont_feats = reconstruction[:, :4]
        attrib_feats = F.sigmoid(reconstruction[:, 4:])
        reconstruction = torch.cat((cont_feats, attrib_feats), dim=1)
        
        final_image_feats = hidden_2
        final_all_feats = torch.cat((final_image_feats, reconstruction), dim=1)

        classification = self.fc_class_1(final_all_feats)

        return classification, reconstruction

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionNet(nn.Module):
    def __init__(self, config):
        super(InceptionNet, self).__init__()
        self.instance_net = vision.models.inception_v3(pretrained=config.pretrained) 
        # first layer: 151 --> stride 1, 299 --> stride 2
        self.instance_net.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.instance_net.fc = nn.Linear(2048, 1000)
        self.feat_fc = nn.Linear(1000, config.num_class)
        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, x):
        out = self.instance_net(x)
        if type(out) is tuple:
            out = out[0]
        out = self.feat_fc(self.dropout(F.relu(out)))
        return out #logits

class LOLNet(nn.Module):
    def __init__(self):
        super(LOLNet, self).__init__()
        self.net = nn.Linear(100, 100)
    def forward(self, x):
        print(x.size())
        return self.net(x)

class ResNetFE(nn.Module):
    def __init__(self, config):
        super(ResNetFE, self).__init__()
        self.net = vision.models.resnet50(pretrained=config.pretrained) 
        # first layer: 151 --> stride 1, 299 --> stride 2
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.net.fc = nn.Linear(512 * block.expansion, config.num_class)
        self.dropout = nn.Dropout(p=config.dropout)  
        print(len(list(self.net.children())))    
        for num, child in enumerate(self.net.children()):
            if num < 6:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        out = self.net(x)
        out = self.dropout(F.relu(out))
        return out #logits

class DenseNetFE(nn.Module):
    def __init__(self, config):
        super(DenseNetFE, self).__init__()
        num_init_features = 64
        self.net = vision.models.densenet121(pretrained=config.pretrained)
        self.net.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.net.classifier = nn.Linear(6400, 1000)
        for num, child in enumerate(self.net.children()):
            if num < 6:
                for param in child.parameters():
                    param.requires_grad = False
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, x):
        out = self.net(x)
        # except out just came from a linear layer and is a 1000-dim logit
        out = self.dropout(F.relu(out))
        #out = self.fc_2(self.dropout(F.relu(out)))
        return out

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.net = vision.models.resnet152(pretrained=config.pretrained) 
        # first layer: 151 --> stride 1, 299 --> stride 2
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.net.fc = nn.Linear(512 * block.expansion, config.num_class)
        self.feat_fc = nn.Linear(1000, config.num_class)
        self.dropout = nn.Dropout(p=config.dropout)        

    def forward(self, x):
        out = self.net(x)
        out = self.feat_fc(self.dropout(F.relu(out)))
        return out #logits

class ModifiedDenseNet(nn.Module):
    def __init__(self, config):
        super(ModifiedDenseNet, self).__init__()
        num_init_features = 64
        self.net = vision.models.densenet201(pretrained=config.pretrained)
        self.net.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.net.classifier = nn.Linear(6400, 1000)
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc_1 = nn.Linear(1000, 500)
        self.fc_2 = nn.Linear(500, config.num_class)
    
    def forward(self, x):
        out = self.net(x)
        # except out just came from a linear layer and is a 1000-dim logit
        out = self.fc_1(self.dropout(F.relu(out)))
        out = self.fc_2(self.dropout(F.relu(out)))
        return out

class FiveLayerConvnet(nn.Module):
    def __init__(self, config):
        super(FiveLayerConvnet, self).__init__()
        self.net = nn.Sequential(
                BasicConv2d(1, 128, kernel_size=3, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                BasicConv2d(128, 128, kernel_size=3, stride=2), 
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                BasicConv2d(128, 128, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), 
                BasicConv2d(128, 128, kernel_size=3, stride=2), 
            )
        self.fc = nn.Linear(2048, 1000)
        self.feat_fc = nn.Linear(1000, config.num_class)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        out = self.net(x)
        batch_size = out.size()[0]
        out = out.view(batch_size, -1)
        out = self.feat_fc(self.dropout(F.relu(self.fc(out))))
        return out
