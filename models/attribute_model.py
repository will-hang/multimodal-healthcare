import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision as vision
import sys
from scipy.misc import imresize

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
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc_1 = nn.Linear(1000, 500)
        self.fc_2 = nn.Liner(500, config.num_class)
    
    def forward(self, x):
        out = self.net(x)
        # except out just came from a linear layer and is a 1000-dim logit
        out = self.fc_1(self.dropout(F.relu(out)))
        out = self.fc_2(self.dropout(F.relu(out)))
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

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

def build_model(config):
    if config.model == 'conv':
        return FiveLayerConvnet(config)
    elif config.model == 'inception':
        return InceptionNet(config)
    elif config.model == 'dense':
        return ModifiedDenseNet(config)

def get_loss_and_acc(config, logits, labels):
    pred = np.argmax(logits.data.cpu().numpy(), axis=1)
    acc = np.mean(pred == labels.data.cpu().numpy())
    loss = config.loss(logits, labels)
    return loss, acc

def build_and_train(config, train_fold, val_fold):
    model = build_model(config).cuda()
    config.loss = nn.CrossEntropyLoss()
    config.optimizer = optim.Adam(model.parameters(), lr=config.lr)
    #config.scheduler = lr_scheduler.ReduceLROnPlateau(config.optimizer, 'min')

    best_val = 0.0

    save_train_loss = []
    save_train_acc = []
    save_val_loss  = []
    save_val_acc = []

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch(model, config, train_fold, epoch, mode='Train')
        val_loss, val_acc = run_epoch(model, config, val_fold, epoch, mode='Test')
        
        if val_acc > best_val: 
            best_val = val_acc
        
        print('Best val accuracy: {}'.format(best_val))
        #config.scheduler.step(val_loss)

        save_train_loss.append(train_loss)
        save_train_acc.append(train_acc)
        save_val_loss.append(val_loss)
        save_val_acc.append(val_acc)

    np.save('outputs/train_loss.npy', np.asarray(save_train_loss))
    np.save('outputs/train_acc.npy', np.asarray(save_train_acc))
    np.save('outputs/val_loss.npy', np.asarray(save_val_loss))
    np.save('outputs/val_acc.npy', np.asarray(save_val_acc))
    
    return best_val

def prepare_inputs(images):
    #images = np.array([imresize(image, (299, 299)) for image in images])
    #print(images)
    mean = 0.203418499378 
    std = 0.0694907381909
    # 0.204751553205 0.0690093641502
    images /= 255.0
    images -= mean
    images /= std
    return images.astype(np.float32)

def run_epoch(model, config, fold, epoch, mode='Train'):
    '''
    main training method
    '''
    total_loss = 0.0 # <- haha
    total_acc = 0.0
    it = 0
    if mode == 'Train':
        model.train()
    else:
        model.eval()
    for images, labels, attributes in fold.get_iterator():
        it += 1
        # feed into model
        images = prepare_inputs(images)
        images = Variable(
            torch.from_numpy(images).float().unsqueeze_(1), 
            volatile=mode is not 'Train'
        ).cuda()
        labels = Variable(torch.from_numpy(labels).long(), volatile=mode is not 'Train').cuda()
        logits = model(images) #, attributes)
        loss, acc = get_loss_and_acc(config, logits, labels)
        loss_num = loss.data[0]

        if mode == 'Train':
            config.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5.0)
            config.optimizer.step()

            if it % 10 == 0:
                total_loss += loss_num
                total_acc += acc
                print('Epoch {} | Iteration {} | Loss {} | Accuracy {} | LR {}'.format(
                    epoch, it, loss_num, acc, config.lr))
                sys.stdout.flush()
        else:
            total_loss += loss_num
            total_acc += acc
    
    if mode == 'Train':
        total_loss /= it / 10
        total_acc /= it / 10
    else:
        total_loss /= it
        total_acc /= it

    print('{} loss:      {}'.format(mode, total_loss))
    print('{} accuracy:  {}'.format(mode, total_acc))

    return total_loss, total_acc 
