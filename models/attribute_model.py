import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision as vision
import sys

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ThreeLayerConvnet(nn.Module):
    def __init__(self, config):
        super(ThreeLayerConvnet, self).__init__()
        self.net = nn.Sequential(
                BasicConv2d(1, 128, kernel_size=3, stride=2),
                BasicConv2d(128, 128, kernel_size=3, stride=2),
                BasicConv2d(128, 128, kernel_size=3, stride=2),
            )
        self.fc = nn.Linear(2048, 1000)
        self.feat_fc = nn.Linear(1000, config.num_class)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        out = self.net(x)
        out = self.feat_fc(self.dropout(F.relu(self.fc(out))))
        return out

def build_model(config):
    return ThreeLayerConvnet(config)

def get_loss_and_acc(logits, labels):
    pred = np.argmax(logits.data.cpu().numpy(), axis=1)
    acc = np.mean(pred == labels)
    loss = config.loss(logits, labels)
    return loss, acc

def build_and_train(config, train_fold, val_fold):
    model = build_model(config).cuda()
    config.loss = nn.CrossEntropyLoss()
    config.optimizer = optim.Adam(model.parameters(), lr=config.lr)
    config.scheduler = lr_scheduler.ReduceLROnPlateau(config.optimizer, 'min')

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
        scheduler.step(val_loss)

        save_train_loss.append(train_loss)
        save_train_acc.append(train_acc)
        save_val_loss.append(val_loss)
        save_val_acc.append(val_acc)

    np.save('outputs/train_loss.npy', np.asarray(save_train_loss))
    np.save('outputs/train_acc.npy', np.asarray(save_train_acc))
    np.save('outputs/val_loss.npy', np.asarray(save_val_loss))
    np.save('outputs/val_acc.npy', np.asarray(save_val_acc))
    
    return best_val

def run_epoch(model, config, fold, epoch, mode='Train'):
    '''
    main training method
    '''
    total_loss = 0.0 # <- haha
    total_acc = 0.0
    it = 0

    for images, labels, attributes in fold.get_iterator():
        it += 1
        # feed into model
        print(images)
        images = Variable(torch.Tensor(images).float(), volatile=mode is not 'Train').cuda()
        labels = Variable(torch.Tensor(labels).long(), volatile=mode is not 'Train').cuda()
        logits = model(images) #, attributes)
        loss, acc = get_loss_and_acc(logits, labels)

        if mode == 'Train':
            config.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5.0)
            loss_num = loss.data[0]
            config.optimizer.step()

            if it % 10 == 0:
                train_loss.append(loss_num)
                train_acc.append(acc)
                print('Epoch {} | Iteration {} | Loss {} | Accuracy {} | LR {}'.format(
                    epoch, it, loss_num, acc, config.optimizer.param_groups['lr']))
                sys.stdout.flush()

    total_loss /= it
    total_acc /= it

    print('{} loss:      {}'.format(mode, total_loss))
    print('{} accuracy:  {}'.format(mode, total_acc))

    return total_loss, total_acc 
