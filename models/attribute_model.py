import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision as vision
import sys
from scipy.misc import imresize
from torchvision import transforms, utils
from util.image_transforms import RandomRotate
import models.modules as modules
import scipy.ndimage
import scipy.misc

def build_model(config):
    if config.mode == 2:
        return modules.AttributeNet(config)
    if config.model == 'convnet':
        return modules.FiveLayerConvnet(config)
    elif config.model == 'inceptionnet':
        return modules.InceptionNet(config)
    elif config.model == 'densenet':
        return modules.ModifiedDenseNet(config)
    elif config.model == 'resnet':
        return modules.ResNet(config)

def get_attrib_loss_and_acc(config, logits, labels, pred_attr, real_attr):
    pred = np.argmax(logits.data.cpu().numpy(), axis=1)
    acc = np.mean(pred == labels.data.cpu().numpy())
    attr_loss = F.l1_loss(pred_attr, real_attr)
    loss = config.loss(logits, labels) + config.recon_weight * attr_loss
    return loss, attr_loss, acc, pred

def get_loss_and_acc(config, logits, labels):
    pred = np.argmax(logits.data.cpu().numpy(), axis=1)
    acc = np.mean(pred == labels.data.cpu().numpy())
    loss = config.loss(logits, labels)
    return loss, acc, pred

def build_and_train(config, train_fold, val_fold, test_fold):
    model = build_model(config).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('net is built')
    config.loss = nn.CrossEntropyLoss()
    config.optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=1e-4)
    config.scheduler = lr_scheduler.ReduceLROnPlateau(config.optimizer, 'min')

    best_val = 0.0

    save_train_loss = []
    save_train_acc = []
    save_val_loss  = []
    save_val_acc = []
    save_test_acc = []

    for epoch in range(config.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, config, train_fold, epoch, mode='Train')
        val_loss, val_acc, _, _ = run_epoch(model, config, val_fold, epoch, mode='Val')
        _, test_acc, all_labels, all_preds = run_epoch(model, config, test_fold, epoch, mode='Test')
        
        if val_acc > best_val: 
            best_val = val_acc
        
        print('Best val accuracy: {}'.format(best_val))
        print('Test accuracy: {}'.format(test_acc))
        config.scheduler.step(val_loss)

        save_train_loss.append(train_loss)
        save_train_acc.append(train_acc)
        save_val_loss.append(val_loss)
        save_val_acc.append(val_acc)
        save_test_acc.append(test_acc)

        np.save('outputs/train_loss_{}.npy'.format(config.experimentid), np.asarray(save_train_loss))
        np.save('outputs/train_acc_{}.npy'.format(config.experimentid), np.asarray(save_train_acc))
        np.save('outputs/val_loss_{}.npy'.format(config.experimentid), np.asarray(save_val_loss))
        np.save('outputs/val_acc_{}.npy'.format(config.experimentid), np.asarray(save_val_acc))
        np.save('outputs/test_acc_{}.npy'.format(config.experimentid), np.asarray(save_test_acc))
        np.save('outputs/labels{}.npy'.format(config.experimentid), all_labels)
        np.save('outputs/preds{}.npy'.format(config.experimentid), all_preds)
    
    return best_val

def prepare_data(config, images, labels, attributes, mode):
    #if config.mode == 2:
    #    images = np.array([imresize(image, (224, 224)) for image in images])
    #print(images)
    mean = 118.546752674
    std = 56.8170623267
    # 0.203418499378 0.0694907381909
    images = images.astype(np.uint8)
    if mode == 'Train' and config.augment > 0:
        transform = transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.Scale(320),
            vision.transforms.RandomCrop(299),
            vision.transforms.ToTensor()
        ])
        
        a_images = []
        a_labels = []
        a_attribs = []

        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]
            attribs = attributes[idx]
            times = 1
            images_ = []
            images_.append(image)
            a_images.append(image)
            for _ in range(config.flips):
                rotated = scipy.ndimage.interpolation.rotate(image, random.randrange(1, 360))
                rotated = scipy.misc.imresize(rotated, (299, 299))
                images_.append(rotated)
                a_images.append(rotated)
                times += 1
            for im in images_:
                for _ in range(config.augment):
                    expim = np.expand_dims(im, axis=2)
                    a_images.append(transform(expim).numpy().squeeze())
                    times += 1
            a_labels.extend([label] * times)
            a_attribs += [attribs] * times
         
        images = np.asarray(a_images)
        images = np.expand_dims(images, axis=1)
        labels = np.asarray(a_labels)
        attributes = np.asarray(a_attribs).astype(np.float32)
    else:
        images = np.expand_dims(images, axis=1)

    images = images.astype(np.float64)
    images -= mean
    images /= std
    #for image in images:
    images = torch.from_numpy(images).float()
    labels = torch.from_numpy(labels).long()
    attributes = torch.from_numpy(attributes).float()
    return images, labels, attributes

def run_epoch(model, config, fold, epoch, mode='Train'):
    '''
    main training method
    '''
    total_loss = 0.0 # <- haha
    total_acc = 0.0
    all_labels = []
    all_preds = []
    it = 0
    if mode == 'Train':
        model.train()
    else:
        model.eval()
    mode_data = None
    if config.mode == -1:
        more_data = fold
    else:
        more_data = fold.get_iterator()

    for item in more_data:
        images, labels, attributes = item
        it += 1
        # feed into model
        if config.mode != -1:
            images, labels, attributes = prepare_data(config, images, labels, attributes, mode)
        images = Variable(
            images,
            volatile=mode is not 'Train'
        ).cuda()
        labels = Variable(
            labels,
            volatile=mode is not 'Train'
        ).cuda()
        attributes = Variable(
            attributes,
            volatile=mode is not 'Train'
        ).cuda()

        attr_loss_num = None
        pred = None

        if config.mode != 2:
            logits = model(images)
            loss, acc, pred = get_loss_and_acc(config, logits, labels)
        else:
            logits, reconstruction = model(images)
            loss, attr_loss, acc, pred = get_attrib_loss_and_acc(config, logits, labels, reconstruction, attributes)
            attr_loss_num = attr_loss.data[0]

        loss_num = loss.data[0]

        if mode == 'Train':
            config.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5.0)
            config.optimizer.step()

            if it % 10 == 0:
                total_loss += loss_num
                total_acc += acc
                print(pred, labels.data.cpu().numpy())
                if config.mode != 2:
                    print('Epoch {} | Iteration {} | Loss {} | Accuracy {} | LR {}'.format(
                        epoch, it, loss_num, acc, config.lr)
                    )
                else:
                    print('Epoch {} | Iteration {} | Loss {} | Recon Loss {} | Accuracy {} | LR {}'.format(
                        epoch, it, loss_num, attr_loss_num, acc, config.lr)
                    )
                sys.stdout.flush()
        else:
            total_loss += loss_num
            total_acc += acc
            all_labels.extend(labels.data.cpu().numpy())
            all_preds.extend(pred)
    
    if mode == 'Train':
        total_loss /= it / 10
        total_acc /= it / 10
    else:
        total_loss /= it
        total_acc /= it

    print('{} loss:      {}'.format(mode, total_loss))
    print('{} accuracy:  {}'.format(mode, total_acc))

    return total_loss, total_acc, all_labels, all_preds
