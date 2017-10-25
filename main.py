import random
import numpy as np
import argparse
import torch
import models.attribute_model as attribute_model
from torchvision import datasets, transforms
from util.DataMaster import DataMaster 
from util.DataMasterROIs import DataMaster as DataMasterROIs
# Instantiate the parser
parser = argparse.ArgumentParser(description='hyperparams', add_help=False)
parser.add_argument('--mode', type=int, nargs = '?', default= 0)
parser.add_argument('--model', type=str, nargs = '?', default='conv')
parser.add_argument('--epochs', type=int, nargs = '?', default= 150)
parser.add_argument('--batch_size', type=int, nargs = '?', default= 4)
parser.add_argument('--image_width', type=int, nargs = '?', default= 512)
parser.add_argument('--image_height', type=int, nargs = '?', default= 512)
parser.add_argument('--attrib_size', type=int, nargs = '?', default=104)
parser.add_argument('--dilation_rate', type=int, nargs = '?', default= 2)
parser.add_argument('--fold_count', type=int, nargs = '?', default= 10)
parser.add_argument('--cross_validation', type=bool, nargs = '?', default=False)
parser.add_argument('--num_class', type=int, nargs = '?', default= 3)
parser.add_argument('--dropout', type=float, nargs = '?', default= 0.5)
parser.add_argument('--lr', type=float, nargs = '?', default= 4e-4)
parser.add_argument('--max_grad_norm', type=float, nargs = '?', default= 5.0)
parser.add_argument('--beta', type=float, nargs = '?', default= 1e-2)
parser.add_argument('--reg', type=float, nargs = '?', default= 0.05)
parser.add_argument('--save_model', type=bool, nargs = '?', default= True)
parser.add_argument('--save_dir', type=str, nargs = '?', default= 'checkpts')
parser.add_argument('--preprocess_from_scratch', type=bool, nargs = '?', default= False)
parser.add_argument('--load_saved_model', type=bool, nargs = '?', default= False)
parser.add_argument('--train_phase', type=bool, nargs = '?', default= True)
parser.add_argument('--pretrained', type=bool, nargs = '?', default=True)
parser.add_argument('--augment', type=int, nargs = '?', default=0)
parser.add_argument('--experimentid', type=int, nargs = '?', default=0)
parser.add_argument('--flips', type=int, nargs = '?', default=0)
parser.add_argument('--recon_weight', type=float, nargs = '?', default=0.0)
parser.add_argument('--layers_frozen', type=int, nargs = '?', default=0)

def build_and_train(config, train_fold, val_fold, test_fold):
    '''
    Run different build and training process for different experiments
    '''
    val_error = 0.0
    print("begin training")
    val_error = attribute_model.build_and_train(config, train_fold, val_fold, test_fold)
    
    return val_error

if __name__ == '__main__':
    '''
    main function
    '''
    config = parser.parse_args()
    if config.mode == 0:
        dm = DataMaster(config.batch_size, config.fold_count, config.preprocess_from_scratch)
    elif config.mode == 1 or config.mode == 2:        
        dm = DataMasterROIs(config.batch_size, config.fold_count, config.preprocess_from_scratch)
    elif config.mode == -1:
        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=config.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config.batch_size, shuffle=True, **kwargs)
        val_acc = build_and_train(config, train_loader, test_loader)
        print("Final validation accuracy: {}".format(val_acc))
    if config.cross_validation:
        for fold in range(config.fold_count):
            train_fold, val_fold, test_fold = dm.next_fold()
            val_acc = build_and_train(config, train_fold, val_fold, test_fold)
            print("Validation accuracy for fold {}: {}".format(fold, val_acc))
    else:
        train_fold, val_fold, test_fold = dm.next_fold()
        val_acc = build_and_train(config, train_fold, val_fold, test_fold)
        print("Final validation accuracy: {}".format(val_acc))
