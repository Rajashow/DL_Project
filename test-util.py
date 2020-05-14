""" Adapted from Pulkit's modified main() from hw-4 """
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local files
from resnetbasic import BasicRes
from sketchdataset import SketchDataSet

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(train_loader, test_loader, learning_rate, num_epochs, experiment_name, 
            momentum, weight_decay, inc_learning, upper_lr, n_classes):
    net = BasicRes(n_classes).to(device)
    net.load_state_dict(torch.load('./' + experiment_name + '_best_detector.pth'))
    
    # evaluate the network on the test data
    tot = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            images, target = images.to(device), target.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred, 1)
            tot += target.size(0)
            correct += (pred_class == target).sum().item()

    
    print('Accuracy: {}'.format(100 * correct/tot))

def get_parser():
    parser = argparse.ArgumentParser(description='Resnet training')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--name', type=str, required=False, default="default-experiment")
    parser.add_argument('--momentum', type=float, required=False, default=0.9)
    parser.add_argument('--weightdecay', type=float, required=False, default=5e-4)
    parser.add_argument('--incrlr', action='store_true')
    parser.add_argument('--upperlr', type=float, required=False, default=0.01)
    return parser

def parse_args(args_dict):
    learning_rate = args_dict['lr']
    num_epochs = args_dict['epochs']
    batch_size = args_dict['batch']
    experiment_name = args_dict['name']
    momentum = args_dict['momentum']
    weight_decay = args_dict['weightdecay']
    inc_learning = args_dict['incrlr']
    upper_lr = args_dict['upperlr']

    for key in args_dict.keys():
        print('parsed {} for {}'.format(args_dict[key], key))
    
    return learning_rate, num_epochs, batch_size, experiment_name, momentum, \
        weight_decay, inc_learning, upper_lr



def main():
    '''
    Defualt main call:
        `main.py --lr 0.001 --epochs 50 --batch 1 --name default-experiment \
            --momentum 0.9 --weightdecay 5e-4 --incrlr False --upperlr 0.01`
    '''
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    learning_rate, num_epochs, batch_size, experiment_name, momentum, \
        weight_decay, inc_learning, upper_lr = parse_args(vars(args))

    # Load data and split 80-20 to training-testing
    train_dataset = SketchDataSet("./data/", is_train=True)
    test_dataset = SketchDataSet("./data/", is_train=False)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Loaded %d train images' % len(train_dataset))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Loaded %d test images' % len(test_dataset))

    test(train_loader, test_loader, learning_rate, num_epochs, experiment_name, 
            momentum, weight_decay, inc_learning, upper_lr, train_dataset.num_of_classes())

if __name__ == '__main__':
    main()
