""" Adapted from Pulkit's modified main() from hw-4 """
import os
import random
import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models

# Import resnet basic
from resnet-basic import BasicRes

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader, test_loader, learning_rate, num_epochs, experiment_name, 
            momentum, weight_decay, inc_learning, upper_lr):
    net = BasicRes()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Record best test loss
    best_test_loss = np.inf
    
    # Update learning rate?
    # Set to False if you want no updates to the learning rate
    if inc_learning:
        lr_increment = (upper_lr-learning_rate)/num_epochs

    
    prev_test_loss = 0.0
    test_loss = 0.0
    stagnant_test = False
    for epoch in range(num_epochs):
        net.train()
        
        # Update learning rate
        if inc_learning:
            learning_rate += lr_increment

        # Is the test error stagnant? Then decay learning rate
        if stagnant_test:
            stagnant_test = False
            print("decayed lr")
            learning_rate /= 5

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        total_loss = 0.
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            
            pred = net(images)
            loss = criterion(pred,target)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))

        # evaluate the network on the test data
        with torch.no_grad():
            prev_test_loss = test_loss
            test_loss = 0.0
            net.eval()
            for i, (images, target) in enumerate(test_loader):
                images, target = images.to(device), target.to(device)

                pred = net(images)
                loss = criterion(pred,target)
                test_loss += loss.item()
            test_loss /= len(test_loader)
        
        # Stagnant test loss?
        if abs(prev_test_loss - test_loss) < 0.3:
            stagnant_test = True

        # Save models
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            print('Updating best test loss: %.5f' % best_test_loss)
            torch.save(net.state_dict(),'{}_best_detector.pth'.format(experiment_name) if len(experiment_name) else 'best_detector.pth' )
        torch.save(net.state_dict(),'{}_detector.pth'.format(experiment_name) if len(experiment_name) else 'detector.pth')

def get_parser():
    parser = argparse.ArgumentParser(description='Resnet training')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--name', type=str, required=False, default="default-experiment")
    parser.add_argument('--momentum', type=float, required=False, default=0.9)
    parser.add_argument('--weight-decay', type=float, required=False, default=5e-4)
    parser.add_argument('--incr-lr', type=bool, required=False, default=False)
    parser.add_argument('--upper-lr', type=float, required=False, default=0.01)
    return parser

def parse_args(args_dict):
    learning_rate = args_dict['lr']
    num_epochs = args_dict['epochs']
    batch_size = args_dict['batch']
    experiment_name = args_dict['name']
    momentum = args_dict['momentum']
    weight_decay = args_dict['weight-decay']
    inc_learning = args_dict['incr-lr']
    upper_lr = args_dict['upper-lr']

    for key in args_dict.keys():
        print('parsed {} for {}'.format(arg_dicts[key], key))
    
    return learning_rate, num_epochs, batch_size, experiment_name, momentum,
        weight_decay, inc_learning, upper_lr



def main():
    '''
    Defualt main call:
        `main.py --lr 0.001 --epochs 50 --batch 24 --name default-experiment \
            --momentum 0.9 --weight-decay 5e-4 --incr-lr False --upper-lr 0.01`
    '''
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    learning_rate, num_epochs, batch_size, experiment_name, momentum,
        weight_decay, inc_learning, upper_lr = parse_args(vars(args))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Loaded %d train images' % len(train_dataset))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Loaded %d test images' % len(test_dataset))

    train(train_loader, test_loader, learning_rate, num_epochs, experiment_name, 
            momentum, weight_decay, inc_learning, upper_lr)

if __name__ == '__main__':
    main()