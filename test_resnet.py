""" Adapted from Pulkit's modified main() from hw-4 """
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from wann import wann, wannModel
# Import local files
from resnetbasic import BasicRes
from sketchdataset import SketchDataSet

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(test_loader, experiment_name, n_classes):
    net = BasicRes(n_classes).to(device)
    net.load_state_dict(torch.load(
        './' + experiment_name + '_best_detector.pth'))

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


def test_wann_net(test_loader, wann_path, model_path):
    wann_path = wann_path.replace(".json", "")
    wann_obj = wann.load_json(wann_path)

    net = wannModel(wann_obj).to(device)
    net.load_state_dict(torch.load(model_path))

    # evaluate the network on the test data
    tot = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            target = target.to(device)
            images = images.reshape(images.shape[0], 784).to(device)
            pred = net(images)
            _, pred_class = torch.max(pred, 1)
            tot += target.size(0)
            correct += (pred_class == target).sum().item()

    print(f'Accuracy: {100 * correct/tot:.3}%')


def get_parser():
    parser = argparse.ArgumentParser(description='Resnet testing')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--name', type=str, required=False,
                        default="default-experiment")
    return parser


def parse_args(args_dict):
    batch_size = args_dict['batch']
    experiment_name = args_dict['name']

    for key in args_dict.keys():
        print('parsed {} for {}'.format(args_dict[key], key))

    return batch_size, experiment_name


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    batch_size, experiment_name = parse_args(vars(args))

    # Load data and split 80-20 to training-testing
    train_dataset = SketchDataSet("./data/", is_train=True)
    test_dataset = SketchDataSet("./data/", is_train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Loaded %d train images' % len(train_dataset))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Loaded %d test images' % len(test_dataset))

    test(test_loader, experiment_name, test_dataset.num_of_classes())


if __name__ == '__main__':
    main()
