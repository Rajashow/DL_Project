import argparse
import torch
from wann import wann, wannModel
from torch.utils.data import DataLoader
from sketchdataset import SketchDataSet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    parser = argparse.ArgumentParser(description='Wann testing')
    parser.add_argument('--wann_path', type=int, required=False,
                        default="wann_model.json")
    parser.add_argument('--model_path', type=str, required=False,
                        default="best_model_path.pth")
    return parser


def parse_args(args_dict):
    wann_path = args_dict['wann_path']
    model_path = args_dict['model_path']

    for key in args_dict.keys():
        print('parsed {} for {}'.format(args_dict[key], key))
    return wann_path, model_path


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    wann_path, model_path = parse_args(vars(args))
    batch_size = 32
    test_dataset = SketchDataSet("./data/", is_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Loaded %d test images' % len(test_dataset))
    test_wann_net(test_loader, wann_path, model_path)
