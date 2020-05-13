from wann import wannModel
from torch.utils.data import DataLoader
from sketchdataset import SketchDataSet
from wann import wann
from train import Train
import torch


wann_class = wann

hyper_params = {"p_weighed_rank": .5, "w": -2, "%_reap": .5}
class_args = {"input_dim": (784), "output_dim": 6}
trainer = Train(wann_class, class_args, 100, hyper_params)

train_dataset = SketchDataSet("./data/", is_train=True)

batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print('Loaded %d train images' % len(train_dataset))
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainer.populate()
for _ in range(2):
    trainer._self_mutate()

net = wannModel(trainer.pop[0])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), .003)
total_loss = 0
for i, (images, target) in enumerate(train_loader):
    images, target = images.to(device), target.to(device)
    images = images.reshape(batch_size, 784).to(device)
    target = target.to(device)
    pred = net(images)
    loss = criterion(pred, target)
    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
    if i == 2:
        break
