# WANN vs ResNet

## Initialization
Install the requirements. On Google Colab, only `jenkspy` and `unidip` need to be intalled.

```
pip install -r requirements.txt
```

Run the download script to obtain the dataset used for the 18 classes we trained on.

```
sh download.sh
```

## Training

To train the resnet, run the following command.

```
python train_resnet.py --lr 0.0001 --epochs 50 --batch 1 --name default-experiment --momentum 0.9 --weightdecay 5e-4 --incrlr False --upperlr 0.01
```

The flags correspond to the following:
* `lr` = learning rate

* `epochs` = number of epochs

* `batch` = batch size (__warning:__ RAM usable quickly increases as batch size increases)

* `name` = what the model .pth file will be named

* `momentum` = gradient descent momentum

* `weightdecay` = gradient descent weight decay

* `incrlr` = if included, then we increment the learning rate to the upper limit `upperlr`. If omitted, we do not increment the learning rate.

* `upperlr` = the upper limit learning rate we iteratively sum to

A similar command exists for training a population of WANN's with the NEAT algorithm.

```
python train_wann.py --epochs 50 --batch 24 --name default-experiment
```

### Models
Model files are generated for resnet and WANN training. For a name `experiment-name`, you get model files `experiment-name_detector.pth` and `experiment-name_best_detector.pth` for resnet. For WANN, the file `experiment-name.json` is generated.

### Testing
The commands for testing are similar to training. To test the accuracy of the resnet and WANN, call python on the ```test_resnet.py``` and ```test_wann.py``` files.

## References

General sources:

* [Gaier - Weight Agnostic NNs](https://weightagnostic.github.io/)
* [Lu - Free-hand Sketch Recognition](http://cs231n.stanford.edu/reports/2017/pdfs/420.pdf)

Specific sources:

* Data
  * [Google quickdraw](https://github.com/googlecreativelab/quickdraw-dataset)
  * [TU-Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) not used
due to the existence of ambiguous categories. The pruned data set due to
[Schneider](https://dl.acm.org/doi/10.1145/2661229.2661231) was apparently never
released.
* numpy NEAT algorithm - ?
* Coding:
  * [Sample CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  * [Adding res connection](https://discuss.pytorch.org/t/add-residual-connection/20148/8)
  * [torch.nn docu](https://pytorch.org/docs/stable/nn.html)
