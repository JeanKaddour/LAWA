# Latest Weight Averaging (LAWA)
[Paper Link](https://arxiv.org/abs/2209.14981)
# Abstract
Training vision or language models on large datasets can take days, if not weeks. We show that averaging the weights of the k latest checkpoints, each collected at the end of an epoch, can speed up the training progression in terms of loss and accuracy by dozens of epochs, corresponding to time savings up to ~68 and ~30 GPU hours when training a ResNet50 on ImageNet and RoBERTa-Base model on WikiText-103, respectively. We also provide the code and model checkpoint trajectory to reproduce the results and facilitate research on reusing historical weights for faster convergence.

# Codebase structure
* There is a standalone LAWA Scheduler in `lawa_scheduler.py`, which can be imported into any PyTorch training loop
* Besides that, this repo contains two forks, one for each of the `bert` and `imagenet` experiments
  * the ImageNet code is forked from [PyTorch examples](https://github.com/pytorch/examples/tree/main/imagenet)
  * the roBERTa code is forked from [fairseq](https://github.com/facebookresearch/fairseq)
* I tried to add LAWA to the forks with minimal modifications
  * that means, most code is not touched by me
  * Installation instructions and requirements are contained in the respective README file of each  
  * this LAWA implementation does not require additional packages, it only requires PyTorch
  * the key LAWA averaging operations can be found in `lawa_utils.py`
* As of now, this codebase is designed for averaging checkpoints post-training, not for online training
  * I wanted to be able to experiment with historical checkpoints after running a full training run only once
  * that means, to try out LAWA, you first need to obtain training trajectory checkpoints (either by downloading mine, see below, or creating you own)
  * then, you can experiment with averaging a number of checkpoints, as shown in the results-reproducing shell scripts below


# Reproduce the results
* In both folders (`bert` and `imagenet`), there is a shell script called `reproduce_results.sh`
* To run it, you need to modify the checkpoint and dataset paths 

# Checkpoints
* [ResNet50/ImageNet Checkpoints (0.19GB per checkpoint, 17GB in total)](https://www.dropbox.com/sh/q7371s2heklkpuk/AABcvP2_fs3DhVs2jmE_DGIWa?dl=0)
* [RoBERTa-Base/WikiText103 Checkpoints (1.57GB per checkpoint, 299GB in total)](https://www.dropbox.com/sh/e6gren6nfcyrort/AACHZGKcEem2m0CQJ7a1pTs6a?dl=0)

# Citation
If you find this code useful for your work, please cite our paper as
```bibtex
@article{lawa,
  title={Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging},
  author={Kaddour, Jean},
  journal={arXiv preprint arXiv:2209.14981},
  year={2022},
  url={https://arxiv.org/abs/2209.14981}
}
```
