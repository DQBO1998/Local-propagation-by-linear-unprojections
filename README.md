Local propagation by linear unprojections
---
### Environment setup (with Anaconda | Miniconda)
```
name: ml
channels:
- pytorch
- nvidia
dependencies:
- pip
- matplotlib
- pytorch
- torchvision
- torchaudio
- pytorch-cuda=11.7
- jupyter
- pip:
    - functorch
```
---
### Summary
This is one of many (independent) exploratory projects in the search for an alternative to back-propagation. Since it is
an ongoing effort, do not expect to find anything conclusive (not yet, at least).

The approach explored here is based on an idea proposed by [Nøkland and Eidnes](https://arxiv.org/abs/1901.06656), where 
each layer of a neural network is trained independently of the others. Once an inference signal goes through the network, 
each layer receives a separate learning signal.
