import time
from math import ceil, floor, sqrt
from types import LambdaType
from typing import Iterable, Sequence, Callable

import torch as th
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@th.no_grad()
def unproj(input_shape: Sequence[int], layer: nn.Module, target_features: int) -> nn.Sequential:
    # Create a probe to send through the model and catch at the other end.
    in_probe = th.empty((1, *input_shape))
    out_probe = layer(in_probe)
    # Count the number of output features and create the linear projection.
    hidden_features = out_probe[0].flatten().shape[0]
    linear = nn.Linear(hidden_features, target_features, bias=False)
    # Note: We need to flatten the output and assume the target will also be flat.
    # This so that the optimization is agnostic to the task at hand.
    return nn.Sequential(nn.Flatten(), linear)


@th.no_grad()
def _probe_tracker(in_probe: th.Tensor, model: nn.Sequential) -> tuple[th.Tensor, nn.Module]:
    probe = in_probe
    for layer in model:
        yield probe, layer
        probe = layer(probe)


@th.no_grad()
def make_decoders(input_shape: Sequence[int], model: nn.Sequential) -> Iterable[nn.Module]:
    # Create a probe to send through the model and catch at the other end.
    in_probe = th.empty((1, *input_shape))
    out_probe = model(in_probe)
    target_features = out_probe[0].flatten().shape[0]
    # Create a linear decoder for each layer, except the last one.
    return (*[unproj(probe.shape[1:], layer, target_features) for probe, layer in _probe_tracker(in_probe, model[:-1])],
            nn.Identity())


@th.no_grad()
def make_encoders(model: nn.Sequential) -> Iterable[nn.Module]:
    # Note: Lambda layers hold no parameters. So, we use them to hide the parameters
    # of the messenger, since its only job is to move inputs X into hidden space H and
    # its weights must remain frozen.
    return nn.Identity(), *[model[0:i] for i in range(1, len(model))]


def locally_optimize_parallel(model: nn.Sequential,
                        opt,
                        hidden_criterion: Callable[[th.Tensor, th.Tensor], th.Tensor],
                        criterion: Callable[[th.Tensor, th.Tensor], th.Tensor],
                        epochs: int,
                        loader: DataLoader) -> tuple[int, int, th.Tensor]:
    X, _ = next(iter(loader))
    in_shape = X[0].shape
    decoders = make_decoders(in_shape, model)
    encoders = make_encoders(model)
    criteria = [*[hidden_criterion for _ in model[:-1]], criterion]
    for epoch in range(1, epochs + 1):
        for idx, (encoder, layer, decoder, distance) in enumerate(zip(encoders, model, decoders, criteria)):
            encoder.requires_grad_(False)
            loss = th.tensor(0.)
            batches = 0
            opt.zero_grad()
            for X, Y in loader:
                with th.no_grad():
                    H_prior = encoder(X)
                H_after = decoder(layer(H_prior))
                loss += distance(H_after, Y)
                batches += 1
            loss.backward()
            opt.step()
            encoder.requires_grad_(True)
            if idx + 1 == len(model):
                yield epoch, loss.detach().cpu().item() / batches, encoders, decoders


def locally_optimize(model: nn.Sequential,
                     opt,
                     hidden_criterion: Callable[[th.Tensor, th.Tensor], th.Tensor],
                     criterion: Callable[[th.Tensor, th.Tensor], th.Tensor],
                     epochs: int,
                     loader: DataLoader) -> tuple[int, int, th.Tensor]:
    X, _ = next(iter(loader))
    in_shape = X[0].shape
    decoders = make_decoders(in_shape, model)
    encoders = make_encoders(model)
    criteria = [*[hidden_criterion for _ in model[:-1]], criterion]
    for idx, (encoder, layer, decoder, distance) in enumerate(zip(encoders, model, decoders, criteria)):
        encoder.requires_grad_(False)
        h_model = nn.Sequential(layer, decoder)
        for epoch in range(1, epochs + 1):
            loss = th.tensor(0.)
            batches = 0
            opt.zero_grad()
            for X, Y in loader:
                with th.no_grad():
                    H_prior = encoder(X)
                H_after = h_model(H_prior)
                loss += distance(H_after, Y)
                batches += 1
            loss.backward()
            opt.step()
            yield idx, epoch, loss.detach().cpu().item() / batches, encoders, decoders
        encoder.requires_grad_(True)


def backprop_optimize(model: nn.Module,
                      opt: Optimizer,
                      criterion: Callable[[th.Tensor, th.Tensor], th.Tensor],
                      epochs: int,
                      loader: DataLoader) -> tuple[int, th.Tensor]:
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = th.tensor(0.)
        batches = 0
        for X, Y in loader:
            loss += criterion(model(X), Y)
            batches += 1
        loss.backward()
        opt.step()
        yield epoch, loss.detach().cpu().item() / batches


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        assert callable(func)
        self.func = func

    def forward(self, x):
        return self.func(x)
