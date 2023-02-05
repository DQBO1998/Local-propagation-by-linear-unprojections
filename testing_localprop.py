import torch as th
from functorch import vmap
from torch import nn
from torch.linalg import norm
from torch.nn.functional import cosine_similarity, normalize
from torch.utils.data import Dataset
from localprop import *
import matplotlib.pyplot as plt
from IPython import display


@th.no_grad()
def average_loss(comp: Callable[[th.Tensor, th.Tensor], th.Tensor], loader: DataLoader):
    losses = tuple(comp(X, Y).detach().cpu().item() for X, Y in loader)
    return sum(losses) / len(losses)


@th.no_grad()
def plot_LP_progress(encoders,
                     layers,
                     decoders,
                     epoch,
                     epochs,
                     layer_idx,
                     losses: Sequence | None,
                     tests: Sequence | None = None):
    star, no_star = '*', ''
    width = ceil(sqrt(len(layers) + 1))
    height = width
    while width * (height - 1) >= len(layers) + 1:
        height -= 1
    fig, axes = plt.subplots(height, width)
    axes = axes.flatten()
    for idx, (ax, encoder, layer, decoder) in enumerate(zip(axes[:-1], encoders, layers, decoders)):
        X, Y = decision_boundary(nn.Sequential(encoder, layer, decoder), (-1, 1), resolution=60)
        ax.imshow(Y, cmap='plasma', origin='lower')
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$x_1$')
        ax.set_aspect('equal')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_title(f'Layer {idx} | {star if layer_idx == idx else no_star}')
    ax_plt = axes[-1]
    if losses:
        ax_plt.plot(range(1, epoch + 1), losses, label=f'Loss {round(losses[-1], 4)}')
    if tests:
        ax_plt.plot(range(1, epoch + 1), tests, label=f'Test loss {round(tests[-1], 4)}')
    ax_plt.set_xlabel('Epoch')
    ax_plt.set_ylabel('Loss')
    ax_plt.set_xlim((1, epochs))
    if losses or tests:
        ax_plt.legend()
    plt.figtext(0, -0.2, f'Epoch {epoch} / {epochs}')
    plt.tight_layout()
    plt.show()
    if epoch < epochs:
        fig.clf()
    display.clear_output(True)


@th.no_grad()
def plot_BP_progress(model: nn.Module,
                     epoch: int,
                     epochs: int,
                     loss_hist: Sequence,
                     dataset_name: str,
                     test_hist: Sequence | None = None,
                     plot_decision_boundary: bool = True):
    fig, (axs_db, axs_loss) = plt.subplots(2, 1)
    axs_loss.plot(range(1, epoch + 1), loss_hist, label=f'Loss {round(loss_hist[-1], 4)}')
    if test_hist:
        axs_loss.plot(range(1, epoch + 1), test_hist, label=f'Test loss {round(test_hist[-1], 4)}')
    axs_loss.set_xlabel('Epoch')
    axs_loss.set_ylabel('Loss')
    axs_loss.set_xlim((1, epochs))
    axs_loss.legend()
    # Create subplot for decision boundaries.
    if plot_decision_boundary:
        X, Y = decision_boundary(model, (-1, 1), resolution=500)
        axs_db.imshow(Y, cmap='plasma', origin='lower')
    axs_db.set_xlabel(r'$x_0$')
    axs_db.set_ylabel(r'$x_1$')
    axs_db.set_aspect('equal')
    axs_db.xaxis.set_ticklabels([])
    axs_db.yaxis.set_ticklabels([])
    # Display and cleanup.
    text = f'\nOn {dataset_name}  with Back-prop'
    plt.figtext(0, -0.2, text)
    plt.tight_layout()
    plt.show()
    if epoch < epochs:
        fig.clf()
    display.clear_output(True)


@th.no_grad()
def decision_boundary(model: nn.Module, xrange, resolution: int = 100):
    X1s, X0s = th.meshgrid(th.linspace(xrange[0], xrange[1], resolution),
                           th.linspace(xrange[0], xrange[1], resolution))
    X = th.stack((X0s, X1s), dim=2)
    Y = vmap(lambda x_arr: vmap(lambda x: model(x.view(1, 2))[0])(x_arr))(X)
    return X, Y


def _signed_xor(noise: float = 0., variance: float = 1., batch_size: int = 100):
    while True:
        # Sample from input domain.
        X = (th.rand(batch_size, 2) - 0.5) * variance
        # Assign classes.
        c1 = th.tensor([1.])
        c2 = th.tensor([0.])
        rule = lambda x: x[0] * x[1] >= 0.
        Y = vmap(lambda x: th.where(rule(x), c1, c2))(X)
        # Apply noise to X.
        X = X + th.randn(batch_size, 2) * noise
        # Output batch.
        yield X, Y


class SignedXOR(Dataset):

    def __init__(self, noise, variance, samples):
        generator = _signed_xor(noise=noise, variance=variance, batch_size=samples)
        self.X, self.Y = next(generator)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


datasets = {
    'signed_xor': (lambda: SignedXOR(0., 2., 1000), lambda: SignedXOR(0., 2., 1000))
}

descriptions = {
    'signed_xor': 'Given a point $\\vec{x} \\in \\mathbb{R}^2$, if $x_1 \\cdot x_2 \\geq '
                  '0$, assign class 1. Otherwise, assign class 0. '
}
