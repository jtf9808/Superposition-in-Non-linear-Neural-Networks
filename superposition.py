import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import imageio


class DataGenerator:
    def __init__(self, dimensions=20, sparsity=0.0):
        self.dimensions = dimensions
        self.sparsity = sparsity

    def generate_data(self, n=1):
        data = []
        for i in range(n):
            row = []
            for j in range(self.dimensions):
                if random.uniform(0, 1) < (1.0 - self.sparsity):
                    row.append(random.uniform(0, 1))
                else:
                    row.append(0)
            data.append(row)
        return np.array(data).T


class LinearNet(nn.Module):

    name = "Linear Network"

    def __init__(self, m=20, n=5):
        super().__init__()

        self.w = nn.Parameter(torch.empty(n, m))
        self.b = nn.Parameter(torch.zeros((m, 1)))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        x = self.w @ x
        x = self.w.T @ x
        x = x + self.b
        return x


class NonLinearNet(LinearNet):

    name = "Non-linear Network"

    def forward(self, x):
        x = super().forward(x)
        return F.relu(x)


def importance_loss(output, target, importance):
    importances = torch.Tensor(
        [[importance ** (i + 1) for i in range(output.shape[0])]]
    )
    loss = torch.sum(importances @ (output.abs() - target) ** 2)
    return loss


def create_gif(image_folder, gif_name):
    images = []
    for file_name in sorted(
        os.listdir(image_folder), key=lambda x: int(x.split(".")[0])
    ):
        if file_name.endswith(".png"):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=0.5, loop=0)


if __name__ == "__main__":
    for sparsity in [0.0, 0.25, 0.5, 0.8, 0.9, 0.95]:
        for network_type in [LinearNet, NonLinearNet]:
            dg = DataGenerator(5, sparsity=sparsity)
            model = network_type(5, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            lr = 0.001
            for epoch in range(10000 + 1):
                if epoch != 0:
                    x_train = torch.Tensor(dg.generate_data(n=1024))
                    optimizer.zero_grad(set_to_none=True)
                    out = model(x_train)
                    loss = importance_loss(out, x_train, importance=0.7)
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch: {epoch}, Loss: {loss.item()}")

                if epoch % 100 == 0:
                    numpy_w = model.w.detach().numpy()
                    numpy_b = model.b.detach().numpy()

                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
                    # W.T W heatmap
                    cax1 = ax1.matshow(numpy_w.T @ numpy_w, vmin=-1, vmax=1)
                    fig.colorbar(cax1, ax=ax1)
                    ax1.set_title("$W^TW$", loc="left", fontsize=20)
                    ax1.set_xticks(range(numpy_w.shape[1]))
                    ax1.set_xticklabels(range(1, numpy_w.shape[1] + 1))
                    ax1.set_yticks(range(numpy_w.shape[1]))
                    ax1.set_yticklabels(range(1, numpy_w.shape[1] + 1))
                    # b heatmap
                    cax2 = ax2.matshow(numpy_b, vmin=-1, vmax=1)
                    ax2.xaxis.set_tick_params(which="both", top=False)
                    ax2.xaxis.set_tick_params(labeltop=False)
                    fig.colorbar(cax2, ax=ax2)
                    ax2.set_title("$b$", loc="left", fontsize=20)
                    ax2.set_yticks(range(numpy_w.shape[1]))
                    ax2.set_yticklabels(range(1, numpy_w.shape[1] + 1))
                    # W column vector plot
                    origin = np.array(
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
                    )  # origin point
                    max_size = np.max(numpy_w)
                    ax3.quiver(
                        *origin,
                        numpy_w.T[:, 0],
                        numpy_w.T[:, 1],
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                    )
                    ax3.set_xlim([-1.5, 1.5])
                    ax3.set_ylim([-1.5, 1.5])
                    ax3.set_title("$W_i$", loc="left", fontsize=20)
                    # W column vector magnitudes
                    w_mags = np.linalg.norm(numpy_w, axis=0)
                    ax4.barh(
                        [str(i) for i in reversed(range(1, w_mags.shape[0] + 1))],
                        w_mags[::-1],
                        color="skyblue",
                        edgecolor="black",
                    )
                    ax4.set_xlim([0, 1])
                    ax4.set_title("$\|W_i\|$", loc="left", fontsize=20)

                    fig.suptitle(
                        f"{model.name}, Sparsity: {sparsity:.2f}, Epoch: {epoch}",
                        fontsize=20,
                    )
                    plt.tight_layout()
                    plt.savefig(f"plots/{epoch}.png")
                    plt.close()

            create_gif(
                "plots",
                f'gifs/{model.name.lower().replace(" ", "_").replace("-", "_")}_sparsity_{sparsity:.2f}_plots.gif',
            )

            for f in os.listdir("plots"):
                os.remove(f"plots/{f}")
