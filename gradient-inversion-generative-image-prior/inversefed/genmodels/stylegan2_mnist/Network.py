import math
# from typing import Turple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class MappingNetwork(nn.Module):
    """Create mapping network that project latent space to style space"""

    def __init__(self, features, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = F.normalize(z, dim=1)
        return self.net(z)


class Generator(nn.Module):
    def __init__(self, log_resolution, d_latent, n_features=32, max_features=512):
        super().__init__()
        # Set number of features for each layer
        features = [min(max_features, n_features * 2 * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_block = len(features)
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        # Generator blocks
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_block)]
        self.blocks = nn.ModuleList(blocks)

        self.up_sample = UpSample()

    def get_noise(self, batch_size):
        noise = []
        resolution = 4

        for i in range(self.n_block):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size,1,resolution, resolution, device='cuda')
            n2 = torch.randn(batch_size, 1, resolution, resolution, device='cuda')

            noise.append([n1,n2])

            resolution *= 2
        return noise

    def forward(self, w):
        batch_size = w.shape[1]
        input_noise = self.get_noise(batch_size)
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        for i in range(1, self.n_block):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new
        return rgb


class GeneratorBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.ToRGB = ToRGB(d_latent, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.ToRGB(x, w)
        return x, rgb


class ToRGB(nn.Module):
    def __init__(self, d_latent, features):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        grayscale = True
        if grayscale:
            self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False)
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
            self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)



    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class StyleBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)

        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    def __init__(self, log_resolution, n_features = 64, max_features = 512):
        super().__init__()
        grayscale = True
        if grayscale:
            self.from_rgb = nn.Sequential(EqualizedConv2d(1, n_features, 1), nn.LeakyReLU(0.2, True))
        else:
            self.from_rgb = nn.Sequential(EqualizedConv2d(1, n_features, 3), nn.LeakyReLU(0.2, True))
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x):
        # x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(DownSample(), EqualizedConv2d(in_features, out_features, kernel_size=1))
        self.block = nn.Sequential(EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2, True),
                                   EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2, True))
        self.down_sample = DownSample()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim = 1)


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.smooth = Smooth()

    def forward(self, x):
        return self.smooth(self.up_sample(x))


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class Smooth(nn.Module):
    """
    Blur each channel
    """

    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]  # define the blue kernel
        kernel = torch.tensor([[kernel]], dtype=torch.float32)  # making it a pytorch tensor

        kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = nn.Parameter(kernel, requires_grad=False)  # fix the kernel so it doesn't get update
        self.pad = nn.ReplicationPad2d(1)  # pad the kernel

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)  # reshape
        x = self.pad(x)  # pad
        x = F.conv2d(x, self.kernel)  # smooth
        return x.view(b, c, h, w)  # reshape back and return


class EqualizedLinear(nn.Module):
    """
    This applies the equalized weight method for a linear layer
    input:
      input features
      output features
      bias
    return:
      y = wx+b

    """

    def __init__(self, in_features, out_features, bias=0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias)



class EqualizedWeight(nn.Module):
    """
    Introduced in https://arxiv.org/pdf/1710.10196.pdf
    Instead of draw weight from a normal distribution of (0,c),
    It draws from (0,1) instead, and times the constant c.
    so that when using optimizer like Adam which will normalize through
    all gradients, which may be a problem if the dynamic range of the weight
    is too large.

    input:
    shape: [in_features,out_features]

    return:
    Randomized weight for corresponding layer
    """

    def __init__(self, shape):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class DiscriminatorLoss(nn.Module):
    def forward(self,f_real, f_fake):
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(nn.Module):
    def forward(self,f_fake):
        return -f_fake.mean()


class GradientPenalty(nn.Module):
    """
    <a id="gradient_penalty"></a>

    ## Gradient Penalty

    This is the $R_1$ regularization penality from the paper
    [Which Training Methods for GANs do actually Converge?](https://papers.labml.ai/paper/1801.04406).

    $$R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{p_\mathcal{D}(x)}
    \Big[\Vert \nabla_x D_\psi(x)^2 \Vert\Big]$$

    That is we try to reduce the L2 norm of gradients of the discriminator with
    respect to images, for real images ($P_\mathcal{D}$).
    """

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        * `x` is $x \sim \mathcal{D}$
        * `d` is $D(x)$
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to $1$ since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm $\Vert \nabla_{x} D(x)^2 \Vert$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\Vert \nabla_x D_\psi(x)^2 \Vert$
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    """
    <a id="path_length_penalty"></a>

    ## Path Length Penalty

    This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude
    change in the image.

    $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
      \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$

    where $\mathbf{J}_w$ is the Jacobian
    $\mathbf{J}_w = \frac{\partial g}{\partial w}$,
    $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and
    $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.

    $a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
    as the training progresses.

    $\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
    $$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant $\beta$ used to calculate the exponential moving average $a$
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # Exponential sum of $\mathbf{J}^\top_{w} y$
        # $$\sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
        # where $[\mathbf{J}^\top_{w} y]_i$ is the value of it at $i$-th step of training
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate $y \in \mathcal{N}(0, \mathbf{I})$
        y = torch.randn(x.shape, device=device)
        # Calculate $\big(g(w) \cdot y \big)$ and normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / math.sqrt(image_size)

        # Calculate gradients to get $\mathbf{J}^\top_{w} y$
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculate L2-norm of $\mathbf{J}^\top_{w} y$
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate $a$
            # $$\frac{1}{1 - \beta^N} \sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            # $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
            # \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment $N$
        self.steps.add_(1.)

        # Return the penalty
        return loss
