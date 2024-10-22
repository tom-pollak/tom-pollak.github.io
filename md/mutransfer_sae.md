# Applying μTransfer to Scale Sparse Autoencoders

> 26 September 2024

In this post, we'll explore how to apply [μTransfer](https://arxiv.org/abs/2203.03466) (Maximal Update Parameterization) to scale a Sparse Autoencoder (SAE), by increasing the hidden dimension (`d_sae`) of your SAE (i.e., increase the expansion factor). We want to apply μTransfer to ensure consistent training dynamics across different scales of the hidden dimension.

This work is heavily based on [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/) (please read this first). By applying μTransfer scaling to a model:

- **Consistent Training Dynamics**: Ensures activations, gradients, and weight updates remain consistent across expansion factors.
- **Simplified Scaling**: Only parameters connected to the scaled dimension (`d_sae`) are adjusted.
- **Stable Training**: Prevents issues like exploding activations / gradients.
- **Stable Hyperparameters**: No need to sweep learning rates, however this method doesn't solve regularization parameters.

> **TL;DR**: Jump to [Scaling Rules](https://tom-pollak.github.io/pages/mutransfer_sae.html#:~:text=Scaling%20Rules%20Summary) to find how to scale initialization & learning rates.

## Original SAE

Our SAE model, `JumpReLUSAE`, from [Gemma Scope from Scratch](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=8wy7DSTaRc90) (TODO: Gated SAE reference)

```python
import torch
import torch.nn as nn

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae, sigma=0.02):
        super().__init__()
        # Initialize parameters
        self.W_enc = nn.Parameter(torch.randn(d_model, self.d_sae) * sigma)
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, d_model) * sigma)
        self.threshold = nn.Parameter(torch.randn(self.d_sae) * sigma)
        self.b_enc = nn.Parameter(torch.randn(self.d_sae) * sigma)
        self.b_dec = nn.Parameter(torch.randn(d_model) * sigma)

    def encode(self, input\_acts):
        pre\_acts = input\_acts @ self.W_enc + self.b_enc
        mask = (pre\_acts > self.threshold)
        acts = mask * torch.relu(pre\_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, input\_acts):
        acts = self.encode(input\_acts)
        recon = self.decode(acts)
        return recon
```

- **`d_model`**: Fixed input and output dimension of the SAE.
- **`d_sae`**: Hidden dimension, determined by `d_model * expansion_factor`.

Our goal is to scale up the `expansion_factor`, (and therefore `d_sae`) and apply μTransfer principles to give us information on scaled learning rates and initialization.

## Applying μTransfer Scaling

### Definitions

- Input and Output Dimension (Fixed): $d_{\text{model}}$
- Expansion Factor: $\text{expansion\_factor}$
- Hidden Dimension: $d_{\text{sae}} = \text{expansion\_factor} \times d_{\text{model}}$
- Base Initialization Variance: $\sigma_{\text{base}}^2$
- Base Learning Rate: $\eta_{\text{base}}$
- **Width Multiplier**: $m_d = \text{expansion\_factor}\ /\ \text{expansion\_factor}_{\text{base}}$

### Scaling Principles

- Parameters connected to scaled dimensions: Adjust initialization variance and learning rate inversely with $m_d$
- Parameters connecting only to fixed dimensions: Keep the same.
- Output Scaling: Scale decoder's output by $\alpha_{\text{output}} = 1\ /\ m_d$.

---

[TODO I'm going to derive the initialization variance and learning rate for each of the weights based on the forwards and backwards pass]

## Decoder Weights (`W_dec`)

Dimensions: $d_{\text{sae}} \times d_{\text{model}}$

Reconstruction output:
$$
\text{recon} = \text{acts} \times W_{\text{dec}} + b_{\text{dec}}
$$

### Forward Pass

To maintain consistent training dynamics across different $d_{\text{sae}}$, we need the varience of $\text{recon}$ to remain constant. We follow the derivation similar to [Forward Pass at Initialization](https://blog.eleuther.ai/mutransfer/#forward-pass-at-initialization).

At initialization, $\text{acts}$ and $W_{\text{dec}}$ are independent and have zero mean.

$$
\text{Var}(\text{recon}) = \text{Var}(\text{acts} \times W_{\text{dec}}) = \text{Var}(\text{acts}) \times \text{Var}(W_{\text{dec}}) \times d_{\text{sae}}
$$

Since $d_{\text{sae}}$ scales with $m_d$:

$$
d_{\text{sae}} = m_d \times d_{\text{sae, base}}
$$

$$
\implies \text{Var}(\text{recon}) = \text{Var}(\text{acts}) \times \text{Var}(W_{\text{dec}}) \times (m_d \times d_{\text{sae, base}})
$$

To keep $\text{Var}(\text{recon})$ constant across scales, we need to scale $\text{Var}(W_{\text{dec}})$ inversely with $m_d$

$$
\text{Var}(W_{\text{dec}}) = \frac{\sigma^2_{\text{base}}}{m_d}
$$

$$
\implies \text{Var}(\text{recon}) = \text{Var}(\text{acts}) \times \sigma^2_{\text{base}} \times d_{\text{sae, base}}
$$

### Backwards Pass

The gradient with respect to $W_\text{dec}$:

$$
\nabla_{W_{\text{dec}}}\mathcal{L} = \text{acts}^\top \nabla_{\text{recon}}\mathcal{L}
$$

The magnitude of $\nabla_{W_{\text{dec}}}\mathcal{L}$ scales with $d_\text{sae}$ since $\text{acts}$ has dimension $d_\text{sae}$

To main consistent updates to $\Delta W_\text{dec}$, we scale the learning rate inversely with $m_d$:

$$
\eta_{W_\text{dec}} = \frac{\eta_\text{base}}{m_d}
$$

This aligns with the derivations in the [Effect of weight update on activations](https://blog.eleuther.ai/mutransfer/#effect-of-weight-update-on-activations).

### Output Scaling

Even with the above adjustments, [correlations between $\text{acts}$ and $W_\text{dec}$ develop during training](https://blog.eleuther.ai/mutransfer/#:~:text=Since%20we%20have,the%20complexity%20here), causing $\text{Var}(\text{recon})$ to grow with $m_d$

1. Only neurons with positive pre-activation inputs become active
2. During backpropagation, only active neurons contribute to weight updates
3. Certain neurons consistently activate for specific inputs, reinforcing the weight-activation relationship.
4. Due to the positive correlation, the variance of $\text{recon}$ increases.

$$
\text{Var}(\text{recon}) \propto m_d
$$

Therefore we apply a [scaling factor](https://blog.eleuther.ai/mutransfer/#effect-of-weight-update-on-activations:~:%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20text=To%2520ensure%2520proper%2520scales%2520of%2520activations%252C%2520the%2520output%2520logit%2520forward%2520pass%2520is%2520scaled&text=the%20output%20logit%20forward%20pass%20is%20scaled):

$$
\text{recon} = (\text{acts} \times W_\text{dec} + b_\text{dec}) \times \alpha_\text{output}, \quad \alpha_\text{output} = \frac{1}{m_d}
$$

## Encoder Weights (`W_enc`)

Dimensions: $d_{\text{model}} \times d_{\text{sae}}$

### Forward Pass

Encoder computes pre-activations:

$$
\text{pre\_acts} = \text{input\_acts} \times W_\text{enc} + b_\text{enc}
$$

Breaking down the matrix multiply:

$$
\text{pre\_acts}_i = \sum\limits^{d_\text{model}}_{k = 1}{\text{input\_acts}_k \times W_{\text{enc}, k, i} + b_{\text{enc}, i}}
$$

A.k.a each $\text{pre\_acts}$ is a sum over $d_\text{model}$ terms. Since $d_\text{model}$ is fixed, scaling $d_\text{sae}$ does _not_ affect the variance of $\text{pre\_acts}$

$$
\text{Var}(\text{pre\_acts}) = \text{Var}(\text{input\_acts}) \times \text{Var}(W_{\text{enc}}) \times d_{\text{model}}
$$

### Backwards Pass

While the forward pass variance remains constant, the backwards pass introduces scaling issues.

The gradient with respect to $W_\text{enc}$:

$$
\nabla_{W_{\text{enc}}}\mathcal{L} = \text{input\_acts}^\top \nabla_{\text{pre\_acts}}\mathcal{L}
$$

The gradient $\nabla_{\text{pre\_acts}}\mathcal{L}$ has dimensions affected by $d_\text{sae}$, causing the magnitude of $\nabla_{W_{\text{enc}}}\mathcal{L}$ increases with $m_d$.

To maintain consistent weight updates, we scale learning rate and initialization variance inversely with $m_d$:

$$
\text{Var}(W_\text{enc}) = \frac{\sigma^2_\text{base}}{m_d} \quad \eta_{W_\text{enc}} = \frac{\eta_\text{base}}{m_d}
$$

You can see more detail of this in Appendix 1 and [ElutherAI's backwards gradient pass](https://blog.eleuther.ai/mutransfer/#backward-gradient-pass-at-initialization)

## Encoder bias (`b_enc`) & `threshold`

$b_\text{enc}$ and $\text{threshold}$ are directly connected to $d_\text{sae}$. Therefore their activations and gradients scale with $m_d$ similar to $W_\text{dec}$

## Decoder Bias (`b_dec`)

Connects directly to $d_{\text{model}}$, no scaling is required.

## Scaling Rules Summary

$$
m_d = \text{expansion\_factor}\ /\ \text{expansion\_factor}_\text{base}
$$

| Parameter                          | Initialization Variance                                 | Learning Rate            |
| -----------------------------------| --------------------------------------------------------| -------------------------|
| Encoder Weights ($W_{\text{enc}}$) | $\sigma^2_\text{base} / m_d$                            | $\eta_\text{base} / m_d$ |
| Decoder Weights ($W_{\text{dec}}$) | $\sigma^2_\text{base} / m_d$                            | $\eta_\text{base} / m_d$ |
| Encoder Bias ($b_{\text{enc}}$)    | $\sigma^2_\text{base} / m_d$                            | $\eta_\text{base} / m_d$ |
| threshold                          | $\sigma^2_\text{base} / m_d$                            | $\eta_\text{base} / m_d$ |
| Decoder Bias ($b_{\text{dec}}$)    | $\sigma^2_{\text{base}}$                                | $\eta_{\text{base}}$     |
| Output Scaling                     | Multiply output by $\alpha_{\text{output}} = 1 / {m_d}$ | N/A                      |

## Updated SAE with μTransfer Scaling

```python
import torch
import torch.nn as nn
import math

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, expansion_factor, sigma_base=0.02, expansion_factor_base=None):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.expansion_factor_base = expansion_factor_base if expansion_factor_base is not None else expansion_factor
        self.m_d = self.expansion_factor / self.expansion_factor_base  # Width multiplier
        self.d_sae = int(self.expansion_factor * self.d_model)
        self.alpha_output = 1 / self.m_d

        # Scale initialization variance inversely with m_d
        sigma_scaled = sigma_base / math.sqrt(self.m_d)

        # Initialize parameters
        self.W_enc = nn.Parameter(torch.randn(d_model, self.d_sae) * sigma_scaled)
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, d_model) * sigma_scaled)
        self.threshold = nn.Parameter(torch.randn(self.d_sae) * sigma_scaled)
        self.b_enc = nn.Parameter(torch.randn(self.d_sae) * sigma_scaled)
        self.b_dec = nn.Parameter(torch.randn(d_model) * sigma_base)

    def encode(self, input\_acts):
        pre\_acts = input\_acts @ self.W_enc + self.b_enc
        mask = (pre\_acts > self.threshold)
        acts = mask * torch.relu(pre\_acts)
        return acts

    def decode(self, acts):
        recon = acts @ self.W_dec + self.b_dec
        recon = recon * self.alpha_output  # Apply output scaling
        return recon

    def forward(self, input\_acts):
        acts = self.encode(input\_acts)
        recon = self.decode(acts)
        return recon

d_model = 768  # GPT-2
expansion_factor_base = 16
expansion_factor = 32
sigma_base = 0.02
lr_base = 1e-3

sae = JumpReLUSAE(d_model, expansion_factor, sigma_base, expansion_factor_base)

scaled_params = [sae.W_enc, sae.W_dec, sae.b_enc, sae.threshold]
fixed_params = [sae.b_dec]

optimizer = torch.optim.AdamW([
    {'params': scaled_params, 'lr': lr_base / sae.m_d},
    {'params': fixed_params, 'lr': lr_base}
])
```

## References

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/)

## Appendices

### Appendix 1: Deriving $W_\text{enc}$ backwards pass

The gradient with respect to $W_\text{enc}$:

$$
\nabla_{W_{\text{enc}}}\mathcal{L} = \text{input\_acts}^\top \nabla_{\text{pre\_acts}}\mathcal{L}
$$


- $\text{input\_acts}^{\top}$ has dimensions $d_\text{model} \times \text{batch\_size}$.
- $\nabla_{\text{pre\_acts}}\mathcal{L}$ has dimensions $\text{batch\_size} \times d_\text{sae}$.
- $\implies \nabla_{W_\text{enc}}\mathcal{L}$ has dimensions $d_\text{model} \times d_\text{sae}$.

Gradient w.r.t $\text{pre\_acts}$:

$$
\nabla_{\text{pre\_acts}}\mathcal{L} = \nabla_{\text{acts}}\mathcal{L} \odot \nabla_{\text{pre\_acts}}\text{acts}
$$

- $\nabla_{\text{pre\_acts}}\text{acts}$ is a binary mask of active neurons.


Variance of $\nabla_{\text{pre\_acts}}\mathcal{L}$:

$$
\text{Var}(\nabla_{\text{pre\_acts}}\mathcal{L}) = \text{Var}(\nabla_{\text{acts}}\mathcal{L}) \times p_\text{active}
$$

- $p_\text{active}$ is the probability that a neuron is active (approximately constant).

#### Dependence on Decoder Weights ($W_\text{dec}$)

$$
\nabla_{\text{acts}}\mathcal{L} = \nabla_{\text{recon}}\mathcal{L} \times W_\text{dec}^{\top}
$$

- $W_\text{dec}^{\top}$ has dimensions $d_\text{model} \times d_\text{sae}$.

- As $d_\text{sae}$ increases, $W_\text{dec}$ becomes larger, affecting the variance of $\nabla_{\text{acts}}\mathcal{L}$ and consequently $\nabla_{\text{pre\_acts}}\mathcal{L}$.

#### Variance of Each Element in $\nabla_{W_\text{enc}}\mathcal{L}$

- Each element of $\nabla_{W_\text{enc}}\mathcal{L}$ is computed as:

$$
[\nabla_{W_\text{enc}}\mathcal{L}]_{ij} = \sum_{n=1}^{\text{batch\_size}} \text{input\_acts}_{ni} \times [\nabla_{\text{pre\_acts}}\mathcal{L}]_{nj}
$$

- The variance of each element depends on $\text{Var}(\text{input\_acts})$ and $\text{Var}(\nabla_{\text{pre\_acts}}\mathcal{L})$.

- Since $\text{Var}(\nabla_{\text{pre\_acts}}\mathcal{L})$ increases with $d_\text{sae}$ (due to $W_\text{dec}$), the variance of each element in $\nabla_{W_\text{enc}}\mathcal{L}$ also increases with $d_\text{sae}$.

#### Scaling Total Number of Elements

Total number of elements in $\nabla_{W_\text{enc}}\mathcal{L}$ is $d_\text{model} \times d_\text{sae}$ (derived above).

- As $d_\text{sae}$ increases, the total number of elements increases proportionally, scaling the magnitude of the gradient.

