# Applying μTransfer to Scale Sparse Autoencoders

> 26 September 2024

In this post, we'll explore how to apply [μTransfer](https://arxiv.org/abs/2203.03466) (Maximal Update Parameterization) to scale a Sparse Autoencoder (SAE), by increasing the hidden dimension (`d_sae`) of your SAE (i.e., increase the expansion factor). We want to apply μTransfer to ensure consistent training dynamics across different scales of the hidden dimension.

This work is heavily based on [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/) (please read this first). By applying μTransfer scaling to a model:

- **Consistent Training Dynamics**: Ensures activations, gradients, and weight updates remain consistent across expansion factors.
- **Simplified Scaling**: Only parameters connected to the scaled dimension (`d_sae`) are adjusted.
- **Stable Training**: Prevents issues like exploding activations / gradients.

## Original SAE

Our SAE model, `JumpReLUSAE`, taken from [Gemma Scope from Scratch](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=8wy7DSTaRc90)

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

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, input_acts):
        acts = self.encode(input_acts)
        recon = self.decode(acts)
        return recon
```

- **`d_model`**: Fixed input and output dimension of the SAE.
- **`d_sae`**: Hidden dimension, determined by `d_model * expansion_factor`.

Our goal is to scale up the `expansion_factor`, (and therefore `d_sae`) and apply μTransfer principles to give us information on scaled learning rates and initialization.

## Applying μTransfer Scaling

### Definitions

- Input and Output Dimension (Fixed): $d_{\text{model}}$
- Expansion Factor: $\text{expansion_factor}$
- Hidden Dimension: $d_{\text{sae}} = \text{expansion_factor} \times d_{\text{model}}$
- Width Multiplier: $m_d = \frac{\text{expansion_factor}}{\text{expansion_factor}_{\text{base}}}$
- Base Initialization Variance: $\sigma_{\text{base}}^2$
- Base Learning Rate: $\eta_{\text{base}}$

### Scaling Principles

- Parameters connected to scaled dimensions: Adjust initialization variance and learning rate inversely with $m_d$
- Parameters connecting only to fixed dimensions: Keep the same.
- Output Scaling: Apply an output scaling factor to maintain consistent variance in the output. Specifically, multiply the decoder's output by $\alpha_{\text{output}} = \frac{1}{m_d}$.

## Derivations for each parameter

### Decoder Weights (`W_dec`)

> Starting with the decoder weights, since these are most interesting

Dimensions: $d_{\text{sae}} \times d_{\text{model}}$

Reconstruction output:
$$
\text{recon} = \text{acts} \times W_{\text{dec}} + b_{\text{dec}}
$$

#### Variance of Reconstruction

To maintain consistent training dynamics across different $d_{\text{sae}}$, we need to ensure variance of $\text{recon}$ remains constant.

At initialization, $\text{acts}$ and $W_{\text{dec}}$ have 0 mean and are independent.

$$
\text{Var}(\text{recon}) = \text{Var}(\text{acts} \times W_{\text{dec}}) = \text{Var}(\text{acts}) \times \text{Var}(W_{\text{dec}}) \times d_{\text{sae}}
$$

Since $d_{\text{sae}}$ scales with $m_d$ $\quad (d_{\text{sae}} = m_d \times d_{\text{sae, base}})$

$$
\implies \text{Var}(\text{recon}) = \text{Var}(\text{acts}) \times \text{Var}(W_{\text{dec}}) \times (m_d \times d_{\text{sae, base}})
$$

To keep $\text{Var}(\text{recon})$ constant across scales, we need to scale $\text{Var}(W_{\text{dec}})$ inversely with $m_d$

$$
\text{Var}(W_{\text{dec}}) = \frac{\sigma^2_{\text{base}}}{m_d}
$$

Substituting in:

$$
\text{Var}(\text{recon}) = \text{Var}(\text{acts}) \times \left(\frac{\sigma^2_{\text{base}}}{m_d}\right) \times (m_d \times d_{\text{sae, base}})
$$

$$
\implies \text{Var}(\text{recon}) = \text{Var}(\text{acts}) \times \sigma^2_{\text{base}} \times d_{\text{sae, base}}
$$

#### Scaling Learning Rate

Update to $W_{\text{dec}}$ with SGD:

$$
\Delta W_{\text{dec}} = -\eta W_{\text{dec}} \nabla_{W_{\text{dec}}}\mathcal{L}
$$

- $\eta W_{\text{dec}}$: Learning rate of $W_{\text{dec}}$
- $\nabla_{W_{\text{dec}}}\mathcal{L}$: Gradient of loss w.r.t $W_{\text{dec}}$

$$
\nabla_{W_{\text{dec}}}\mathcal{L} = \text{acts}^\top \nabla_{\text{recon}}\mathcal{L}
$$

Since $\text{acts}$ is of size $d_\text{sae}$, scaling $d_\text{sae}$ affects the magnitude of the gradient. To keep the magnitude constant we can scale the learning rate inversely with $m_d$, which should compensate for the increase in gradient magnitude.

$$
\eta_{W_\text{dec}} = \frac{\eta_\text{base}}{m_d}
$$

#### Output Scaling

Even after adjusting initialization variance and learning rate of $W_\text{dec}$, there is an additional factor that can cause variance to increase during training -- [the development of correlations between weights and activations.](https://blog.eleuther.ai/mutransfer/#:~:text=Since%20we%20have,the%20complexity%20here)

1. Only neurons with positive pre-activation inputs become active
2. During backpropagation, only active neurons contribute to weight updates
3. Certain neurons consistently activate for specific inputs, reinforcing the weight-activation relationship.
4. Due to the positive correlation, the variance of $\text{recon}$ increases.

Using the above steps, we can see how reconstruction variance is proportional to $m_d$. With a larger $m_d$, there are more parameters contributing to correlation and increase in variance.

$$
\text{Var}(\text{recon}) = \text{Var}(\text{acts} \times W_{\text{dec}}) = \text{Var}(\text{acts}) \times \text{Var}(W_{\text{dec}}) \times d_{\text{sae}}
$$
$$
\implies \text{Var}(\text{recon}) \propto m_d
$$

Therefore $\text{Var}(\text{recon})$ grows as $m_d$ increases, scaling inversely $m_d$ helps control the output variance.

$$
\text{recon} = (\text{acts} \times W_\text{dec} + b_\text{dec}) \times \alpha_\text{output}, \quad \alpha_\text{output} = \frac{1}{m_d}
$$

### Encoder Weights (`W_enc`)

Dimensions: $d_{\text{model}} \times d_{\text{sae}}$

#### Variance of Pre-Activations

$$
\text{Var}(\text{pre_acts}) = \text{Var}(\text{input_acts}) \times \text{Var}(W_{\text{enc}}) \times d_{\text{model}}
$$

The variance of $\text{pre_acts}$ is not affected by scaling $d_{\text{sae}}$, therefore no scaling is needed. Similarly, the gradient magnitudes are not dependent on $d_\text{model}$ 

### Encoder bias (`b_enc`) & `threshold`

Similar to `W_enc`, these parameters connect to the encoder's $\text{pre_act}$, which do not depend on $d_{\text{sae}}$. Therefore no scaling is needed.

### Decoder Bias (`b_dec`)

Connects directly to $d_{\text{model}}$, no scaling needed

## Scaling Rules Summary

| Parameter                              | Initialization Variance                                     | Learning Rate                    |
| -------------------------------------- | ----------------------------------------------------------- | -------------------------------- |
| Encoder Weights ($W_{\text{enc}}$)     | $\sigma^2_{\text{base}}$                                    | $\eta_{\text{base}}$             |
| **Decoder Weights ($W_{\text{dec}}$)** | $\frac{\sigma^2_{\text{base}}}{m_d}$                        | $\frac{\eta_{\text{base}}}{m_d}$ |
| Encoder Bias ($b_{\text{enc}}$)        | $\sigma^2_{\text{base}}$                                    | $\eta_{\text{base}}$             |
| ($\text{threshold}$)                   | $\sigma^2_{\text{base}}$                                    | $\eta_{\text{base}}$             |
| Decoder Bias ($b_{\text{dec}}$)        | $\sigma^2_{\text{base}}$                                    | $\eta_{\text{base}}$             |
| **Output Scaling**                     | Multiply output by $\alpha_{\text{output}} = \frac{1}{m_d}$ | N/A                              |

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

        sigma_fixed = sigma_base
        sigma_scaled = sigma_base / math.sqrt(self.m_d)

        # Initialize parameters
        self.W_enc = nn.Parameter(torch.randn(d_model, self.d_sae) * sigma_fixed)
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, d_model) * sigma_scaled)
        self.threshold = nn.Parameter(torch.randn(self.d_sae) * sigma_fixed)
        self.b_enc = nn.Parameter(torch.randn(self.d_sae) * sigma_fixed)
        self.b_dec = nn.Parameter(torch.randn(d_model) * sigma_fixed)

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        recon = acts @ self.W_dec + self.b_dec
        recon = recon * self.alpha_output  # Apply output scaling
        return recon

    def forward(self, input_acts):
        acts = self.encode(input_acts)
        recon = self.decode(acts)
        return recon

d_model = 768  # GPT-2
expansion_factor_base = 16
expansion_factor = 32
sigma_base = 0.02
lr_base = 1e-3

sae = JumpReLUSAE(d_model, expansion_factor, sigma_base, expansion_factor_base)

scaled_params = [sae.W_dec]
fixed_params = [sae.W_enc, sae.b_enc, sae.threshold, sae.b_dec]

optimizer = torch.optim.AdamW([
    {'params': scaled_params, 'lr': lr_base / sae.m_d},
    {'params': fixed_params, 'lr': lr_base}
])
```

## References

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/)
