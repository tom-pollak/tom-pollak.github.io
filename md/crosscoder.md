# Crosscoder

> These are some rough notes for Anthropic's [Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html)

Residual stream is _linear_ and _additive_, can be though of as different equivalent graphs
- With an extra edge that allows earlier layers to infulence later layers

Circuits can be split across two layers, but in parallel.
- If the model has more layers than the length of the circuit


IF features are represented across multiple layers, we should be able to apply dictionary learning to them _jointly_.
- This is the setup for a crosscoder!

Crosscoder can also help when a feature stays in the residual stream for _many layers_
- If we tried to understand this in terms of a residual stream at every layer, we'd have duplicated feats across layers.


Crosscoders allow attribution of layer 1s features to layer 10 without attributing through a _chain of layers_

## Acausal Crosscoder formulization

$$
f(x_j) = \text{ReLU}( W_{l_1}^{\text{enc}} a_{l_1}(x_j) + W_{l_2}^{\text{enc}} a_{l_2}(x_j) + W_{l_3}^{\text{enc}} a_{l_3}(x_j) + b_{\text{enc}} )
$$

Where each $W_{\text{enc}}$ projects to the same dimension and is _independently additive_
- This is where the causal relevance comes in!

### Loss / L1 vs L2

L1 allows comparison with SAE and transcoders

L2 makes more sense if we want to look at all the layers together as a single vector (like an SAE is)
- However encourages splitting across multiple layers, so not compatiable with SAEs etc.

_Seperately_ normalize the activations of each layer, so each layer contributes to the loss.

---

**Causality:** do we want earlier layers to predict later layers (not present in acausal)

Modeling residual stream or layer outputs?


Difference with [Shared SAEs](https://arxiv.org/pdf/2103.15949) (train a single SAE for multiple layers of residual stream
- Doesn't allow for features to rotate (which apparently they do? why?)

---

#### Crosscoders are more 8x feature efficient than per-layer SAEs

Per-layer:
- $F$ features for each $L$ layers, therefore $L \times F$ features in total

Crosscoder
- $F$ features, but trained with $L$ times as many FLOPS

Indicates that there is a significant number of linearly correlated features across layers, which are interpreted by the crossencoder as cross-layer features.


#### Crosscoders achieve 15-40x lower L0 than per-layer

Similar perf could be found with per-layer SAEs post-hoc by clustering features based on activation similarity, however crosscoders "bake in" this clustering at training time.

## Feature Analysis

#### Are crosscoder features localized to a few layers?

Most features peak in strength in a particular layer, and decay in later layers.

- Similar to Shared SAEs, which assumes features same if it is represented by the same direction across layers
- Crosscoder assumes same if it activates on the same data points across layers
  - _Allows feature directions to change across layers_ ("feature drift")

##### Gradual formation of features distributed across layers evidence for crossencoder superposition?

Could also be that a feature is _created_ at one layer and amplified at the next.

#### Do feature decoder directions remain consistent across layers?

Wide variety: some features definitely rotate _while maintaining magnitude_

**Decay:** Magnitude Peaks in a certain layer, and trails off.


##### Rotation

No rotation and no decay: Entirely active square.

Rotates and maintians: Aka in the plot, the diagnonal is very active, while off-diagnoal is low. Each layers features have low cosine similarity with each other, but the magnitude is still there.

Overall, most feature decoder directions are stable across layers, even when decoder norms are strong (no decay).

Preliminary research: crosscoder features that peak in a particular layer are qualitatively similar to features obtained by SAEs on that layer.

## Masked Crosscoders

### Causality Experiments

**Weakly Causal:** encoder reads from layer $i$ alone, and decoder attempts to reconstruct layer $i$ and all subsequent layers.

**Cross-layer transcoder:** feature reads from residual stream at layer $L$ and predicts the output of all later layers.

### Pre/Post MLP Crosscoders

Find new features in the MLP output.

1. Train a crosscoder on the pre-MLP stream and outputs that the MLP writes back to the residual stream.
2. Read only encoding of pre-MLP, but reconstruct both pre-MLP and MLP output.

Different from a transcoder, reconstruct both MLP _input_ & output

- Identifies features shared in pre and post MLP space, and features specific to one or the other.
- Because encoder vectors are only in the pre-MLP space, we can analyze how these newly computed features where made

How to find inputs to newly constructed features

1. Dot product of feature encoder vector with feature decoded vector, weighted by source feature activation.

Example:

Post-MLP feature fires on uniqueness: "special" "particular" "exceptional"
Pre-MLP inputs fire for particular words in particular contexts.

##### How are "stable" features (no decay) embedded in pre and post MLP space?

Postive correlation on average, aka similar directions, but high variance.
- May explain why feature directions drift across layers -- MLP relay features without non-linear xfm (perhaps so other features can read from different spaces?) aka the particular words in context are written written to different subspaces for different later features to read.

## Model Diffing

> Cross Model features!

Previous work:
- [Universal neurons](https://distill.pub/2020/circuits/zoom-in): edge detectors, high low frequency detections in vision models

Analogous circuits?

Model diffing: finetuned models -- finding what has changed from previous model. Could be interesting for my [LoRA](https://github.com/tom-pollak/interp-lora-causal-circuits) work?

- **Feature Set Comparison:** Universal features vs unique to specific model
- **Circuit Comparison:** Even for universal features, downstream effects may be different. Compare circuits they participate in.
- **Differences in Feature Geometry:** Compare cosine similarity across models. If variants of the same model, we can find the absolute difference aswell (aka drift)

