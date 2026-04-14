## Cut Your Losses in Large-Vocabulary Language Models

This software project accompanies the research paper:
**[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)**,
*Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, and Philipp Krähenbühl*.

![](assets/cce_figure.png)

As language models grow ever larger, so do their vocabularies. This has shifted the memory footprint of LLMs during training disproportionately to one single layer: the cross-entropy in the loss computation. Cross-entropy builds up a logit matrix with entries for each pair of input tokens and vocabulary items and, for small models, consumes an order of magnitude more memory than the rest of the LLM combined. We propose Cut Cross-Entropy (CCE), a method that computes the cross-entropy loss without materializing the logits for all tokens into global memory. Rather, CCE only computes the logit for the correct token and evaluates the log-sum-exp over all logits on the fly. We implement a custom kernel that performs the matrix multiplications and the log-sum-exp reduction over the vocabulary in flash memory, making global memory consumption for the cross-entropy computation negligible. This has a dramatic effect. Taking the Gemma 2 (2B) model as an example, CCE reduces the memory footprint of the loss computation from 24 GB to 1 MB, and the total training-time memory consumption of the classifier head from 28 GB to 1 GB. To improve the throughput of CCE, we leverage the inherent sparsity of softmax and propose to skip elements of the gradient computation that have a negligible (i.e., below numerical precision) contribution to the gradient. Experiments demonstrate that the dramatic reduction in memory consumption is accomplished without sacrificing training speed or convergence.

## Getting started

**Requirements**

1. Python 3.9+
2. PyTorch 2.4+
3. Triton 3.0+
4. Ampere (or newer) GPU


**Note:**  For operating systems that are not supported by Triton (e.g., MacOS), we include a highly optimized version of
linear-cross-entropy using `torch.compile`. This implementation will be set to the default on MacOS.

### Basic usage

**Installation**
```bash
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
```

**Usage**

Given a model loss computation that looks like the following,
```python
import torch.nn.functional as F

embeddings = model.compute_embedding(inputs)
classifier = model.get_classifier_weights()

logits = embeddings @ classifier.T

loss = F.cross_entropy(logits.float(), labels)
```

you can instead compute the loss as follows,

```python
from cut_cross_entropy import linear_cross_entropy

embeddings = model.compute_embedding(inputs)
classifier = model.get_classifier_weights()

# Note: There is no need to upcast embeddings or classifier to float32
# like you need to do with logits when using F.cross_entropy.
# The CCE kernel will automatically use fp32 for operations that are unstable
# in bf16/fp16.
loss = linear_cross_entropy(embeddings, classifier, labels)
```

In causal language modeling, it is common that the model embeddings and labels need to be shifted
such that the model predicts the next token.

```python
from cut_cross_entropy import linear_cross_entropy

embeddings = model.compute_embedding(inputs)
classifier = model.get_classifier_weights()

shift_embeddings = embeddings[..., :-1, :].flatten(0, -2)
shift_labels = labels[..., 1:]

manual_shift_loss = linear_cross_entropy(shift_embeddings, classifier, shift_labels)
```

Instead, pass `shift=1` to perform this computation without allocating the shift_embeddings matrix.
```python
from cut_cross_entropy import linear_cross_entropy

embeddings = model.compute_embedding(inputs)
classifier = model.get_classifier_weights()

# This is the same as manual_shift_loss above
auto_shift_loss = linear_cross_entropy(embeddings, classifier, labels, shift=1)
```

We also provide a highly optimized implementation of linear-cross-entropy loss using `torch.compile`.
This is a good option
for scenarios where speed is the primary goal and the model has a relatively small vocabulary compared to its
hidden dimension (when |V| >> D, `cce` will both save memory _and_ be faster).
This option also works on the CPU and older GPUs, making it useful for testing.

```python
from cut_cross_entropy import linear_cross_entropy

embeddings = model.compute_embedding(inputs)
classifier = model.get_classifier_weights()

loss = linear_cross_entropy(embeddings, classifier, labels, ..., impl="torch_compile")
```


There are several other implementations available depending on your needs.

| impl | Description |
|------|-------------|
| cce  | The CCE implementation as described in the paper. This is may be the fastest and uses the least amount of memory. Generally recommended to start here. |
| torch_compile | A highly optimized `torch.compile` implementation. This is typically the fastest but uses the most amount of memory. Good as a reference and for systems that don't support Triton. |
| cce_kahan | Uses Kahan summation (or fp32) to improve numerical precision. This comes at the cost of more memory usage (albeit only a temporary buffer in the backward pass). This is useful for long sequence lengths or if the model is particularly sensitive to numerical imprecision.
| cce_kahan_full_c | Same as cce_kahan and removes gradient filtering on the classifier gradient. This is useful for pretraining but will be slower.
| cce_kahan_full_c_full_e (cce_exact) | This additionally removes gradient filtering from the embedding gradient. This is useful as a reference point/sanity check. |


### Vocabulary Parallelism

We also support computing linear cross-entropy loss for classifier weights sharded
along the vocabulary dimensions. To use this, provided a `VocabParallelOptions` instance
to `linear_cross_entropy`. This takes 3 parameters, the `start` and `stop` indices of this rank's
shard, and the `torch.distributed.ProcessGroup` for this rank's vocab parallel group.



```python
import torch

from cut_cross_entropy import linear_cross_entropy, VocabParallelOptions

# The vocab parallel group for this rank.
#  This group can be created/retrieved in many different ways,
# for instance,
# torch.distributed.new_group(...)
# device_mesh.get_group(mesh_dim="model_parallel")
# etc
vp_group = ...


embeddings = model.compute_embedding(inputs)
vp_classifier = model.get_classifier_weights()

vp_start, vp_stop = model.get_classifier_range()
vp_opts = VocabParallelOptions(vp_start, vp_stop, group=vp_group)

# alternatively, there is an option to create this
# by linearly dividing the vocab across ranks
vp_opts = VocabParallelOptions.from_vocab(model.vocab_size, group=vp_group)

# All ranks in the vocab parallel group will return the same loss
loss = linear_cross_entropy(embeddings, vp_classifier, labels, ...,
  vocab_parallel_options=vp_opts)

loss.backward()

# All ranks will compute the same embeddings.grad, but each rank will have only the classifier gradient
# corresponding to its part of the full classifier matrix (as defined by vp_classifier).
```



### Computing Related Quantities

`linear_cross_entropy` can be used as an efficient way to compute the negative log likelihood
of a specified token. This can be used to compute various quantities.


```python
from cut_cross_entropy import linear_cross_entropy


# linear_cross_entropy computes negative log likelihood for a target token
nll = linear_cross_entropy(embeddings, classifier, target_token, reduction="none")

# Perplexity
ppl = torch.exp(nll.mean(-1))

# DPO (beta and reference omitted)
dpo_loss = -F.logsigmoid(nll[dispreferred].sum(-1) - nll[preferred].sum(-1))

# PPO
ppo_loss = -torch.minimum(toch.exp(-nll - old_logp) * adv, adv + eps * adv.abs())
```


### Z Loss

`linear_cross_entropy` can also be used to compute Z loss (a loss on the logsumexp).

```python
from cut_cross_entropy import linear_cross_entropy

loss, lse = linear_cross_entropy(embeddings, classifier, labels, ..., return_lse=True)

z_loss = lse.pow(2).mean()

# We also have a helper function to compute Z loss that will automatically remove ignored tokens/etc.
from cut_cross_entropy.utils import compute_z_loss

z_loss = compute_z_loss(lse, labels, shift=shift)


loss = loss + z_loss_weight * z_loss
```


### Generalized Usage

While we have discussed using CCE in the context of large language models, the only constraint
to use CCE is that loss can be formulated using something that resembles following:

```python
logits = X @ A.T + b  # (b is an optional bias)
loss = F.cross_entropy(logits.float(), targets)
```

Given that format, CCE can then be used as
```python
loss = linear_cross_entropy(X, A, target_token, bias=b)
```

This is a very general and encompasses vision models, contrastive losses, e.g. CLIP, etc.


### Transformers Integration

**Installation**

Install cut-cross-entropy with transformers dependencies
```bash
pip install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git"
```

**Usage**

If you are using transformers, you can patch transformers to use CCE directly. Note that
logits will no longer be returned (`None` will be returned instead).
```python
from cut_cross_entropy.transformers import cce_patch

cce_patch("llama")

# or

model = ...
model = cce_patch(model)
```

We currently support the Llama, Phi3, Mistral, and Gemma2 families of models.

`cce_patch` takes two options. The first is the linear-cross-entropy implementation to use. Currently `"cce"` or `"torch_compile"`.

The second
is the loss reduction. We support `"mean"`, `"sum"`, and `"none"`, that mirror their PyTorch counterpart.
`"mean"` is the default and what the transformers trainer API expects.
However,
`"none"` in particular can enable for efficient computation of quantities based on the loss.

For example, the following efficiently computes the perplexity of a batch of sequences:
```python
import transformers

from cut_cross_entropy.transformers import cce_patch


model = transformers.AutoModelForCausalLM.from_pretrained(...)

model = cce_patch(model, reduction="none")

labels = input_ids.clone()
labels[~attention_mask] = -100 # -100 is the ignore index for PyTorch and CCE.

outputs = model(input_ids, attention_mask, labels=labels)

loss = outputs[0] # A (B, T - 1) tensor because reduction="none". T - 1 because the first input token has
# no loss.

ppl = torch.exp(
    # [:, 1:] because the first token has no loss
    loss.sum(1) / (labels[:, 1:] != -100).count_nonzero(dim=1)
).mean()  # Average perplexity over the batch
```



### Training and reproducing the benchmark results

We provide a training in `training/train.py`.

**Installation**
```bash
pip install "cut-cross-entropy[all] @ git+https://github.com/apple/ml-cross-entropy.git"
```

**Training**

Use `scripts/train.sh` to train a full model.

**Benchmarking**

The benchmark script can be run via `python -m benchmark`.

Expected output with A100 SMX4, PyTorch 2.4.1, and CUDA 12.4.

```
          method        kind  runtime_ms  op_mem_mb test_data
0            cce     loss-fw        46.4        1.1    gemma2
1  torch_compile     loss-fw        49.9     4000.1    gemma2
2       baseline     loss-fw        81.9    24000.0    gemma2
3            cce     loss-bw        89.3     1163.0    gemma2
4  torch_compile     loss-bw        92.3    12000.0    gemma2
5       baseline     loss-bw       122.4    16000.0    gemma2
6            cce  loss-fw-bw       134.8     1164.0    gemma2
7  torch_compile  loss-fw-bw       144.0    16000.1    gemma2
8       baseline  loss-fw-bw       208.8    28000.0    gemma2
```

#### `linear_cross_entropy_kl` fused-vs-dense notes

Point-in-time measurements for the fixed-`T=1` `linear_cross_entropy_kl` path versus
`_dense_linear_cross_entropy_kl`, collected on `NVIDIA A100-SXM4-80GB` with `bfloat16`
and `warmup=1`.

Reproduce with:

```bash
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py sweep --preset=long-seq --dtype=bfloat16 --warmup=1
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py sweep --preset=batch-size --dtype=bfloat16 --warmup=1
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py sweep --preset=hidden-dim --dtype=bfloat16 --warmup=1
```

Long-sequence and vocab sweep:

| Case | Dense total ms | Fused total ms | Fused - Dense ms | Dense peak MiB | Fused peak MiB | Fused - Dense MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=8192, V=4096, D=2048` | 11.86 | 17.26 | +5.40 | 1008.4 | 296.7 | -711.7 |
| `B=1, S=16384, V=4096, D=2048` | 13.37 | 27.78 | +14.40 | 1968.5 | 514.1 | -1454.4 |
| `B=1, S=32768, V=4096, D=2048` | 21.12 | 48.58 | +27.46 | 3888.8 | 946.7 | -2942.0 |
| `B=1, S=65536, V=4096, D=2048` | 41.43 | 97.02 | +55.59 | 7729.3 | 1813.8 | -5915.4 |
| `B=1, S=65536, V=8192, D=2048` | 81.74 | 173.10 | +91.36 | 14929.3 | 1878.4 | -13050.9 |

Batch-size sweep at fixed `S=8192, V=4096, D=2048`:

| Case | Dense total ms | Fused total ms | Fused - Dense ms | Dense peak MiB | Fused peak MiB | Fused - Dense MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1` | 11.82 | 16.32 | +4.50 | 1008.4 | 296.7 | -711.7 |
| `B=2` | 13.07 | 26.84 | +13.77 | 1968.5 | 514.1 | -1454.4 |
| `B=4` | 21.11 | 49.83 | +28.72 | 3888.8 | 946.7 | -2942.0 |
| `B=8` | 41.46 | 96.18 | +54.72 | 7729.3 | 1813.8 | -5915.4 |

Hidden-dimension sweep at fixed `B=1, S=8192, V=4096`:

| Case | Dense total ms | Fused total ms | Fused - Dense ms | Dense peak MiB | Fused peak MiB | Fused - Dense MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `D=1024` | 10.43 | 7.73 | -2.70 | 960.4 | 157.5 | -802.9 |
| `D=2048` | 10.68 | 16.40 | +5.72 | 1008.4 | 295.9 | -712.4 |
| `D=4096` | 11.95 | 27.77 | +15.83 | 1104.4 | 577.6 | -526.8 |

Observed trends:

- For fixed `V=4096, D=2048`, the batch-size sweep is nearly identical to the long-seq sweep once `N = B * S` matches. For example, `B=8, S=8192` and `B=1, S=65536` both land near `+55 ms` slowdown and save about `5.9 GiB`, so the slowdown is primarily tracking token count `N`, not whether `N` comes from batch or sequence.
- `V` is the strongest extra multiplier once `N` is already large. At `N=65536, D=2048`, increasing `V` from `4096` to `8192` changes the fused-vs-dense total delta from `+55.59 ms` to `+91.36 ms`.
- `D` also matters, but less than `N` and `V` in the tested range. At `N=8192, V=4096`, the fused-vs-dense total delta moves from `-2.70 ms` at `D=1024` to `+15.83 ms` at `D=4096`.
- The fused path remains primarily a memory optimization. At `B=1, S=65536, V=8192, D=2048`, fused peak memory stays below `1.9 GiB` while dense reaches about `14.9 GiB`.

Numerical accuracy:

The same benchmark script can compare the fused path against two different references for both forward outputs
and backward gradients:

- `reference=dense`: compare against `_dense_linear_cross_entropy_kl` evaluated directly in the input dtype.
- `reference=exact`: upcast the quantized inputs to float32 first, then evaluate the reference with strict float32 matmuls.

Accuracy cases use the same hidden-state scaling as the unit tests:
`student_h, teacher_h ~ N(0, 1 / sqrt(D))`, while `student_c` and `teacher_c`
remain standard normal.

Reproduce with:

```bash
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py accuracy_sweep --preset=default --dtype=float32 --reference=exact
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py accuracy_sweep --preset=default --dtype=bfloat16 --reference=dense
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py accuracy_sweep --preset=default --dtype=bfloat16 --reference=exact
```

The accuracy commands pin `torch.set_float32_matmul_precision("highest")` so float32 references stay on
strict float32 matmul instead of falling back to TF32. Point-in-time A100 results below report max absolute
error. Relative error can look artificially large on near-zero gradients, so max-abs is the more stable
summary metric here.

`float32` against `reference=exact`:

| Case | all max abs | ce max abs | kl max abs | grad_h max abs | grad_c max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=256, V=32768, D=2048` | 3.814697e-06 | 1.907349e-06 | 1.668930e-06 | 1.788139e-07 | 3.783498e-10 |
| `B=4, S=256, V=32768, D=2048` | 2.861023e-06 | 1.907349e-06 | 1.788139e-06 | 5.122274e-08 | 1.964509e-10 |
| `B=1, S=8192, V=4096, D=2048` | 3.814697e-06 | 1.907349e-06 | 1.907349e-06 | 1.018634e-09 | 1.509761e-10 |

`bfloat16` against `reference=dense`:

| Case | all max abs | ce max abs | kl max abs | grad_h max abs | grad_c max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=256, V=32768, D=2048` | 7.333755e-03 | 7.528305e-03 | 2.689362e-04 | 1.220703e-04 | 9.536743e-07 |
| `B=4, S=256, V=32768, D=2048` | 7.741928e-03 | 7.712364e-03 | 3.195405e-04 | 3.051758e-05 | 4.768372e-07 |
| `B=1, S=8192, V=4096, D=2048` | 8.093834e-03 | 7.872581e-03 | 1.972795e-03 | 3.814697e-06 | 1.192093e-07 |

`bfloat16` against `reference=exact`:

| Case | all max abs | ce max abs | kl max abs | grad_h max abs | grad_c max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=256, V=32768, D=2048` | 1.335144e-05 | 8.583069e-06 | 5.841255e-06 | 6.190129e-05 | 5.745387e-07 |
| `B=4, S=256, V=32768, D=2048` | 1.621246e-05 | 1.144409e-05 | 6.139278e-06 | 1.557544e-05 | 4.572212e-07 |
| `B=1, S=8192, V=4096, D=2048` | 1.907349e-05 | 1.525879e-05 | 7.629395e-06 | 3.421213e-06 | 1.090411e-07 |

Accuracy takeaways:

- In `float32`, the fused path stays in the expected `e-6` range against the exact reference. Across these cases, `all_loss` max abs stays below `3.82e-06`, `ce_loss` below `1.91e-06`, and `kl_loss` below `1.91e-06`.
- In `bfloat16`, the choice of reference matters a lot. Against the dense bf16 baseline, forward max abs stays around `7e-3` to `8e-3` for `all_loss`/`ce_loss`; against the exact float32-upcast reference, the same fused outputs are much closer, with `all_loss` max abs below `1.91e-05`, `ce_loss` below `1.53e-05`, and `kl_loss` below `7.63e-06`.
- That gap between `bfloat16 vs dense` and `bfloat16 vs exact` shows that most of the apparent bf16 forward discrepancy comes from the dense bf16 reference materializing logits in bf16 before `target_logit`, `logsumexp`, and teacher-expectation terms are formed. The fused path keeps those forward accumulations in fp32, so it tracks the exact reference much more closely.
- In `bfloat16`, backward gradients are still noisier than `float32`, but they remain well below the forward dense-reference gap. Against the exact reference, `grad_h` max abs stays below `6.20e-05` and `grad_c` below `5.75e-07`.

Backward kernel breakdown:

The fused backward path currently reconstructs logits, recovers probabilities from the saved LSE values,
and runs both `_mm_backward` branches inside one Triton kernel. To attribute time by stage, the benchmark
profiles four kernel variants and differences them:

- `common-only`: `softmax/LSE/prob` work only
- `common+dE`: common work plus the `dE` matmul-backward branch
- `common+dC`: common work plus the `dC` accumulation branch
- `full`: both `dE` and `dC`

Reproduce with:

```bash
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py profile_backward_sweep --preset=backward-breakdown --dtype=bfloat16 --warmup=1 --iters=5
PYTHONPATH=$PWD python benchmark/linear_cross_entropy_kl.py profile_backward --batch_size=8 --seq_len=8192 --vocab_size=4096 --hidden_dim=2048 --dtype=bfloat16 --warmup=1 --iters=5
```

Point-in-time A100 bf16 results:

| Case | Active rows | Common ms | dE extra ms | dC extra ms | Residual ms | Full kernel ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=8192, V=4096, D=2048` | 7784 | 0.11 | 4.42 | 4.47 | -0.75 | 8.25 |
| `B=1, S=65536, V=4096, D=2048` | 62253 | 0.11 | 32.93 | 35.06 | -5.04 | 63.06 |
| `B=1, S=65536, V=8192, D=2048` | 62272 | 0.11 | 65.09 | 69.79 | -11.09 | 123.90 |
| `B=8, S=8192, V=4096, D=2048` | 62283 | 0.11 | 32.79 | 35.06 | -4.95 | 63.00 |

Backward-kernel takeaways:

- `softmax/LSE/prob` reconstruction is negligible in the tested large-token regimes, about `0.11 ms` or `0.1%` to `1.4%` of the full backward kernel.
- The dominant costs are the two `_mm_backward` branches. `dC` is consistently a bit slower than `dE`, around `55%` versus `52%` of the full kernel time in the large-`N` cases.
- Matching `N = B * S` gives nearly identical backward-kernel breakdowns: `B=8, S=8192, V=4096` and `B=1, S=65536, V=4096` both land near `63 ms` full-kernel time with almost the same `dE/dC` split.
- Doubling `V` at fixed `N` roughly doubles both heavy branches: from about `33/35 ms` to about `65/70 ms` for `dE/dC`, which confirms that the backward slowdown is driven overwhelmingly by the matrix-product branches, especially the `dC` path, rather than by `softmax/LSE` reconstruction.
- The residual term is negative because the full kernel shares some work and memory traffic between the `dE` and `dC` branches, so profiling them separately and summing the parts slightly overestimates the combined runtime.

### Development

If dependencies are installed locally, `cut-cross-entropy` will work without a pip install as long as `python` is executed in the root path of the github repo.

To install directly from the github repo, either use an (editable) install or manipulate PYTHONPATH, e.g.

```bash
pip install -e ".[dev]"

# or
pip install ".[dev]"

# or
export PYTHONPATH=/path/to/ml-cross-entropy:${PYTHONPATH}
```

## Citation

```
@inproceedings{wijmans2025cut,
  author       = {Erik Wijmans and
                  Brody Huval and
                  Alexander Hertzberg and
                  Vladlen Koltun and
                  Philipp Kr\"ahenb\"uhl},
  title        = {Cut Your Losses in Large-Vocabulary Language Models},
  booktitle    = {International Conference on Learning Representations},
  year         = {2025},
}
```


## License
This sample code is released under the [LICENSE](LICENSE) terms.

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
