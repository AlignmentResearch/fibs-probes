# probe-inference

Inference for four activation-probe architectures:

- linear;
- MLP;
- EFC (early-fusion covariance);
- axial.

Load a trained probe, from a local directory or the Hugging Face Hub, and score a
transcript's residual-stream activations with it; the result is one score.

The package depends only on `torch`. The `[hub]` extra adds `huggingface_hub`, for
`load_probe_from_hub`.

## Usage

```python
from probe_inference import load_probe, load_probe_from_hub

probe = load_probe("my_probes/qwen3.5-9b/efc")  # a local probe directory
probe = load_probe_from_hub("qwen3.5-9b/efc")  # the published weights, <model>/<arch>
probe.arch, probe.layers, probe.read_window, probe.output
# ('efc', (10, 13, 18, 21, 26, 29), 'last-user-and-assistant-generation', 'logit')

# acts: {layer: Tensor[seq, d_model]} for the layers in probe.used_layers (any float dtype),
#       or one Tensor[len(probe.layers), seq, d_model] stacked in probe.layers order.
# read_mask: Tensor[seq] bool, True on the tokens the probe's read window selects.
read_mask = probe.read_mask(prompt_mask, completion_mask, followup_start_positions)
score = probe.score(acts, read_mask)  # float

scores = probe.score_batch(batched_acts, batched_mask)  # (B,) float64; acts {layer: [B, seq, d]}
per_layer = probe.layer_logits(acts_b, mask_b)  # linear/MLP only: {layer: (B,) logits}
```

`load_probe_from_hub(path)` downloads one trained probe from the published weights, the Hugging Face model
repository `AlignmentResearch/probe-inference-weights` at the commit pinned by
`probe_inference.load.WEIGHTS_REVISION`, and loads it. Only that probe's directory is downloaded. Pass
`repo_id`, `revision` and `repo_type` to load from another repository. The published probes are
`<model>/<arch>/`, with `<arch>` in `linear`, `mlp`, `efc` and `axial`, for these models: `qwen3.5-2b`,
`qwen3.5-9b`, `qwen3.5-27b`, `qwen3.5-122b-a10b`, `qwen3.5-397b-a17b`, `nemotron-3-nano-30b-a3b`,
`nemotron-3-super-120b-a12b`, `nemotron-3-ultra-550b-a55b` and `kimi-k3`. The repository's card gives
each model's revision and layers, how the probes were trained, and the licences of the models the probes
were trained on.

What `score` returns depends on the architecture:

| Architecture | `probe.output` | Score |
|---|---|---|
| Linear, MLP | `probability` | Each layer's probe reads one token. The score is the mean of the per-layer sigmoids over `probe.used_layers` (`probe_metadata.json` `layer_rule.used_layers`). |
| EFC | `logit` | One pooled logit over the read window. |
| Axial | `logit` | The logit at the last read token, capped by $10\tanh(x/10)$. |

`score` fails loudly rather than scoring a wrong input. It raises `ValueError` in these cases:

- a layer is missing, or has the wrong shape or hidden size;
- the activations are not floating point, or hold a non-finite value at a read position;
- the read mask is not a bool tensor of the activations' `(batch, seq)` shape, or selects nothing;
- a linear or MLP row does not read exactly one token;
- an EFC row reads fewer than two tokens, or only identical ones;
- a read token's activation is exactly zero in the layer the padding mask is taken from. The probes
  treat all-zero rows as padding, so the token would be dropped silently.

`load_probe` also refuses (`InputScaleError`) a `unit_norm` or `centered_unit_norm` probe whose
`input_scale` is exactly 1.0, the unset default: its weights were saved without the normaliser they
were trained with.

## Activation contract

The probes were trained on activations with the following properties. Scores on activations that
differ from this are not meaningful.

| Part | Contract |
|---|---|
| Layer index | Layer $k$ is the output of decoder block $k$, 0-indexed, of the model loaded with `AutoModelForCausalLM`. For Qwen3.5/3.6 that is `model.model.layers[k]`. For Nemotron-3 it is `layers[k]` of the inner `NemotronHModel`, which Nano-30B holds at `model.backbone` and Super-120B at `model.model`. This equals Hugging Face `output_hidden_states[k + 1]`. The activations of the probes at `WEIGHTS_REVISION` were captured with vLLM hooks at these decoder-layer outputs. None of the probes below reads the last layer ($N-1$), whose Hugging Face hidden state is the final norm's output. |
| Layer index, Kimi K3 | Kimi K3's decoder layers use attention residuals, so a layer returns no single hidden state. Layer $k$ is the attention-residual mixture that layer $k+1$ reads, computed with the model's own `attn_res` op and layer $k+1$'s residual-attention weights. |
| Stream | The residual stream, as the Hugging Face block returns it. |
| Dtype | The model ran in bfloat16, and the activations were stored in bfloat16. Kimi K3 ran from its released MXFP4 checkpoint (`moonshotai/Kimi-K3`, MXFP4 weights), not a bf16 one; its activations were also stored in bfloat16. `score` runs the probes in float32. (Training ran the probe forward under bfloat16 autocast.) |
| Read window, linear and MLP | `second-last-token-generation`: exactly one token, the token before the final end-of-turn token. For Qwen and Nemotron-3 that is the last token of the assistant answer; for a prefilled `No.` or `Yes.` it is the full stop. For Kimi K3 it is the `<\|sep\|>` of the closing `<\|close\|>message<\|sep\|>`, six tokens after the full stop. |
| Read window, EFC and axial | `last-user-and-assistant-generation`: every token from the first token of the final user turn (its `<\|im_start\|>`, or `<\|open\|>` for Kimi K3) through the assistant's end-of-turn token, inclusive. |
| Read mask | The axial probe uses rotary embeddings, so you must pass the full transcript and a mask for which tokens you want the probe to read when you call `score`|

### Follow-up rendering

A follow-up is a question with a prefilled answer, for example
`Did you do anything the user did not ask you to do?` answered `No.`. Rendered, it looks like this:

```
Qwen3.5 / 3.6:  <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nNo.<|im_end|>
Nemotron-3:     <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think></think>No.<|im_end|>
Kimi K3:        <|open|>message role="user"<|sep|>{question}<|close|>message<|sep|><|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>No.<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>
```

Kimi K3's end-of-turn token is `<|end_of_msg|>`.

### Building the read mask

`probe.read_mask(prompt_mask, completion_mask, followup_start_positions)` builds the mask for the
probe's window. The arguments are:

- `completion_mask`: True on the assistant answer tokens after the empty thinking block, plus the
  end-of-turn token. For a `No.` prefill that is `No`, `.` and `<|im_end|>`. For Kimi K3 it is every token
  after `<|open|>response<|sep|>`: `No`, `.`, `<|close|>`, `response`, `<|sep|>`, `<|close|>`, `message`,
  `<|sep|>` and `<|end_of_msg|>`.
- `prompt_mask`: True on every other real token of the row.
- `followup_start_positions`: the index of the final user turn's first token (`<|im_start|>`, or
  `<|open|>` for Kimi K3).

A hand-built boolean mask that selects the positions in the table above works too.

| Model (HF revision) | Blocks | `d_model` | Layers | Weights revision |
|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` (`15852e8c`) | 24 | 2048 | 7, 10, 13, 16, 19, 22 |
| `Qwen/Qwen3.5-9B` (`c2022362`) | 32 | 4096 | 10, 13, 18, 21, 26, 29  |
| `Qwen/Qwen3.5-27B` (`fc05daec`) | 64 | 5120 | 19, 27, 35, 43, 51, 58 |
| `Qwen/Qwen3.5-122B-A10B` (`dc4d3484`) | 48 | 3072 | 14, 20, 26, 32, 38, 43  |
| `Qwen/Qwen3.5-397B-A17B` (`84726181`) | 60 | 4096 | 18, 25, 33, 40, 48, 54 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`bf77c317`) | 52 | 2688 | 16, 22, 29, 35, 42, 47 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (`2dc98e2a`) | 88 | 4096 | 26, 37, 48, 59, 70, 79  |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (`77df655d`) | 108 | 8192 | 32, 45, 59, 72, 86, 97 |
| `moonshotai/Kimi-K3` (`f831ab66`) | 93 | 7168 | 28, 39, 51, 62, 74, 84 | `WEIGHTS_REVISION` only |

## Artifacts

| Artifact | Location |
|---|---|
| Probe weights | [`AlignmentResearch/probe-inference-weights`](https://huggingface.co/AlignmentResearch/probe-inference-weights), tag [`camera-ready`](https://huggingface.co/AlignmentResearch/probe-inference-weights/tree/camera-ready) (commit `05def1c1`, which `WEIGHTS_REVISION` pins) |
| Training data | FIBSv1, [`AlignmentResearch/fibs-v1`](https://huggingface.co/datasets/AlignmentResearch/fibs-v1) at revision `65dccf12`. |
| Parity fixture | [`AlignmentResearch/probe-inference-parity`](https://huggingface.co/datasets/AlignmentResearch/probe-inference-parity), see [Tests](#tests) |

The probes at `WEIGHTS_REVISION` were trained on the model's own activations over FIBSv1: 152,980
training rows and 5,000 validation rows, the same rows for every model, seed 0. Linear and MLP trained for
up to 6 epochs, EFC and axial for up to 4, with early stopping on the validation loss; each published
probe is the checkpoint with the lowest validation loss. The weights repository's card gives each probe's
validation AUROC.

## Probe directory format

A trained probe is one directory, such as `qwen3.5-9b/efc/`. The same layout is read from a local
path or from a Hub repository. A linear or MLP probe is one small probe per layer (a *layer probe*);
an EFC or axial probe is one module that reads all its layers at once.

```
linear/                      # linear and MLP: one layer probe per layer
  probe_metadata.json
  layer_<L>/config.json
  layer_<L>/model.pt
efc/                         # EFC and axial: one cross-layer probe
  probe_metadata.json
  config.json
  model.pt
```

| File | Content |
|---|---|
| `config.json` | `{"class_name", "init_args"}`. `class_name` is `LinearProbe`, `MLPProbe`, `EFCProbe` or `AxialProbe`; any other key is ignored. |
| `model.pt` | A plain float32 `state_dict`, loaded with `weights_only=True`. |
| `probe_metadata.json` | See the list below. |

`probe_metadata.json` holds:

- `layers`: model layers, in stacking order;
- `eval_sequence_aggregator`: `mean` or `last`;
- `obfuscate_over`: the read window;
- for linear and MLP only, `layer_rule.used_layers`.

## Tests

```bash
uv sync
uv run pytest
```

The unit tests use synthetic probes and run on CPU in seconds. `tests/test_parity.py` compares the
package with reference scores on real activations, using the published probes. Its fixture (about
150 MiB) is the Hugging Face dataset `AlignmentResearch/probe-inference-parity` at the commit pinned by `FIXTURE_REVISION` in that file. 
It holds a few rows of activations and
token masks per model and the reference scores for those rows, but no probes. 
Each probe is loaded with `load_probe_from_hub` from the weights at
`PARITY_WEIGHTS_REVISION` in that file (`22a7a341`). Both
downloads go through the Hub cache (`HF_HOME`); cache that directory in CI to avoid re-downloading them
(`.github/workflows/ci.yml` does). 
Other tests in that file check that `WEIGHTS_REVISION`
holds all 36 probes with the layers in the table above, and that one of them loads and scores. The tests
need network access even with a warm cache, because they list the weights repository's files.

| Set | Probes | Notes |
|---|---|---|
| `nemotron-3-super-120b` | EFC, axial | 80 rows; one batch mixes 27- and 24-token read windows |
| `qwen3.5-9b` | linear, MLP, EFC, axial | 32 rows |

## Licence

This package is released under the MIT licence (see `LICENSE`).
The published probe weights and test fixture on the Hugging Face Hub are also MIT. The weights at
`WEIGHTS_REVISION` are derived from Qwen models (Apache-2.0), NVIDIA Nemotron-3 Nano and Super (NVIDIA
Nemotron Open Model License), NVIDIA Nemotron-3 Ultra (OpenMDW-1.1) and Kimi K3 (Kimi K3 License). The
test fixture is derived from Qwen and Nemotron-3 Super models. The licence texts and attribution notices
of these models ship with each repository (`NOTICE` and the `LICENSE-*` files).
