# From a FIBS row to a probe score

This example runs the whole path of `probe-inference`. It reads rows of FIBS, the follow-up deception dataset, and
computes Qwen3.5-2B's residual-stream activations at the probe layers (see the activation contract in the top-level
README). It then builds each probe's read mask with `probe.read_mask` and scores the row with `probe.score`. By
default the notebook scores with the published Qwen3.5-2B probes. With `USE_RANDOM_PROBES = True` it writes and
loads probes with random weights instead.

- `fibs_end_to_end.ipynb`: the walk-through, committed with its outputs cleared.
- `fibs.py`: `iter_fibs`, the rows of a FIBS split as `FibsRow`s, read from the Hub a few rows at a time.
- `activations.py`: `render`, `collect`, `select_rows`, `collect_rows` and a command line that stores activations.
- `token_masks.py`: `token_masks`, the prompt mask, completion mask and follow-up start that `probe.read_mask` takes,
  found in a rendered row's token ids.
- `random_probes.py`: `write_random_probes`, random-weight probes in the probe directory format.

## Requirements

- FIBS, `AlignmentResearch/fibs-v1` at `65dccf12`, the revision the published probes were trained on. The dataset is
  public, so no Hugging Face token is needed. A split is read in part, a row group at a time: the notebook's rows
  fetch about 10-15 MB per split, not the whole ~200 MB file.
- `Qwen/Qwen3.5-2B` at `15852e8c`: a download of about 5 GB on the first run. It needs `transformers>=5.2`.
- The published probes, `AlignmentResearch/probe-inference-weights` at the package's `WEIGHTS_REVISION`: at most a
  few tens of MB for the four Qwen3.5-2B probes. Random probes need no download.
- [`uv`](https://docs.astral.sh/uv/). The example has its own environment (`examples/fibs/pyproject.toml` and
  `uv.lock`), so none of this is a dependency of the package.

## Run

Run a copy of the notebook, not the committed file, so that Jupyter's autosave does not write outputs into the
tracked notebook. Copies named `*.local.ipynb` are gitignored, and `tests/test_examples_committed_clean.py` fails if a
committed notebook has outputs.

```bash
cp examples/fibs/fibs_end_to_end.ipynb examples/fibs/fibs_end_to_end.local.ipynb
uv run --project examples/fibs --group notebook jupyter lab examples/fibs/fibs_end_to_end.local.ipynb
```

To execute it without opening it, write the result outside the repository:

```bash
uv run --project examples/fibs --group notebook jupyter nbconvert --to notebook --execute \
    --output-dir /tmp/fibs-out examples/fibs/fibs_end_to_end.ipynb
```

To keep the token ids, token masks and activations of a few rows of each split in a `.pt` file (`--max-tokens` and
`--device` are the other options):

```bash
uv run --project examples/fibs python examples/fibs/activations.py --per-quadrant 2 --out fibs_activations.pt
```

Each record holds the activations of the whole sequence at the six probe layers in bfloat16, with the masks that
`probe.read_mask` takes, so a probe scores a record without the model:

```python
data = torch.load("fibs_activations.pt", weights_only=True)
probe = load_probe_from_hub("qwen3.5-2b/efc")
for record in data["records"]:
    read_mask = probe.read_mask(record["prompt_mask"], record["completion_mask"], record["followup_start"])
    score = probe.score(dict(zip(data["layers"], record["acts"])), read_mask)
```

The example's offline tests need neither the network nor the model, and CI runs them:

```bash
uv run --project examples/fibs pytest examples/fibs/tests
```

Runtime on a Linux CPU with 8 threads, with the model and data cached: the eight forward passes take about 10 s, and
the whole notebook about 95 s.

## Caveats

- **Not an evaluation.** The published probes were trained on FIBSv1 (152,980 training and 5,000 validation rows of
  its 163,016), so the notebook's rows are probably in their training data. Its per-label table is a plumbing check.
- **Random probes** use PyTorch's default initialisation, so their scores mean nothing.
- **Numerics.** The probes were trained on activations from vLLM runs on CUDA. This example runs Hugging Face
  transformers on the CPU, MPS or CUDA, in bfloat16, and the activations differ in the low bits.
- **Rows.** The example skips rows with tool definitions or tool calls, rows whose follow-up answer has a reasoning
  trace, and rows longer than 1,500 rendered tokens (to keep a forward pass to seconds). Tool-call arguments are
  stored as JSON strings, which Qwen3.5's chat template cannot iterate, and the probes were trained with a specific
  policy for rendering tools that the example does not reproduce.
- **Warnings.** transformers says that `causal_conv1d` and `flash-linear-attention` are not installed and that
  Qwen3.5's linear-attention layers use a slower pure-PyTorch path. The result is the same.
