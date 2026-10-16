# Manga Page Sequence Understanding — Experiments

This repository contains an experimental Google Colab notebook for transcribing story text from **three consecutive English manga pages**. The model is expected to preserve page boundaries and reading order, transcribe dialogue/thoughts/narration/vocalizations, assign consistent speaker IDs, and exclude sound effects and other non-story text.

The main record of the work is [`experiments.md(1).ipynb`](./experiments.md%281%29.ipynb). It contains the baseline, data checks, multiple fine-tuning attempts, debugging and recovery cells, and validation/submission preparation. This README documents the experiments actually present in the notebook, including iterations that were later superseded.

## Task and output format

The intended output is structured JSON with three page arrays. Each transcribed line has a `speaker` and `text` field. The later inference workflow adds `sequence_id` to the outer record so predictions can be matched reliably to their input sequence.

Example structure:

```json
{
  "sequence_id": "seq_example",
  "pages": [
    [{"speaker": "character1", "text": "Example dialogue."}],
    [],
    [{"speaker": "character2", "text": "Another line."}]
  ]
}
```

The prompt emphasizes:
- Keep each line on the page where it appears.
- Preserve intended reading order and punctuation.
- Keep speaker IDs consistent across all three pages (`character1`, `character2`, etc.).
- Use `UNKNOWN` when the speaker cannot be identified confidently.
- Do not invent dialogue or move text between pages.
- Exclude sound effects, credits, signs, labels, and other non-story text.
- Return an empty array for a page with no qualifying story text.
- Return valid JSON only, with no explanatory prose.

## Dataset and preparation

The notebook expects task files to be made available in the Colab `/content` directory:

- `images-20261010T124807Z-1-001.zip`
- `test-20261010T125233Z-1-001.zip`
- `labels (1).jsonl`
- `sequences.json`
- `score.py`

The development archive is extracted beneath `/content/manga_dataset`. The notebook inspects the JSONL labels and sequence metadata, counts extracted image files, and validates that metadata image paths can be resolved. It builds image lookups from sequence folders and filenames.

### Reproducible split

The notebook filters metadata to the **80 development sequences**, then uses Python's `random.Random(42)` to create a reproducible split:

- **64 training sequences**
- **16 validation sequences**

The split is saved to `manga_dataset/train_val_split_seed42.json`. Assertions check the expected counts, no overlap, and complete coverage of the development IDs. Validation labels/references are used for evaluation rather than training.

## Baseline and initial experiments

The notebook first loads the base vision-language model and runs inference on validation examples. Raw outputs and predictions are saved so that the base model can be inspected and compared with later adapter-based runs. The notebook also checks required files, model/processor availability, GPU access, and validation ID/label consistency before evaluation.

Baseline-related files referenced in the notebook include:
- `baseline_raw_outputs.jsonl`
- `baseline_predictions.jsonl`
- `baseline_scores.json`
- `validation_references.jsonl`

Exact files present depend on which cells completed in the Colab session.

## Model and compute setup

- **Base model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Framework:** PyTorch and Hugging Face Transformers
- **Fine-tuning:** PEFT LoRA / QLoRA
- **Quantization:** 4-bit bitsandbytes quantization using NF4, double quantization, and float16 compute
- **Image processing:** `qwen-vl-utils` and the model's `AutoProcessor`
- **Memory management:** gradient checkpointing, reduced image-pixel budgets, smaller trainable adapters, and later chunked loss computation

The notebook is designed around a GPU-enabled Colab runtime. Exact GPU requirements and package compatibility depend on the Colab instance and installed library versions.

## Fine-tuning experiments

The notebook is an experiment log rather than a single polished training script. Several configurations and recovery approaches appear in separate cells.

### 1. Initial QLoRA setup

The initial adapter configuration uses:

| Parameter | Setting |
|---|---|
| LoRA rank (`r`) | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Bias | `none` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | Enabled during k-bit model preparation |

The notebook verifies that these projection layers exist before attaching the adapter and prints the number of trainable parameters. An early setup freezes the model's visual encoder and caps the image processor's pixel budget at `512 × 512`.

### 2. Manual / custom training attempts

Later cells experiment with custom training loops and revised learning rates (including `1e-4` and `5e-5` in different attempts), token/label alignment checks, finite-loss and gradient checks, logging sequence-level losses, and saving training records. The notebook includes checks to ensure the assistant target has trainable tokens and that prompt tokens align with the prefix of the full conversation.

These attempts are retained in the notebook as experimentation history. They should not all be run one after another without checking model state, output paths, and whether a cell is intended as a replacement or recovery run.

### 3. Reduced-rank adapter

A later, more memory-conscious configuration reduces the adapter rank:

| Parameter | Later configuration |
|---|---|
| LoRA rank (`r`) | `4` |
| LoRA alpha | `8` |
| LoRA dropout | `0.05` |
| Target modules | Attention projections and MLP projections, as configured in the later training cells |
| Epochs in later Trainer-based runs | `1` |
| Learning rate in later Trainer-based runs | `1e-4` |

This reduces the number of adapter parameters compared with the earlier rank-8 setup. The notebook also retains separate adapter output directories so newer experiments do not overwrite earlier adapters.

### 4. Freezing the vision encoder

The notebook explicitly sets the visual encoder's parameters to `requires_grad = False`. This keeps the vision encoder frozen while training the LoRA parameters in the language-model projection layers. The model is still used to encode the page images during the forward pass; its visual encoder weights are not updated by training.

### 5. Lower image-resolution / pixel-budget experiments

Image resolution was adjusted to reduce GPU memory use. The notebook includes:
- An early cap of `512 × 512` pixels.
- A later pixel budget of `401,408` pixels.
- A further reduced pixel budget of `200,704` pixels in the clean-restart experiment.

These are alternative configurations from different runs, not one simultaneous setting. The later notebook cells explicitly compare inference on original-resolution images as well, so training-time image resizing and evaluation-time resolution should not be conflated.

### 6. Pages-only targets and sequence IDs supplied by Python

An important later change is to **stop asking the model to generate `sequence_id` as part of its learned response**. The training target is focused on the `pages` content, while the inference code already knows which sequence it is processing.

During inference, Python passes the sequence ID through the current loop/function and attaches it to the parsed result. This avoids spending generation tokens on an identifier that is already known and avoids mismatches between a generated ID and the actual input sequence. The output record still includes `sequence_id` for evaluation and submission compatibility.

The notebook includes both earlier prompts that request `sequence_id` in the generated JSON and later pages-only training/inference experiments. The pages-only setup is the later refinement, not a description of every earlier cell.

### 7. Chunked loss computation

The clean-restart experiment introduces chunked loss computation, with a configured `CHUNK_SIZE = 64`, to reduce peak memory use when calculating token-level loss. This is another memory-management experiment in addition to 4-bit quantization, gradient checkpointing, freezing the vision encoder, reduced LoRA rank, and smaller image pixel budgets.

## Post-generation processing and robustness

The notebook spends substantial effort making model outputs usable even when generation does not produce perfectly formatted JSON.

### Deterministic generation and repetition control

The later inference helper uses deterministic decoding (`do_sample=False`) and a repetition penalty of `1.1`. The generation helpers also set a maximum output-token budget (commonly `1200` or `1800` tokens for the three-page response, depending on the experiment). These settings aim to reduce repeated text and keep generation bounded; they do not guarantee correct transcription.

### JSON parsing and repair

The later inference cells define a JSON parser that attempts to extract/parse the expected JSON object from the raw generated text. Parsed results are checked against the required schema rather than accepted blindly. The intended outer structure contains `sequence_id` and `pages`, with exactly three page arrays and each dialogue entry containing `speaker` and `text`.

### Per-page fallback

A sequence-level generation can fail to parse or return an invalid structure. The notebook therefore adds a fallback path that attempts generation one page at a time when the three-page output cannot be accepted. The fallback page results are then assembled into the three-page record, and the output tracks whether page fallback was used.

This is a recovery mechanism, not an equivalent replacement for successful three-page inference: page-by-page fallback may lose cross-page speaker consistency that the three-page context is intended to support.

### Output validation and failure accounting

Before a prediction is accepted, the notebook checks that:
- The returned `sequence_id` matches the input sequence ID.
- Exactly three pages are present.
- Every page is a list.
- Predictions and failures are recorded separately.

The validation workflow saves successful predictions, failure details, and scores. A later cell builds a complete 16-sequence prediction file; failed sequences are represented with three empty page arrays so the scorer receives a record for every reference ID. This ensures complete evaluation coverage, but empty fallbacks should be treated as failed predictions rather than successful transcription.

## Validation and evaluation

The notebook evaluates the **16-sequence validation split** using the provided `score.py` script and validation references. It includes:
- Baseline inference and scoring.
- Adapter-based inference.
- An original-resolution validation run.
- JSON/schema assertions and sequence-ID matching.
- Separate failure logs and a count of sequences requiring page-level fallback.
- A complete prediction file covering all 16 reference IDs for scoring.

The notebook also inspects the test archive, locates its `sequences.json` manifest and submission template, and counts test images as part of submission preparation.

**No final score or improvement claim is stated here.** The notebook contains multiple runs and recovery cells, and a reliable result should be reported only from the exact saved prediction file and scorer output for the chosen run.

## Outputs and paths

Paths referenced by the notebook include:

- `/content/manga_dataset/`
- `/content/manga_dataset/train_val_split_seed42.json`
- `/content/baseline_raw_outputs.jsonl`
- `/content/baseline_predictions.jsonl`
- `/content/validation_references.jsonl`
- `/content/manga_qlora_adapter/`
- `/content/manga_qlora_recovery_63/`
- `/content/manga_pages_only_r4_401k/`
- `/content/manga_pages_only_r4_200k/`
- `/content/manga_evaluation_original_adapter_fixed/predictions.jsonl`
- `/content/manga_evaluation_original_adapter_fixed/failures.json`
- `/content/manga_evaluation_original_adapter_fixed/scores.json`

Not every path is guaranteed to exist in a fresh runtime. The notebook includes a cell that checks which saved artifacts are present.

## Running the notebook

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Upload the dataset archives, labels, metadata, and scorer to the expected locations.
4. Run the environment and file checks.
5. Extract and validate the dataset.
6. Generate or load the fixed seed-42 train/validation split.
7. Run baseline inference and save the outputs.
8. Choose **one** intended training configuration. Do not run competing training/recovery cells concurrently or assume every later cell uses the same in-memory model.
9. Run inference with the matching adapter and its corresponding helper functions.
10. Validate predictions, save failures, and run the official scorer on the matching reference set.

The notebook uses packages including `transformers`, `peft`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`, `torch`, and `Pillow`. Several are installed/imported within the notebook, but compatible versions should be checked if reproducing the experiments outside Colab.

## Limitations and reproducibility notes

- This is a chronological experiment notebook with repeated definitions, alternative settings, recovery cells, and multiple adapter directories—not a clean one-command pipeline.
- Results depend on the exact adapter, inference helper, image-resolution setting, and output file selected.
- Some cells rely on objects loaded by earlier cells. After a runtime reset, rerun the relevant setup cells before inference.
- The input archives and task labels are external files and are not embedded in this README.
- Use the task-provided scoring script and report the score from the exact final predictions; do not infer performance from successful generation counts alone.
- Respect the task's train/validation/test split. Do not train on validation or test references.

## Suggested repository layout

```text
.
├── README.md
├── experiments.md(1).ipynb
├── images-20261010T124807Z-1-001.zip   # provide separately
├── test-20261010T125233Z-1-001.zip     # provide separately
├── labels (1).jsonl                    # provide separately
├── sequences.json                     # provide separately
└── score.py                            # task-provided
```

## License

No project-specific license is declared in the notebook. Check the licenses and terms for the base model, dataset, and task materials before redistributing them.
