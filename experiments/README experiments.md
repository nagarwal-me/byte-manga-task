# Manga Page Sequence Understanding

So the goal here was pretty straightforward: get a vision-language model to read three consecutive manga pages and return the actual story text in a clean JSON format, without mixing up pages, reading order, or speaker IDs.

I started with `Qwen/Qwen2.5-VL-3B-Instruct` as the baseline and checked what its outputs looked like before fine-tuning. Then I experimented with LoRA/QLoRA instead of trying to fine-tune the whole model.

## What I tried

- **LoRA rank 8 → rank 4:** I reduced the adapter rank to make the trainable part smaller. The later setup used `r=4` and `alpha=8`.
- **Freezing the vision encoder:** The model still processes the manga images, but the vision encoder's weights do not get updated during training.
- **Making training fit the GPU:** I used 4-bit NF4 quantization, gradient checkpointing, and smaller image pixel budgets. I tried different limits (`512 × 512`, `401,408` pixels, and `200,704` pixels) across separate runs.
- **Trying different training setups:** I experimented with custom training loops and learning rates such as `1e-4` and `5e-5`. I also added checks for prompt/label alignment and training stability rather than just assuming a run was working.
- **Changing how sequence IDs are handled:** Earlier, the model was asked to generate `sequence_id` itself. Later, I made Python attach the ID after generation, since the code already knows which sequence it is processing. That lets the model focus on the page content.
- **Cleaning up generation:** I used deterministic decoding (`do_sample=False`) and a repetition penalty of `1.1` to help control repeated text. I also added JSON parsing and schema checks so malformed outputs do not get accepted blindly.
- **Adding a fallback:** If the response for all three pages could not be parsed, the notebook tried generating each page separately. This helps recover outputs, but it can make speaker consistency across pages harder to maintain.

## Comparisons and what I learned

I kept a baseline run, fine-tuned runs, and an original-resolution inference run, using a fixed split of 64 training sequences and 16 validation sequences. The notebook also includes failure tracking and scoring with the task's `score.py`.

The main trade-off was between fitting the experiment into limited GPU memory and preserving enough image detail for small manga text. The page-by-page fallback is another trade-off: it can rescue a failed sequence, but loses some of the shared context that helps keep speakers consistent. The Python-based sequence ID handling is a cleaner approach because the model does not need to generate information that is already known.

I haven't claimed that one setup definitively performed best, because that needs to be checked against the scorer output for the exact prediction file being reported.

## What I'd try next

First, I'd run the baseline and the final adapter through the same validation setup and compare their actual scores. Then I'd inspect the errors—especially missed text, wrong page placement, and inconsistent speaker IDs. I'd also test whether the smaller pixel budgets are hurting small text, choose the best configuration from those comparisons, and clean up the notebook so the final training and evaluation path is reproducible.

## Running it

The experiments are in [\`experiments.md(1).ipynb\`](./experiments.md\(1\).ipynb). The notebook expects the task's image archives, `labels (1).jsonl`, `sequences.json`, and `score.py` to be available in the Colab runtime. It uses a GPU and packages including PyTorch, Transformers, PEFT, Accelerate, bitsandbytes, and `qwen-vl-utils`.

The notebook contains alternative and recovery cells, so don't run every training cell as if they were one pipeline. Pick the configuration you want to test, then evaluate the predictions produced by that run.



Write me this whole in Markdown format with hashtags and asterisk signs.
