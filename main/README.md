# Manga Page Transcription

This project explores transcribing dialogue from three consecutive English manga pages while keeping the page order and character labels consistent.

The workflow is in `final.ipynb` and was built to run in Google Colab with a GPU. It uses Qwen2.5-VL-3B-Instruct as the base vision-language model, with a QLoRA/LoRA adapter for fine-tuning.

## What it does

- Groups manga images into three-page sequences.
- Uses labelled development examples to create a reproducible train/validation split.
- Loads the model in 4-bit mode to reduce GPU memory use.
- Fine-tunes an adapter on the training sequences.
- Runs inference on the held-out test sequences and writes predictions in JSONL format.
- Saves raw model responses as well, so outputs that need checking can be traced back to the original generation.

The transcription prompt asks the model to include dialogue, thoughts, narration, and vocalizations; preserve reading order and page boundaries; use consistent speaker labels; and leave out sound effects, credits, signs, and other non-dialogue text.

## Model and tools

- **Base model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Framework:** PyTorch and Hugging Face Transformers
- **Fine-tuning:** PEFT with LoRA/QLoRA
- **Quantization:** 4-bit NF4 through bitsandbytes
- **Image handling:** Pillow and `qwen-vl-utils`
- **Environment:** Google Colab with CUDA-enabled GPU

## Dataset files

The notebook expects the dataset and supporting files to be available in the Colab `/content` directory. The filenames used in the notebook include:

- `images-20261010T124807Z-1-001.zip` — labelled/development images
- `test-20261010T125233Z-1-001.zip` — test images
- `sequences.json` — sequence metadata
- `labels (1).jsonl` — labelled transcription data
- `sample_submission.jsonl` — submission template, where provided
- `score.py` — evaluation script, where provided

The notebook extracts the archives and creates working files under `/content/manga_dataset`, along with a saved split file. If your files have different names or are stored elsewhere, update the corresponding paths in the notebook.

The current split is **64 training sequences and 16 validation sequences**, using seed `42`. The test archive is expected to contain **15 sequences**, each with exactly three pages.

## Running the notebook

1. Upload `final.ipynb` and the dataset/supporting files to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another compatible CUDA GPU).
3. Run the setup and package-installation cells.
4. Run the dataset extraction and metadata cells. Check that all image paths and labels are found before continuing.
5. Run the model-loading and adapter setup cells.
6. Run the training and validation/evaluation cells.
7. Run the test inference cells to generate predictions.
8. Review the prediction summary and raw outputs before using the results.
9. Run the final packaging cell if you need the adapter and prediction files in a single ZIP.

The notebook contains recovery and diagnostic cells because model loading and training can use a substantial amount of GPU memory. If a runtime disconnects, reconnect the GPU and rerun the setup/data cells before attempting to load the saved adapter.

## Output files

The notebook writes its outputs under `/content`, including:

- `manga_qlora_adapter/` — saved adapter and training metadata
- `manga_dataset/train_val_split_seed42.json` — saved train/validation split
- `manga_test_sequences.json` — manifest of the test sequences
- `manga_final_submission_v2/test_predictions.jsonl` — structured predictions
- `manga_final_submission_v2/raw_model_outputs.jsonl` — raw generations and diagnostic details
- `Manga_Transcription_Submission.zip` — packaged adapter, split, predictions, raw outputs, and scoring script

These are Colab paths, not files automatically included alongside the notebook. Download or package them from the runtime if you need to keep them.

## Prediction format

Each JSONL record represents one sequence and contains three page arrays. Each line of transcription has a `speaker` and `text` field. The intended shape is:

```json
{
  "sequence_id": "sequence_id_here",
  "pages": [
    [{"speaker": "character1", "text": "Example dialogue."}],
    [],
    [{"speaker": "UNKNOWN", "text": "Another line."}]
  ]
}
```

The example is illustrative; actual output quality and formatting should be checked before submission.

## Notes and limitations

- This is an experimental transcription pipeline, not a guarantee of publication-ready OCR.
- Manga text can be small, stylized, or embedded in busy artwork, so predictions may omit lines, misread text, or assign speakers incorrectly.
- The notebook includes fallback inference and raw-output logging to help inspect malformed or empty generations. A fallback being used does not mean the transcription is correct.
- Fine-tuning was constrained by GPU memory, so the notebook includes several memory-management and recovery attempts. Check the saved training log and validation scores before making claims about model quality.
- The notebook relies on Google Colab paths and uploaded dataset files; it is not currently packaged as a standalone command-line application.

## Reproducibility

The train/validation split uses seed `42` and is saved to `train_val_split_seed42.json`. Keep that file when comparing runs so the validation set stays consistent. For a fair comparison, use the same image resizing, prompt, and evaluation script for each model variant.
