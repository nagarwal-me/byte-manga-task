##Manga Page Transcription
========================

This notebook is my attempt at transcribing story text from three consecutive English manga pages with a vision-language model. The goal isn't just to read the text: the output also needs to keep the page boundaries, reading order, and speaker IDs consistent.

The main notebook is `final.ipynb`. I ran it in Google Colab and used Qwen2.5-VL-3B-Instruct as the base model, with a LoRA adapter fine-tuned using PEFT and 4-bit quantization.

**What I worked on**
----------------

- Prepared the manga sequences and matched image files to the sequence metadata and labels.
- Created a reproducible train/validation split (seed `42`).
- Fine-tuned the model with a LoRA/QLoRA setup rather than training all model weights.
- Reduced the image resolution during experiments to keep GPU memory use manageable.
- Checked model outputs, added JSON parsing/validation, and tried a page-by-page fallback when a three-page response wasn't valid JSON.
- Saved both the structured predictions and raw model responses so I could inspect failures instead of treating every output as correct.

**The image-resolution change**
---------------------------

One practical change in this notebook was reducing the image resolution. Feeding full-resolution manga pages to a vision-language model can use a lot of GPU memory, especially when processing three pages together. I lowered the image pixel budget during training experiments to `512 × 512` pixels, and in one later experiment to `256 × 256` pixels. For test inference, the notebook also resizes each page with Pillow's `thumbnail((512, 512), Image.Resampling.LANCZOS)` and sets the Qwen processor's `max_pixels` to `65536`.

This was a memory trade-off, not a free improvement: smaller inputs are cheaper to process, but fine dialogue text can become harder to read. The notebook includes these settings as part of the experimentation, rather than assuming that the highest resolution is always practical on the available GPU.

**Model and libraries**
-------------------

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Deep learning: PyTorch
- Model loading and processing: Hugging Face Transformers
- Fine-tuning: PEFT with LoRA/QLoRA
- Quantization: bitsandbytes, 4-bit NF4
- Image processing: Pillow and `qwen-vl-utils`
- Runtime: Google Colab with a CUDA-enabled GPU

**Dataset and files**
-----------------

The notebook expects the dataset and helper files to be available under `/content` in Colab. The filenames referenced include:

- `images-20261010T124807Z-1-001.zip` — labelled/development images
- `test-20261010T125233Z-1-001.zip` — test images
- `sequences.json` — sequence metadata
- `labels (1).jsonl` — labelled transcription examples
- `score.py` — scoring script
- `sample_submission.jsonl` — submission template, if supplied with the dataset

The notebook extracts the images into `/content/manga_dataset` and saves the train/validation split there. If the dataset files are named differently or stored in another location, update the paths in the relevant cells.

The split used in the notebook has 64 training sequences and 16 validation sequences, with seed `42`. Test inference expects 15 sequences, each containing exactly three pages.

**Running it in Colab**
-------------------

1. Upload `final.ipynb` and the required dataset/helper files to Google Colab.
2. Choose a CUDA GPU under Runtime → Change runtime type.
3. Run the setup cells and check that the required packages, GPU, and dataset files are detected.
4. Run the extraction and metadata cells, then check that the image paths and labels line up.
5. Load the base model and configure the quantization and LoRA adapter.
6. Run the training and validation experiments. The notebook includes memory checks and different image-pixel settings used while debugging.
7. Run test inference and inspect the generated JSON and raw model responses.
8. Run the packaging cell if you need the outputs bundled for submission.

If Colab disconnects or runs out of GPU memory, reconnect to a GPU and rerun the setup/data cells as needed. Make sure the adapter has actually been saved before trying to load it again.

**Transcription format**
-------------------

Each prediction is for one sequence and should contain three page arrays. Every extracted line has a `speaker` and `text` field. For example:

    {
      "sequence_id": "sequence_id_here",
      "pages": [
        [{"speaker": "character1", "text": "Example dialogue."}],
        [],
        [{"speaker": "UNKNOWN", "text": "Another line."}]
      ]
    }

The prompt asks the model to include dialogue, thoughts, narration, vocalizations, and punctuation-only speech balloons; preserve reading order and page boundaries; and use consistent speaker IDs. It excludes sound effects, credits, signs, and other non-story text. `UNKNOWN` is used when the speaker can't be identified confidently.

**Outputs**
-------

The notebook writes outputs under `/content`, including:

- `manga_qlora_adapter/` — saved LoRA adapter
- `manga_dataset/train_val_split_seed42.json` — saved train/validation split
- `manga_test_sequences.json` — test-sequence manifest
- `manga_final_submission/` or a versioned submission directory — structured predictions, raw generations, and progress logs
- `Manga_Transcription_Submission.zip` — packaged files, when the packaging cell is run

These are files created inside the Colab runtime; they are not automatically included when you download the notebook. Download them from Colab if you need to keep them.

**Things to keep in mind**
----------------------

- This is an experimental pipeline, not a claim of perfect OCR. Small or stylized lettering can still be missed or misread, and speaker attribution can be wrong.
- Reducing image resolution helped manage the input size, but can also remove details needed to read small text.
- The JSON checks and fallback inference make outputs easier to handle; they don't guarantee that the transcription itself is accurate.
- Training and inference depend on the available GPU memory and on the dataset files being in the expected locations.
- The project currently runs through a notebook and Colab paths, rather than a standalone command-line interface.

**Reproducibility**
---------------

The train/validation split uses seed `42` and is saved as `train_val_split_seed42.json`. Keep that file when comparing experiments. For a fair comparison, keep the prompt, evaluation script, and image-resolution settings consistent, and record any changes between runs.
