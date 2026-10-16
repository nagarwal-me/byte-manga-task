```python
print("hello")
```

```python

import sys
import torch

print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    ))
```

```python

%pip install -q transformers peft accelerate bitsandbytes qwen-vl-utils
```

Checking which product files are available

```python

from pathlib import Path

print("=== Checking project files ===")

locations = [
    Path("/content"),
    Path("/content/drive/MyDrive"),
]

expected_files = [
    "images-20261010T124807Z-1-001.zip",
    "test-20261010T125233Z-1-001.zip",
    "labels (1).jsonl",
    "sequences.json",
    "score.py",
]

for root in locations:
    if not root.exists():
        continue

    print(f"\nSearching: {root}")

    for filename in expected_files:
        matches = list(root.rglob(filename))
        if matches:
            for match in matches[:3]:
                print("FOUND:", match)
        else:
            print("NOT FOUND:", filename)
```

```python

# 1. Verify the environment
import torch
import transformers
import peft
import accelerate
import bitsandbytes
import qwen_vl_utils
from pathlib import Path

print("=== Library Check ===")
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Accelerate:", accelerate.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n=== Dataset File Check ===")

expected_files = [
    "images-20261010T124807Z-1-001.zip",
    "test-20261010T125233Z-1-001.zip",
    "labels (1).jsonl",
    "sequences.json",
    "score.py",
]

for filename in expected_files:
    matches = list(Path("/content").rglob(filename))
    if matches:
        print(f"FOUND {filename}: {matches[0]}")
    else:
        print(f"NOT FOUND: {filename}")
```

```python

from pathlib import Path
import zipfile
import json

root = Path("/content")

# 1. Find any existing label files
print("=== Label files ===")
for p in root.rglob("*"):
    if p.is_file() and "label" in p.name.lower():
        print(p)

# 2. Inspect the contents of both ZIP archives without extracting them
archives = [
    root / "images-20261010T124807Z-1-001.zip",
    root / "test-20261010T125233Z-1-001.zip",
]

for archive in archives:
    print(f"\n=== Archive: {archive.name} ===")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        print("Total entries:", len(names))
        print("First 12 entries:")
        for name in names[:12]:
            print(" ", name)

        label_names = [
            name for name in names
            if "label" in Path(name).name.lower()
        ]
        if label_names:
            print("Label-related entries:", label_names)

# 3. Summarize sequence metadata
metadata_path = root / "sequences.json"
with metadata_path.open(encoding="utf-8") as f:
    metadata = json.load(f)

print("\n=== Sequence metadata ===")
print("Metadata type:", type(metadata).__name__)
print("Number of records:", len(metadata))

if isinstance(metadata, list):
    from collections import Counter
    print("Split counts:", dict(Counter(
        row.get("split", "unspecified") for row in metadata
    )))
```

Extracting the dataset

```python

from pathlib import Path
from collections import Counter
import zipfile
import json
import shutil

ROOT = Path("/content")
DATA = ROOT / "manga_dataset"
DATA.mkdir(exist_ok=True)

# 1. Extract the archives
archives = [
    ROOT / "images-20261010T124807Z-1-001.zip",
    ROOT / "test-20261010T125233Z-1-001.zip",
]

for archive in archives:
    destination = DATA / archive.stem
    if not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(destination)
        print("Extracted:", archive.name)
    else:
        print("Already extracted:", archive.name)

# 2. Inspect the labels
labels_path = ROOT / "labels (1).jsonl"
with labels_path.open(encoding="utf-8") as f:
    labels = [json.loads(line) for line in f if line.strip()]

print("\n=== Labels ===")
print("Labelled sequences:", len(labels))
print("First sequence ID:", labels[0]["sequence_id"])
print("Pages in first sequence:", len(labels[0]["pages"]))

# 3. Inspect metadata
with (ROOT / "sequences.json").open(encoding="utf-8") as f:
    metadata = json.load(f)

print("\n=== Metadata ===")
print("Total sequences:", len(metadata))
print("Split counts:", dict(Counter(
    row.get("split", "unspecified") for row in metadata
)))

# 4. Count extracted image files
images = [
    p for p in DATA.rglob("*")
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
]

print("\n=== Extracted images ===")
print("Image files found:", len(images))
print("Sample image paths:")
for p in images[:5]:
    print(p)
```

```python

from pathlib import Path
import json

ROOT = Path("/content")
DATA = ROOT / "manga_dataset"

with (ROOT / "sequences.json").open(encoding="utf-8") as f:
    metadata = json.load(f)

# Build a lookup using each image's filename and sequence folder.
image_lookup = {}
for path in DATA.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        image_lookup[(path.parent.name, path.name)] = path

missing = []
resolved = 0

for sequence in metadata:
    seq_id = sequence["sequence_id"]

    for image_path in sequence["images"]:
        filename = Path(image_path).name
        key = (seq_id, filename)

        if key in image_lookup:
            resolved += 1
        else:
            missing.append((seq_id, image_path))

print("=== Image Path Validation ===")
print("Sequences checked:", len(metadata))
print("Expected image paths:", sum(len(s["images"]) for s in metadata))
print("Successfully resolved:", resolved)
print("Missing paths:", len(missing))

if missing:
    print("\nFirst missing paths:")
    for item in missing[:10]:
        print(item)
else:
    print("SUCCESS: All metadata image paths resolve to extracted images.")
```

Creating validation split

```python

import json
import random
from pathlib import Path

ROOT = Path("/content")
DATA = ROOT / "manga_dataset"

with (ROOT / "sequences.json").open(encoding="utf-8") as f:
    metadata = json.load(f)

# Keep only labelled development sequences.
dev_ids = [
    row["sequence_id"]
    for row in metadata
    if row["split"] == "development"
]

assert len(dev_ids) == 80, f"Expected 80 development sequences, got {len(dev_ids)}"

# Reproducible 64/16 split.
rng = random.Random(42)
shuffled_ids = dev_ids.copy()
rng.shuffle(shuffled_ids)

train_ids = sorted(shuffled_ids[:64])
val_ids = sorted(shuffled_ids[64:])

# Verify no overlap and correct counts.
assert len(train_ids) == 64
assert len(val_ids) == 16
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids) | set(val_ids) == set(dev_ids)

# Save the split so future runs use exactly the same sequences.
split = {
    "seed": 42,
    "train_ids": train_ids,
    "validation_ids": val_ids,
}

split_path = DATA / "train_val_split_seed42.json"
split_path.write_text(
    json.dumps(split, indent=2) + "\n",
    encoding="utf-8"
)

print("Training sequences:", len(train_ids))
print("Validation sequences:", len(val_ids))
print("Overlap:", len(set(train_ids) & set(val_ids)))
print("Saved split:", split_path)
print("\nFirst 5 training IDs:", train_ids[:5])
print("First 5 validation IDs:", val_ids[:5])
```

Verifying if the installed library supports 4-bit GPU

```python

import torch
import transformers
import bitsandbytes as bnb
from transformers import AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("=== Model Setup Check ===")
print("Model:", MODEL_ID)
print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

# Check whether Transformers can recognize this model's configuration.
from transformers import AutoConfig

try:
    config = AutoConfig.from_pretrained(MODEL_ID)
    print("Model configuration loaded:", config.model_type)
except Exception as e:
    print("MODEL CONFIGURATION ERROR:")
    print(type(e).__name__, str(e)[:1000])

# Check that 4-bit quantization can be configured.
try:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("4-bit configuration created successfully.")
except Exception as e:
    print("4-BIT CONFIGURATION ERROR:")
    print(type(e).__name__, str(e)[:1000])
```

```python

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model in 4-bit mode...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

print("\nModel loaded successfully!")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory allocated (GB):", round(
    torch.cuda.memory_allocated(0) / (1024**3), 2
))
```

```python

import json
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info

# Load dataset metadata and the saved validation split
with open("/content/sequences.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("/content/manga_dataset/train_val_split_seed42.json", "r", encoding="utf-8") as f:
    split = json.load(f)

val_ids = split["validation_ids"]
print("Validation sequences:", len(val_ids))

# Find the images using their sequence ID and filename
image_root = Path("/content/manga_dataset")
image_lookup = {}

for path in image_root.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        image_lookup[(path.parent.name, path.name)] = path

# Locate the first validation sequence
sequence_id = val_ids[0]
record = next(row for row in metadata if row["sequence_id"] == sequence_id)

image_paths = []
for image_entry in record["images"]:
    filename = Path(image_entry).name
    matches = [
        path for (folder, name), path in image_lookup.items()
        if name == filename and folder == sequence_id
    ]
    if not matches:
        # Fall back to matching the filename anywhere in the extracted dataset
        matches = [path for path in image_root.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find image: {filename}")
    image_paths.append(matches[0])

assert len(image_paths) == 3, f"Expected 3 images, found {len(image_paths)}"
print("Testing sequence:", sequence_id)
print("Images:")
for path in image_paths:
    print(" ", path)


instruction = """
Transcribe the story text from these THREE consecutive English manga pages.

IMPORTANT RULES:
1. Keep every line on the page where it actually appears.
2. Preserve the intended reading order within each page.
3. Include dialogue, thoughts, narration, vocalizations, and punctuation-only speech balloons.
4. Exclude sound effects, credits, labels, signs, and other non-story text.
5. Use consistent speaker IDs across all three pages.
6. Use speaker IDs such as character1, character2, character3, etc.
7. Use "UNKNOWN" when the speaker cannot be identified confidently.
8. Do not invent dialogue or move text between pages.
9. If a page has no qualifying story text, return an empty list for that page.

Return ONLY valid JSON in this exact structure:
{
  "pages": [
    [{"speaker": "character1", "text": "Example dialogue."}],
    [],
    []
  ]
}

There must be exactly three page arrays, in input image order.
Each line item must contain only "speaker" and "text".
Do not include Markdown fences or explanations.
"""

messages = [
    {
        "role": "user",
        "content": (
            [{"type": "image", "image": str(path)} for path in image_paths]
            + [{"type": "text", "text": instruction}]
        ),
    }
]

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1200,
        do_sample=False,
    )

new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
generated_text = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
)[0]

print("\n----- MODEL OUTPUT -----\n")
print(generated_text)
```

Establishing baseline

```python

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info

# --------------------------------------------------
# 1. Verify required files and load validation IDs
# --------------------------------------------------
BASE = Path("/content")
DATA_ROOT = BASE / "manga_dataset"
LABELS_PATH = BASE / "labels (1).jsonl"
METADATA_PATH = BASE / "sequences.json"
SPLIT_PATH = DATA_ROOT / "train_val_split_seed42.json"
SCORER_PATH = BASE / "score.py"

for path in [LABELS_PATH, METADATA_PATH, SPLIT_PATH, SCORER_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

if "model" not in globals() or "processor" not in globals():
    raise RuntimeError("Model or processor is missing. Do not reload yet; tell me this error.")

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

with SPLIT_PATH.open("r", encoding="utf-8") as f:
    split = json.load(f)

with LABELS_PATH.open("r", encoding="utf-8") as f:
    labels = [json.loads(line) for line in f if line.strip()]

validation_ids = split["validation_ids"]
label_by_id = {row["sequence_id"]: row for row in labels}
metadata_by_id = {row["sequence_id"]: row for row in metadata}

assert len(validation_ids) == 16, f"Expected 16 validation IDs, got {len(validation_ids)}"
assert all(sid in label_by_id for sid in validation_ids), "Some validation labels are missing."
assert all(sid in metadata_by_id for sid in validation_ids), "Some validation metadata is missing."

print("Validation sequences:", len(validation_ids))
print("Model:", "already loaded")
print("Starting baseline inference...")

# --------------------------------------------------
# 2. Resolve all image paths
# --------------------------------------------------
image_lookup = {}
for path in DATA_ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        image_lookup[(path.parent.name, path.name)] = path

def get_sequence_images(sequence_id):
    entries = metadata_by_id[sequence_id]["images"]
    paths = []

    for entry in entries:
        filename = Path(entry).name
        path = image_lookup.get((sequence_id, filename))
        if path is None:
            raise FileNotFoundError(
                f"Missing image for {sequence_id}: {filename}"
            )
        paths.append(path)

    if len(paths) != 3:
        raise ValueError(f"{sequence_id} has {len(paths)} images, not 3.")

    return paths

# --------------------------------------------------
# 3. Helpers for inference and JSON parsing
# --------------------------------------------------
def generate_text(image_paths, instruction, max_new_tokens=1200):
    content = [{"type": "image", "image": str(p)} for p in image_paths]
    content.append({"type": "text", "text": instruction})
    messages = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0].strip()


def parse_json_response(raw):
    # Accept JSON even if the model wraps it in Markdown fences.
    text = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def clean_lines(lines):
    if not isinstance(lines, list):
        return None

    cleaned = []
    for item in lines:
        if not isinstance(item, dict):
            return None
        if not isinstance(item.get("speaker"), str):
            return None
        if not isinstance(item.get("text"), str):
            return None

        cleaned.append({
            "speaker": item["speaker"],
            "text": item["text"],
        })

    return cleaned


def parse_three_pages(raw):
    obj = parse_json_response(raw)
    if not isinstance(obj, dict):
        return None

    pages = obj.get("pages")
    if not isinstance(pages, list) or len(pages) != 3:
        return None

    cleaned_pages = []
    for page in pages:
        cleaned = clean_lines(page)
        if cleaned is None:
            return None
        cleaned_pages.append(cleaned)

    return cleaned_pages


def parse_single_page(raw):
    obj = parse_json_response(raw)
    if not isinstance(obj, dict):
        return None

    # The single-page prompt specifically requests {"lines": [...]}.
    lines = obj.get("lines")
    return clean_lines(lines)


FULL_PROMPT = """
Transcribe story text from these three consecutive English manga pages.
The images are page 1, page 2, and page 3, in that order.

Include dialogue, thoughts, narration, vocalizations, and punctuation-only
speech balloons. Exclude sound effects, credits, labels, signs, and other
non-story text. Preserve reading order and page boundaries exactly.
Do not invent text or move text from one page to another.
Use speaker IDs consistently across pages, such as character1, character2,
and so on. Use UNKNOWN when the speaker cannot be identified confidently.
A page with no qualifying story text must be [].

Return ONLY valid JSON in this exact format:
{
  "pages": [
    [{"speaker": "character1", "text": "Example."}],
    [],
    []
  ]
}
There must be exactly three arrays inside "pages".
Every line item must contain only "speaker" and "text".
"""

SINGLE_PAGE_PROMPT = """
Transcribe only the story text visible on this ONE manga page.
Include dialogue, thoughts, narration, vocalizations, and punctuation-only
speech balloons. Exclude sound effects, credits, labels, signs, and other
non-story text. Preserve reading order. Do not invent text.
Use speaker IDs such as character1, character2, etc., and UNKNOWN when unsure.
Return ONLY JSON in this format:
{"lines": [{"speaker": "character1", "text": "Example."}]}
If there is no qualifying story text, return {"lines": []}.
"""

# --------------------------------------------------
# 4. Run inference on all 16 validation sequences
# --------------------------------------------------
predictions = []
raw_records = []
fallback_count = 0
failed_pages = 0
start_time = time.time()

for index, sequence_id in enumerate(validation_ids, start=1):
    image_paths = get_sequence_images(sequence_id)
    print(f"\n[{index}/{len(validation_ids)}] {sequence_id}")

    raw = generate_text(image_paths, FULL_PROMPT, max_new_tokens=1400)
    pages = parse_three_pages(raw)
    method = "three_page"

    # If the model doesn't produce three page arrays, retry page by page.
    if pages is None:
        fallback_count += 1
        method = "page_by_page_fallback"
        pages = []

        for page_index, image_path in enumerate(image_paths, start=1):
            try:
                page_raw = generate_text(
                    [image_path], SINGLE_PAGE_PROMPT, max_new_tokens=500
                )
                page_lines = parse_single_page(page_raw)

                if page_lines is None:
                    print(f"  Warning: page {page_index} output could not be parsed.")
                    page_lines = []
                    failed_pages += 1

                pages.append(page_lines)
            except Exception as exc:
                print(f"  Warning: page {page_index} failed: {exc}")
                pages.append([])
                failed_pages += 1

    assert len(pages) == 3

    predictions.append({
        "sequence_id": sequence_id,
        "pages": pages,
    })
    raw_records.append({
        "sequence_id": sequence_id,
        "method": method,
        "raw_output": raw,
    })

    print("  Method:", method)
    print("  Predicted lines per page:", [len(p) for p in pages])

    # Save progress after every sequence, in case the session disconnects.
    with (BASE / "baseline_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (BASE / "baseline_raw_outputs.jsonl").open("w", encoding="utf-8") as f:
        for row in raw_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# --------------------------------------------------
# 5. Save validation references and score predictions
# --------------------------------------------------
references = [label_by_id[sid] for sid in validation_ids]

with (BASE / "validation_references.jsonl").open("w", encoding="utf-8") as f:
    for row in references:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nInference complete.")
print("Sequences predicted:", len(predictions))
print("Sequences requiring page-by-page fallback:", fallback_count)
print("Pages that could not be parsed:", failed_pages)
print("Elapsed minutes:", round((time.time() - start_time) / 60, 2))
print("Predictions saved to:", BASE / "baseline_predictions.jsonl")
print("Raw outputs saved to:", BASE / "baseline_raw_outputs.jsonl")

# Use the official scorer; do not calculate substitute metrics.
score_path = BASE / "baseline_scores.json"
command = [
    sys.executable, str(SCORER_PATH),
    "--references", str(BASE / "validation_references.jsonl"),
    "--predictions", str(BASE / "baseline_predictions.jsonl"),
    "--output", str(score_path),
]

print("\nRunning official scorer...")
result = subprocess.run(command, capture_output=True, text=True)

print(result.stdout)
if result.stderr.strip():
    print("Scorer messages:")
    print(result.stderr[-5000:])

if result.returncode == 0:
    print("Official scorer completed.")
    if score_path.exists():
        print(score_path.read_text(encoding="utf-8"))
else:
    print("SCORER FAILED with exit code:", result.returncode)
    print("Keep the saved predictions. Send me the scorer error; do not rerun inference.")
```

Checking raw output.

```python
import json

path = "/content/baseline_raw_outputs.jsonl"

with open(path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

for record in records:
    raw = json.dumps(record, ensure_ascii=False)
    if (
        "seq_2e4b837ead6b88ae" in raw
        or "seq_535e0754d805105b" in raw
    ):
        print("\n" + "=" * 70)
        print(raw[:12000])
```

Freezing the vision encoder

```python
import torch

# Confirm the model and processor are still loaded
assert "model" in globals(), "Model is not loaded."
assert "processor" in globals(), "Processor is not loaded."
assert torch.cuda.is_available(), "GPU is not available."

# Inspect the model structure before changing training settings
print("Model class:", type(model).__name__)
print("GPU:", torch.cuda.get_device_name(0))
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Find and freeze the vision encoder
vision_encoder = getattr(model.model, "visual", None)

assert vision_encoder is not None, (
    "Could not find model.model.visual; stop here and share the output."
)

for parameter in vision_encoder.parameters():
    parameter.requires_grad = False

# Limit each image to at most 512 × 512 pixels during training
image_processor = processor.image_processor
assert hasattr(image_processor, "max_pixels"), (
    "Image processor has no max_pixels setting; stop here."
)

original_max_pixels = image_processor.max_pixels
image_processor.max_pixels = 512 * 512

print("\nVision encoder frozen:", all(
    not p.requires_grad for p in vision_encoder.parameters()
))
print("Original image pixel cap:", original_max_pixels)
print("Training image pixel cap:", image_processor.max_pixels)
print("Model preparation check complete.")
```

```python
image_processor = processor.image_processor

print("Processor class:", type(image_processor).__name__)
print("Image size:", getattr(image_processor, "size", None))
print("Min pixels:", getattr(image_processor, "min_pixels", None))
print("Max pixels:", getattr(image_processor, "max_pixels", None))
print("Processor attributes:", [
    name for name in vars(image_processor)
    if "pixel" in name.lower() or "size" in name.lower()
])
```

```python
# Set a 512 × 512 pixel budget using the processor's actual configuration
processor.image_processor.size["max_pixels"] = 512 * 512

print("Updated image processor size:", processor.image_processor.size)
print("Vision encoder frozen:", all(
    not p.requires_grad
    for p in model.model.visual.parameters()
))
print("Ready for LoRA setup.")
```

```python
Verifying the exact layers
```

```python
import torch
import bitsandbytes as bnb

# Check that the base model is still quantized
print("4-bit quantization:", getattr(model, "is_loaded_in_4bit", False))
print("Vision encoder frozen:", all(
    not p.requires_grad for p in model.model.visual.parameters()
))

# Inspect the actual projection layers available for LoRA
target_suffixes = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
)

module_counts = {name: 0 for name in target_suffixes}

for name, module in model.named_modules():
    for suffix in target_suffixes:
        if name.endswith(suffix):
            module_counts[suffix] += 1

print("\nLoRA target layer counts:")
for name, count in module_counts.items():
    print(f"{name}: {count}")

assert all(count > 0 for count in module_counts.values()), (
    "Some expected projection layers were not found. Stop and share the output."
)

print("\nLayer verification complete.")
```

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare the quantized model for adapter training
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# Configure trainable LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\nQLoRA adapters attached.")
```

```python
import json
from pathlib import Path

with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

with open("/content/labels (1).jsonl", encoding="utf-8") as f:
    labels = {
        record["sequence_id"]: record
        for record in (json.loads(line) for line in f if line.strip())
    }

train_ids = split["train_ids"]
validation_ids = split["validation_ids"]

print("Training IDs:", len(train_ids))
print("Validation IDs:", len(validation_ids))
print("Training IDs with labels:", sum(sid in labels for sid in train_ids))
print("Validation IDs with labels:", sum(sid in labels for sid in validation_ids))

example = labels[train_ids[0]]
print("\nExample sequence:", example["sequence_id"])
print("Pages:", len(example["pages"]))
print("Lines per page:", [len(page) for page in example["pages"]])
print("Example labelled line:", next(
    (line for page in example["pages"] for line in page),
    "No dialogue lines"
))
```

```python
from pathlib import Path

DATA_ROOT = Path("/content/manga_dataset")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

train_id_set = set(train_ids)
validation_id_set = set(validation_ids)
image_paths_by_sequence = {}

for path in DATA_ROOT.rglob("*"):
    if (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.parent.name in train_id_set | validation_id_set
    ):
        image_paths_by_sequence.setdefault(path.parent.name, []).append(path)

# Sort each sequence's pages consistently
for sid in image_paths_by_sequence:
    image_paths_by_sequence[sid].sort(key=lambda p: p.name)

train_counts = {
    sid: len(image_paths_by_sequence.get(sid, []))
    for sid in train_ids
}
validation_counts = {
    sid: len(image_paths_by_sequence.get(sid, []))
    for sid in validation_ids
}

print("Training sequences with 3 images:",
      sum(count == 3 for count in train_counts.values()), "/", len(train_ids))
print("Validation sequences with 3 images:",
      sum(count == 3 for count in validation_counts.values()), "/", len(validation_ids))

bad_train = {sid: n for sid, n in train_counts.items() if n != 3}
bad_validation = {sid: n for sid, n in validation_counts.items() if n != 3}

print("Training sequences with incorrect image count:", bad_train)
print("Validation sequences with incorrect image count:", bad_validation)

if not bad_train and not bad_validation:
    example_id = train_ids[0]
    print("\nExample training sequence:", example_id)
    for p in image_paths_by_sequence[example_id]:
        print(p)
    print("\nImage-path verification passed.")
else:
    print("\nImage-path verification failed; do not start training yet.")
```

Testing one training example.

```python
import json
import gc
import torch
from qwen_vl_utils import process_vision_info

# Use one labelled TRAINING example only
sid = train_ids[0]
record = labels[sid]
paths = image_paths_by_sequence[sid]

instruction = """
Transcribe the English dialogue, thoughts, narration, and vocalizations
from these three consecutive manga pages, in reading order.
Exclude sound effects, credits, signs, and other non-dialogue text.
Return only valid JSON with exactly three page arrays:
{"pages": [[{"speaker":"character1","text":"..."}], [], []]}
Each line must contain speaker and text. Preserve punctuation and page
boundaries. Keep speaker labels consistent across all three pages.
Use UNKNOWN when the speaker cannot be identified. Do not add explanations.
"""

user_message = {
    "role": "user",
    "content": (
        [{"type": "image", "image": str(p)} for p in paths]
        + [{"type": "text", "text": instruction}]
    )
}

target = json.dumps(
    {"pages": record["pages"]},
    ensure_ascii=False
)

messages = [
    user_message,
    {"role": "assistant", "content": [{"type": "text", "text": target}]}
]

# Build the complete conversation and the prompt-only version
full_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
prompt_text = processor.apply_chat_template(
    [user_message], tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

full_inputs = processor(
    text=[full_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

prompt_inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

prompt_len = prompt_inputs["input_ids"].shape[1]
full_len = full_inputs["input_ids"].shape[1]

# Ensure prompt tokens align before masking the loss
prefix_matches = torch.equal(
    full_inputs["input_ids"][0, :prompt_len],
    prompt_inputs["input_ids"][0]
)

print("Training example:", sid)
print("Images:", len(paths))
print("Prompt tokens:", prompt_len)
print("Total tokens:", full_len)
print("Prompt matches full-input prefix:", prefix_matches)

assert prefix_matches, (
    "Prompt tokens do not align. Stop before training; share this output."
)
assert full_len > prompt_len, "No target tokens found."

# Mask prompt tokens so the loss applies only to the labelled answer
model_inputs = {
    k: v.to(model.device)
    for k, v in full_inputs.items()
    if torch.is_tensor(v)
}

target_labels = model_inputs["input_ids"].clone()
target_labels[:, :prompt_len] = -100

if "attention_mask" in model_inputs:
    target_labels[model_inputs["attention_mask"] == 0] = -100

target_token_count = int((target_labels != -100).sum().item())
print("Target tokens used for loss:", target_token_count)

assert target_token_count > 0

# Test one forward/backward pass without updating the weights
model.config.use_cache = False
model.train()
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

model_inputs["labels"] = target_labels

try:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**model_inputs)

    loss_value = float(outputs.loss.detach().float().cpu())
    outputs.loss.backward()

    trainable_with_grad = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    trainable_total = sum(
        1 for p in model.parameters() if p.requires_grad
    )

    print("\nDry-run loss:", loss_value)
    print("Trainable parameters receiving gradients:",
          trainable_with_grad, "/", trainable_total)
    print("Gradient test passed:",
          trainable_with_grad > 0 and loss_value == loss_value)

finally:
    model.zero_grad(set_to_none=True)
    if "outputs" in locals():
        del outputs
    del model_inputs, target_labels, full_inputs, prompt_inputs
    gc.collect()
    torch.cuda.empty_cache()

print("Dry run complete. No optimizer update was performed.")
```

Running the cleanup cell.

```python
import gc
import torch

model.zero_grad(set_to_none=True)

for name in ["outputs", "model_inputs", "target_labels",
             "full_inputs", "prompt_inputs",
             "image_inputs", "video_inputs"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print("Model loaded:", "model" in globals())
print("QLoRA attached:", hasattr(model, "peft_config"))
```

```python
import inspect
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

forward_params = inspect.signature(model.forward).parameters

print("Supports logits_to_keep:", "logits_to_keep" in forward_params)
print("Supports num_logits_to_keep:", "num_logits_to_keep" in forward_params)
print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

Inspecting model's internal components

```python
import inspect

base = model.get_base_model()

print("Base model:", type(base).__name__)
print("Transformer:", type(base.model).__name__)
print("Language-model head:", type(base.lm_head).__name__)

print("\nTransformer forward parameters:")
print(list(inspect.signature(base.model.forward).parameters.keys()))

print("\nLM head weight shape:", tuple(base.lm_head.weight.shape))
print("4-bit quantization:", getattr(base, "is_loaded_in_4bit", False))
print("GPU allocated:", f"{__import__('torch').cuda.memory_allocated()/1024**3:.2f} GB")
```

```python
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

sid = train_ids[0]
paths = image_paths_by_sequence[sid]

messages = [{
    "role": "user",
    "content": (
        [{"type": "image", "image": str(p)} for p in paths]
        + [{"type": "text", "text": "Transcribe the text on these pages."}]
    )
}]

image_inputs, video_inputs = process_vision_info(messages)

encoded = processor(
    text=[processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

print("Input token shape:", tuple(encoded["input_ids"].shape))
print("Pixel tensor shape:", tuple(encoded["pixel_values"].shape))
print("Image grid:", encoded.get("image_grid_thw"))
print("Configured image size:", processor.image_processor.size)
```

```python
from qwen_vl_utils import process_vision_info
import torch

# Reduce the image pixel budget
processor.image_processor.size["max_pixels"] = 256 * 256

sid = train_ids[0]
paths = image_paths_by_sequence[sid]

messages = [{
    "role": "user",
    "content": (
        [{"type": "image", "image": str(p)} for p in paths]
        + [{"type": "text", "text": "Transcribe the text on these pages."}]
    )
}]

image_inputs, video_inputs = process_vision_info(messages)

encoded = processor(
    text=[processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

print("Input token shape:", tuple(encoded["input_ids"].shape))
print("Pixel tensor shape:", tuple(encoded["pixel_values"].shape))
print("Image grid:", encoded.get("image_grid_thw"))
print("Configured image size:", processor.image_processor.size)
```

```python
import inspect

print(inspect.signature(processor.image_processor.__call__))
print("\nResize implementation:")
print(inspect.getsource(type(processor.image_processor).preprocess)[:5000])
```

Running one more cleanup

```python
import gc
import sys
import torch

# Release references retained by the failed notebook execution
ip = get_ipython()
if ip is not None:
    result = getattr(ip, "last_execution_result", None)
    if result is not None:
        result.error_in_exec = None
        result.error_before_exec = None

for name in ("last_traceback", "last_value", "last_type"):
    if hasattr(sys, name):
        setattr(sys, name, None)

model.zero_grad(set_to_none=True)
gc.collect()
torch.cuda.empty_cache()

print(f"GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print("QLoRA attached:", hasattr(model, "peft_config"))
```

```python
import gc
import sys
import torch

ip = get_ipython()

# Clear references to the failed execution and its traceback
for attr in ("last_value", "last_traceback", "last_type", "last_exc"):
    if hasattr(sys, attr):
        try:
            setattr(sys, attr, None)
        except Exception:
            pass

if ip is not None:
    result = getattr(ip, "last_execution_result", None)
    if result is not None:
        for attr in ("error_in_exec", "error_before_exec", "result"):
            if hasattr(result, attr):
                try:
                    setattr(result, attr, None)
                except Exception:
                    pass

    # Clear cached cell outputs that may retain tensors
    output_cache = ip.user_ns.get("Out")
    if isinstance(output_cache, dict):
        output_cache.clear()

    for name in ("_", "__", "___"):
        ip.user_ns.pop(name, None)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print(f"GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print("Model loaded:", "model" in globals())
print("QLoRA attached:", hasattr(model, "peft_config"))
```

We are now going to be restarting the session.

```python
import torch
from pathlib import Path

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print("\nRequired files:")
for p in [
    "/content/images-20261010T124807Z-1-001.zip",
    "/content/test-20261010T125233Z-1-001.zip",
    "/content/labels (1).jsonl",
    "/content/sequences.json",
    "/content/manga_dataset/train_val_split_seed42.json",
    "/content/baseline_predictions.jsonl",
    "/content/baseline_scores.json",
    "/content/score.py",
]:
    print(("FOUND  " if Path(p).exists() else "MISSING"), p)
```

```python
import json
import torch
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# 1. Reload the existing pretrained model in 4-bit
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Prepare for QLoRA training
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Restore the verified dataset split and labels
with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

train_ids = split["train_ids"]
validation_ids = split["validation_ids"]

with open("/content/labels (1).jsonl", encoding="utf-8") as f:
    labels = {
        record["sequence_id"]: record
        for record in (json.loads(line) for line in f if line.strip())
    }

# 4. Restore image paths
DATA_ROOT = Path("/content/manga_dataset")
image_paths_by_sequence = {}

for path in DATA_ROOT.rglob("*"):
    if (
        path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and path.parent.name in set(train_ids) | set(validation_ids)
    ):
        image_paths_by_sequence.setdefault(path.parent.name, []).append(path)

for sid in image_paths_by_sequence:
    image_paths_by_sequence[sid].sort(key=lambda p: p.name)

assert len(train_ids) == 64
assert len(validation_ids) == 16
assert all(sid in labels for sid in train_ids)
assert all(len(image_paths_by_sequence.get(sid, [])) == 3 for sid in train_ids)
assert all(len(image_paths_by_sequence.get(sid, [])) == 3 for sid in validation_ids)

model.eval()

print("\nTraining setup restored.")
print("Training sequences:", len(train_ids))
print("Validation sequences:", len(validation_ids))
print("GPU allocated:", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
print("Model loaded:", True)
```

```python
import gc
import json
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

model.train()
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

# One training example; never use validation data here
sid = train_ids[0]
record = labels[sid]
paths = image_paths_by_sequence[sid]

instruction = """
Transcribe dialogue, thoughts, narration, and vocalizations from these
three consecutive manga pages in reading order. Exclude sound effects,
credits, signs, and other non-dialogue text.
Return only JSON with exactly three page arrays:
{"pages": [[{"speaker":"character1","text":"..."}], [], []]}
Every line must contain speaker and text. Preserve punctuation and page
boundaries. Keep speaker labels consistent across pages. Use UNKNOWN
when the speaker cannot be identified. Do not add explanations.
"""

user_message = {
    "role": "user",
    "content": (
        [{"type": "image", "image": str(p)} for p in paths]
        + [{"type": "text", "text": instruction}]
    )
}

target = json.dumps({"pages": record["pages"]}, ensure_ascii=False)

messages = [
    user_message,
    {"role": "assistant", "content": [{"type": "text", "text": target}]}
]

# Tokenize full example and prompt separately to identify answer positions
full_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
prompt_text = processor.apply_chat_template(
    [user_message], tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

full_inputs = processor(
    text=[full_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
prompt_inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

prompt_len = prompt_inputs["input_ids"].shape[1]
full_len = full_inputs["input_ids"].shape[1]

assert torch.equal(
    full_inputs["input_ids"][0, :prompt_len],
    prompt_inputs["input_ids"][0]
), "Prompt alignment failed. Stop."

assert full_len > prompt_len, "No answer tokens found."

device = model.device
model_inputs = {
    k: v.to(device)
    for k, v in full_inputs.items()
    if torch.is_tensor(v)
}

input_ids = model_inputs["input_ids"]
target_ids = input_ids[:, prompt_len:]

print("Sequence:", sid)
print("Input tokens:", full_len)
print("Prompt tokens:", prompt_len)
print("Answer tokens:", target_ids.shape[1])

# Call the underlying transformer, not the full causal-LM forward.
# This returns hidden states without creating logits for prompt tokens.
base = model.get_base_model()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = base.model(
        **model_inputs,
        use_cache=False,
        return_dict=True
    )

    # Position t-1 predicts target token t
    answer_hidden = outputs.last_hidden_state[:, prompt_len - 1:-1, :]

    assert answer_hidden.shape[1] == target_ids.shape[1], (
        f"Hidden/target length mismatch: {answer_hidden.shape[1]} "
        f"vs {target_ids.shape[1]}"
    )

    # Only answer positions go through the large vocabulary projection
    answer_logits = base.lm_head(answer_hidden)

    print("Answer-only logits shape:", tuple(answer_logits.shape))

    loss = F.cross_entropy(
        answer_logits.float().reshape(-1, answer_logits.shape[-1]),
        target_ids.reshape(-1)
    )

print("Loss:", float(loss.detach().cpu()))
assert torch.isfinite(loss), "Loss is not finite."

loss.backward()

adapter_params = [
    p for p in model.parameters()
    if p.requires_grad
]
params_with_grad = sum(p.grad is not None for p in adapter_params)
finite_grads = all(
    torch.isfinite(p.grad).all().item()
    for p in adapter_params if p.grad is not None
)

print("Adapter parameters receiving gradients:",
      params_with_grad, "/", len(adapter_params))
print("Gradients finite:", finite_grads)
print(f"GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# No optimizer step: this was a test only
model.zero_grad(set_to_none=True)
del outputs, answer_hidden, answer_logits, loss
del full_inputs, prompt_inputs, model_inputs
gc.collect()
torch.cuda.empty_cache()

print("Dry run finished. No adapter update was performed.")
```

```python
import os
import gc
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from qwen_vl_utils import process_vision_info

# --------------------------------------------------
# 1. Training configuration
# --------------------------------------------------
OUTPUT_DIR = Path("/content/manga_qlora_adapter")
LOG_PATH = Path("/content/manga_qlora_training_log.jsonl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4

# Safety checks: use training IDs only
assert len(train_ids) == 64
assert len(validation_ids) == 16
assert set(train_ids).isdisjoint(validation_ids)
assert all(sid in labels for sid in train_ids)

# The model and adapters already exist in memory
assert hasattr(model, "peft_config"), "QLoRA adapters are missing."
model.config.use_cache = False
model.train()
model.zero_grad(set_to_none=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

instruction = """
Transcribe dialogue, thoughts, narration, and vocalizations from these
three consecutive English manga pages in reading order. Exclude sound
effects, credits, signs, and other non-dialogue text.
Return only valid JSON with exactly three page arrays:
{"pages": [[{"speaker":"character1","text":"..."}], [], []]}
Every line must contain speaker and text. Preserve punctuation and page
boundaries. Keep speaker labels consistent across all three pages.
Use UNKNOWN when the speaker cannot be identified. Do not add explanations.
"""

# Deterministic order for this first training run
ordered_train_ids = list(train_ids)
total_examples = len(ordered_train_ids)
optimizer_steps = 0
running_loss = 0.0
started = time.time()

# Start a fresh log for this run
with open(LOG_PATH, "w", encoding="utf-8") as log_file:
    log_file.write(json.dumps({
        "event": "training_started",
        "train_sequences": total_examples,
        "validation_sequences": len(validation_ids),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "loss_method": "answer-token-only logits",
    }) + "\n")

# --------------------------------------------------
# 2. Train on each labelled sequence
# --------------------------------------------------
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for index, sid in enumerate(ordered_train_ids):
        record = labels[sid]
        paths = image_paths_by_sequence[sid]

        assert len(paths) == 3, f"{sid} does not have three images."

        user_message = {
            "role": "user",
            "content": (
                [{"type": "image", "image": str(p)} for p in paths]
                + [{"type": "text", "text": instruction}]
            )
        }

        target = json.dumps(
            {"pages": record["pages"]},
            ensure_ascii=False
        )

        messages = [
            user_message,
            {"role": "assistant", "content": [
                {"type": "text", "text": target}
            ]}
        ]

        # Build the full conversation and prompt-only versions
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = processor.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        full_inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        prompt_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_len = full_inputs["input_ids"].shape[1]

        # Ensure answer positions align exactly with the prompt boundary
        if not torch.equal(
            full_inputs["input_ids"][0, :prompt_len],
            prompt_inputs["input_ids"][0]
        ):
            raise RuntimeError(
                f"Prompt alignment failed for {sid}; training stopped."
            )

        device = model.device
        model_inputs = {
            key: value.to(device)
            for key, value in full_inputs.items()
            if torch.is_tensor(value)
        }

        input_ids = model_inputs["input_ids"]
        target_ids = input_ids[:, prompt_len:]

        if target_ids.shape[1] == 0:
            raise RuntimeError(f"No target tokens found for {sid}.")

        # Forward through the transformer without full-sequence vocabulary logits
        base = model.get_base_model()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = base.model(
                **model_inputs,
                use_cache=False,
                return_dict=True
            )

            # Position t-1 predicts target token t
            answer_hidden = outputs.last_hidden_state[
                :, prompt_len - 1:-1, :
            ]

            if answer_hidden.shape[1] != target_ids.shape[1]:
                raise RuntimeError(
                    f"Loss alignment mismatch for {sid}: "
                    f"{answer_hidden.shape[1]} hidden positions vs "
                    f"{target_ids.shape[1]} targets."
                )

            # Calculate vocabulary logits only for answer positions
            answer_logits = base.lm_head(answer_hidden)

            loss = F.cross_entropy(
                answer_logits.float().reshape(
                    -1, answer_logits.shape[-1]
                ),
                target_ids.reshape(-1)
            )

            # Scale the loss for gradient accumulation
            scaled_loss = loss / GRAD_ACCUMULATION

        loss_value = float(loss.detach().float().cpu())
        scaled_loss.backward()

        running_loss += loss_value
        epoch_loss += loss_value

        # Release per-example activations before processing the next sequence
        del outputs, answer_hidden, answer_logits
        del scaled_loss, loss, target_ids, input_ids, model_inputs
        del full_inputs, prompt_inputs, image_inputs, video_inputs
        del messages, user_message, full_text, prompt_text, target
        gc.collect()

        # Update adapter weights after every four examples
        if (index + 1) % GRAD_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            # Save the latest adapter after each optimizer update
            model.save_pretrained(
                str(OUTPUT_DIR),
                safe_serialization=True
            )

        # Log every example so progress remains inspectable
        log_record = {
            "event": "example_completed",
            "epoch": epoch + 1,
            "index": index + 1,
            "sequence_id": sid,
            "loss": loss_value,
            "optimizer_steps": optimizer_steps,
            "elapsed_minutes": round(
                (time.time() - started) / 60, 2
            ),
        }

        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_record) + "\n")

        if (index + 1) % 4 == 0 or index == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Sequence {index + 1}/{total_examples} | "
                f"ID: {sid} | "
                f"Loss: {loss_value:.4f} | "
                f"Optimizer updates: {optimizer_steps} | "
                f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

    # All 64 examples divide evenly into groups of four
    assert total_examples % GRAD_ACCUMULATION == 0

    average_loss = epoch_loss / total_examples

    # Save the completed epoch and processor configuration
    model.save_pretrained(
        str(OUTPUT_DIR),
        safe_serialization=True
    )
    processor.save_pretrained(str(OUTPUT_DIR))

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "event": "epoch_completed",
            "epoch": epoch + 1,
            "average_training_loss": average_loss,
            "optimizer_steps": optimizer_steps,
            "elapsed_minutes": round(
                (time.time() - started) / 60, 2
            ),
        }) + "\n")

    print(f"\nEpoch {epoch + 1} complete.")
    print(f"Average training loss: {average_loss:.4f}")

print("\nTRAINING COMPLETE")
print("Optimizer updates:", optimizer_steps)
print("Adapter saved to:", OUTPUT_DIR)
print("Training log saved to:", LOG_PATH)
print(f"Total elapsed minutes: {(time.time() - started)/60:.2f}")
```

```python
import os
import json
import torch

paths = [
    "/content/manga_lora_adapter/trained_checkpoint",
    "/content/manga_qlora_adapter",
    "/content/manga_qlora_training_log.jsonl",
    "/content/manga_training_log.jsonl",
    "/content/baseline_predictions.jsonl",
    "/content/baseline_scores.json",
    "/content/score.py",
]

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("\nFILE STATUS")

for path in paths:
    if os.path.isdir(path):
        files = os.listdir(path)
        print(f"\n{path}/")
        print("  Exists; contents:", files[:15])
    elif os.path.isfile(path):
        print(f"\n{path}")
        if path.endswith(".jsonl") and "training_log" in path:
            with open(path) as f:
                lines = f.readlines()
            print("  Log entries:", len(lines))
            print("  Last entries:", "".join(lines[-3:]))
        else:
            print("  Exists; size:", os.path.getsize(path), "bytes")
    else:
        print(f"\n{path}: NOT FOUND")
```

```python
import inspect
from qwen_vl_utils import process_vision_info

print("Image-processing function:", inspect.signature(process_vision_info))
print("\nGPU memory currently allocated:",
      round(__import__("torch").cuda.memory_allocated() / 1024**3, 2), "GB")
print("GPU memory currently reserved:",
      round(__import__("torch").cuda.memory_reserved() / 1024**3, 2), "GB")
```

Session is being restarted again.

```python
import os
import torch

required_files = [
    "/content/manga_dataset/train_val_split_seed42.json",
    "/content/labels (1).jsonl",
    "/content/sequences.json",
    "/content/baseline_predictions.jsonl",
    "/content/baseline_scores.json",
    "/content/score.py",
    "/content/manga_qlora_training_log.jsonl",
]

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
print("\nFile verification:")

for path in required_files:
    print(("FOUND  " if os.path.exists(path) else "MISSING"), path)

print("\nGPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
print("GPU memory reserved:",
      round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")
```

```python
import os

for root, dirs, files in os.walk("/content"):
    dirs[:] = [
        d for d in dirs
        if d not in {".cache", "__pycache__"}
    ]

    relevant = (
        "adapter" in root.lower()
        or "checkpoint" in root.lower()
        or "training_log" in root.lower()
        or "trainer_state" in root.lower()
    )

    if relevant:
        print(f"\n📁 {root}")
        for name in files[:20]:
            print("  ", name)
```

```python
from PIL import Image
import os
import json

# Load the existing train/validation split
with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

sequence_id = split["train_ids"][0]
image_dir = f"/content/manga_dataset/images-20261010T124807Z-1-001/images/{sequence_id}"

# Resize each page to a maximum image area of 512 × 512 pixels.
# Preserve aspect ratio so the page is not distorted.
MAX_PIXELS = 512 * 512

print("Sequence:", sequence_id)
print("Image directory exists:", os.path.isdir(image_dir))

for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    path = os.path.join(image_dir, filename)

    with Image.open(path) as image:
        original_size = image.size
        resized = image.copy()
        resized.thumbnail((512, 512), Image.Resampling.LANCZOS)

        print(
            f"{filename}: "
            f"original={original_size}, "
            f"resized={resized.size}, "
            f"pixels={resized.width * resized.height:,}"
        )
```

```python
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import torch

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

sequence_id = split["train_ids"][0]
image_dir = (
    f"/content/manga_dataset/images-20261010T124807Z-1-001/"
    f"images/{sequence_id}"
)

# Explicitly resize all three pages in memory.
images = []
for filename in sorted(
    f for f in __import__("os").listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
):
    with Image.open(f"{image_dir}/{filename}") as im:
        page = im.convert("RGB")
        page.thumbnail((512, 512), Image.Resampling.LANCZOS)
        images.append(page.copy())

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": page}
        for page in images
    ] + [{
        "type": "text",
        "text": "Read the dialogue on these three manga pages."
    }]
}]

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ),
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
)

print("Pages processed:", len(images))
print("Pixel tensor shape:", tuple(inputs["pixel_values"].shape))
print("Image grid (T, H, W):", inputs["image_grid_thw"].tolist())
print("Total visual grid cells:", int(inputs["image_grid_thw"][:, 1:].prod(dim=1).sum()))
```

```python
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

print("\nModel device:", next(model.parameters()).device)
print("4-bit quantization:", model.is_loaded_in_4bit)
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
```

```python
import os
import json
import gc
import torch
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------
# 1. Load one training example and its ground truth
# --------------------------------------------------
with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

train_id = split["train_ids"][0]

labels_path = "/content/labels (1).jsonl"
labels_by_id = {}

with open(labels_path) as f:
    for line in f:
        item = json.loads(line)
        labels_by_id[item["sequence_id"]] = item

assert train_id in labels_by_id, f"Missing labels for {train_id}"

image_dir = (
    "/content/manga_dataset/"
    "images-20261010T124807Z-1-001/images/"
    + train_id
)

image_files = sorted(
    name for name in os.listdir(image_dir)
    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
)

assert len(image_files) == 3, (
    f"Expected 3 pages, found {len(image_files)}"
)

images = []

for name in image_files:
    with Image.open(os.path.join(image_dir, name)) as im:
        page = im.convert("RGB")
        page.thumbnail((512, 512), Image.Resampling.LANCZOS)
        images.append(page.copy())

target = {
    "sequence_id": train_id,
    "pages": labels_by_id[train_id]["pages"],
}

target_text = json.dumps(target, ensure_ascii=False)

print("Sequence:", train_id)
print("Pages:", len(images))
print("Target characters:", len(target_text))

# --------------------------------------------------
# 2. Build user prompt and assistant target
# --------------------------------------------------
user_message = {
    "role": "user",
    "content": [
        {"type": "image", "image": image}
        for image in images
    ] + [{
        "type": "text",
        "text": (
            "Transcribe the dialogue from these three consecutive "
            "English manga pages in reading order. Preserve page "
            "boundaries, punctuation, and consistent speaker IDs. "
            "Exclude sound effects, credits, signs, and other "
            "non-dialogue text. Return only valid JSON with keys "
            "'sequence_id' and 'pages'. Each page must be an array "
            "of objects containing 'speaker' and 'text'."
        )
    }]
}

full_messages = [
    user_message,
    {"role": "assistant", "content": target_text}
]

prompt_text = processor.apply_chat_template(
    [user_message],
    tokenize=False,
    add_generation_prompt=True,
)

full_text = processor.apply_chat_template(
    full_messages,
    tokenize=False,
    add_generation_prompt=False,
)

# Tokenize the prompt and full example to verify alignment.
prompt_inputs = processor(
    text=[prompt_text],
    images=images,
    return_tensors="pt",
)

prompt_ids = prompt_inputs["input_ids"]
prompt_len = prompt_ids.shape[1]

del prompt_inputs, prompt_ids
gc.collect()
torch.cuda.empty_cache()

full_inputs = processor(
    text=[full_text],
    images=images,
    return_tensors="pt",
)

full_ids = full_inputs["input_ids"]
full_len = full_ids.shape[1]

assert full_len > prompt_len, "No target tokens found"

# Ensure the prompt is an exact token prefix of the full example.
check_prompt = processor(
    text=[prompt_text],
    images=images,
    return_tensors="pt",
)["input_ids"]

assert torch.equal(
    check_prompt,
    full_ids[:, :prompt_len]
), "Prompt/target token alignment failed"

del check_prompt
gc.collect()
torch.cuda.empty_cache()

target_ids = full_ids[:, prompt_len:].to("cuda")

# Move model inputs to the GPU.
model_inputs = {
    key: value.to("cuda") if torch.is_tensor(value) else value
    for key, value in full_inputs.items()
}

# --------------------------------------------------
# 3. Forward + backward pass; NO optimizer update
# --------------------------------------------------
model.train()
model.zero_grad(set_to_none=True)

base = model.get_base_model()

torch.cuda.reset_peak_memory_stats()

try:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = base.model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
        )

        # Predict target tokens only, reducing logits memory.
        answer_hidden = outputs.last_hidden_state[
            :, prompt_len - 1:-1, :
        ]

        answer_logits = base.lm_head(answer_hidden)

        assert answer_logits.shape[1] == target_ids.shape[1], (
            f"Logit/target mismatch: {answer_logits.shape[1]} vs "
            f"{target_ids.shape[1]}"
        )

        loss = F.cross_entropy(
            answer_logits.float().reshape(-1, answer_logits.shape[-1]),
            target_ids.reshape(-1),
        )

    loss.backward()

    grad_tensors = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    finite_gradients = all(
        torch.isfinite(g).all().item()
        for g in grad_tensors
    )

    print("\nDRY RUN RESULT")
    print("Prompt tokens:", prompt_len)
    print("Full sequence tokens:", full_len)
    print("Target tokens:", target_ids.shape[1])
    print("Training loss:", round(loss.item(), 4))
    print("Adapter tensors with gradients:", len(grad_tensors))
    print("Gradients finite:", finite_gradients)
    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )
    print("Optimizer update performed: NO")

finally:
    model.zero_grad(set_to_none=True)
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
```

Training 64 sequences

```python
import os
import json
import gc
import time
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

# ---------------- CONFIG ----------------
BASE_DIR = "/content/manga_dataset"
IMAGE_ROOT = (
    "/content/manga_dataset/"
    "images-20261010T124807Z-1-001/images"
)
LABELS_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = f"{BASE_DIR}/train_val_split_seed42.json"
OUTPUT_DIR = "/content/manga_qlora_adapter"

MAX_SIDE = 512
LEARNING_RATE = 5e-5
EPOCHS = 1
SAVE_EVERY = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
with open(SPLIT_PATH) as f:
    split = json.load(f)

with open(LABELS_PATH) as f:
    label_rows = [json.loads(line) for line in f if line.strip()]

labels_by_id = {x["sequence_id"]: x for x in label_rows}
train_ids = split["train_ids"]
validation_ids = split["validation_ids"]

assert len(train_ids) == 64, f"Expected 64 train IDs, got {len(train_ids)}"
assert len(validation_ids) == 16
assert not set(train_ids).intersection(validation_ids)
assert all(sid in labels_by_id for sid in train_ids)

print("Training sequences:", len(train_ids))
print("Validation sequences reserved:", len(validation_ids))

# ---------------- HELPERS ----------------
def load_resized_pages(sequence_id):
    folder = os.path.join(IMAGE_ROOT, sequence_id)
    names = sorted(
        n for n in os.listdir(folder)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if len(names) != 3:
        raise ValueError(f"{sequence_id}: expected 3 pages, found {len(names)}")

    pages = []
    for name in names:
        with Image.open(os.path.join(folder, name)) as im:
            page = im.convert("RGB")
            page.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            pages.append(page.copy())
    return pages

def make_messages(sequence_id, pages):
    target = {
        "sequence_id": sequence_id,
        "pages": labels_by_id[sequence_id]["pages"],
    }
    target_text = json.dumps(target, ensure_ascii=False)

    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{
            "type": "text",
            "text": (
                "Transcribe dialogue from these three consecutive English "
                "manga pages in reading order. Preserve page boundaries, "
                "punctuation, and consistent speaker IDs. Exclude sound "
                "effects, credits, signs, and other non-dialogue text. "
                "Return only valid JSON with keys sequence_id and pages. "
                "Each page must be an array of objects containing speaker "
                "and text."
            )
        }]
    }
    full_messages = [
        user_message,
        {"role": "assistant", "content": target_text}
    ]

    prompt_text = processor.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text

def prepare_example(sequence_id):
    pages = load_resized_pages(sequence_id)
    prompt_text, full_text = make_messages(sequence_id, pages)

    # Process prompt and full conversation using the same three images.
    prompt_batch = processor(
        text=[prompt_text],
        images=pages,
        return_tensors="pt",
    )
    full_batch = processor(
        text=[full_text],
        images=pages,
        return_tensors="pt",
    )

    prompt_ids = prompt_batch["input_ids"]
    full_ids = full_batch["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Exact prefix match is required before using this masking scheme.
    if not torch.equal(prompt_ids, full_ids[:, :prompt_len]):
        raise ValueError(
            f"{sequence_id}: prompt tokens do not match full-input prefix"
        )
    if full_ids.shape[1] <= prompt_len:
        raise ValueError(f"{sequence_id}: target has no tokens")

    # Labels use standard causal LM loss. Prompt positions are ignored.
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    # Move tensors to model's device.
    device = next(model.parameters()).device
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in full_batch.items()
    }
    labels = labels.to(device)

    del prompt_batch, full_batch, prompt_ids, full_ids
    return batch, labels

# ---------------- TRAIN ----------------
model.config.use_cache = False
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
)

log_path = f"{OUTPUT_DIR}/training_log.jsonl"
optimizer_steps = 0
start_time = time.time()

with open(log_path, "a") as log:
    log.write(json.dumps({
        "event": "training_started",
        "train_sequences": len(train_ids),
        "validation_sequences": len(validation_ids),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "image_max_side": MAX_SIDE,
        "loss_method": "standard causal LM loss with prompt masking",
    }) + "\n")

    for epoch in range(EPOCHS):
        # Deterministic order makes this run easier to reproduce.
        for index, sequence_id in enumerate(train_ids, start=1):
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            try:
                batch, labels = prepare_example(sequence_id)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(
                        **batch,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = outputs.loss

                if not torch.isfinite(loss).item():
                    raise ValueError(
                        f"Non-finite loss for {sequence_id}: {loss.item()}"
                    )

                loss.backward()

                grads = [
                    p.grad for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if not grads or not all(
                    torch.isfinite(g).all().item() for g in grads
                ):
                    raise ValueError(
                        f"Missing or non-finite adapter gradients: {sequence_id}"
                    )

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )

                optimizer.step()
                optimizer_steps += 1

                record = {
                    "event": "example_completed",
                    "epoch": epoch + 1,
                    "index": index,
                    "sequence_id": sequence_id,
                    "loss": float(loss.detach().item()),
                    "optimizer_steps": optimizer_steps,
                    "elapsed_minutes": round(
                        (time.time() - start_time) / 60, 2
                    ),
                    "peak_gpu_memory_gb": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 2
                    ),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()

                print(
                    f"[{index}/{len(train_ids)}] "
                    f"{sequence_id} | loss={record['loss']:.4f} | "
                    f"updates={optimizer_steps} | "
                    f"peak={record['peak_gpu_memory_gb']:.2f} GB"
                )

                # Save after every successful optimizer update.
                if optimizer_steps % SAVE_EVERY == 0:
                    model.save_pretrained(OUTPUT_DIR)
                    processor.save_pretrained(OUTPUT_DIR)

                    with open(f"{OUTPUT_DIR}/training_state.json", "w") as sf:
                        json.dump({
                            "epoch": epoch + 1,
                            "last_train_index": index,
                            "optimizer_steps": optimizer_steps,
                            "last_sequence_id": sequence_id,
                        }, sf, indent=2)

                del outputs, loss, batch, labels, grads
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as exc:
                log.write(json.dumps({
                    "event": "training_error",
                    "epoch": epoch + 1,
                    "index": index,
                    "sequence_id": sequence_id,
                    "error": repr(exc),
                    "optimizer_steps": optimizer_steps,
                }) + "\n")
                log.flush()
                raise

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/training_state.json", "w") as f:
    json.dump({
        "completed": True,
        "optimizer_steps": optimizer_steps,
        "epochs": EPOCHS,
    }, f, indent=2)

print("\nTRAINING COMPLETE")
print("Optimizer updates:", optimizer_steps)
print("Adapter directory:", OUTPUT_DIR)
print("Training log:", log_path)
```

```python
import os
import json
from PIL import Image

# 1. Preserve the current model separately
RECOVERY_DIR = "/content/manga_qlora_recovery_63"

model.save_pretrained(RECOVERY_DIR)
processor.save_pretrained(RECOVERY_DIR)

with open(f"{RECOVERY_DIR}/recovery_state.json", "w") as f:
    json.dump({
        "completed_optimizer_updates": 63,
        "epochs_completed": 0,
        "last_successful_sequence": "seq_f9a4e67ae3c018f6",
        "failed_sequence": "seq_fc4c2b84b6ec32ff",
        "status": "partial_training_recovery"
    }, f, indent=2)

print("Recovery checkpoint saved:", RECOVERY_DIR)

# 2. Inspect the failed sequence
sid = "seq_fc4c2b84b6ec32ff"
row = labels_by_id[sid]

print("\nFailed sequence:", sid)
print("Number of pages:", len(row["pages"]))
print("Lines per page:", [len(p) for p in row["pages"]])

for page_num, page in enumerate(row["pages"], 1):
    print(
        f"Page {page_num}:",
        "characters =", sum(len(x["text"]) for x in page),
        "| invalid lines =", sum(
            not isinstance(x.get("text"), str) or not x.get("text", "").strip()
            for x in page
        )
    )

folder = os.path.join(IMAGE_ROOT, sid)

for name in sorted(os.listdir(folder)):
    path = os.path.join(folder, name)
    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        with Image.open(path) as im:
            print(name, "| size:", im.size, "| mode:", im.mode)

print("\nGPU allocated:",
      round(__import__("torch").cuda.memory_allocated() / 1024**3, 2), "GB")
```

```python
import os, json

RECOVERY_DIR = "/content/manga_qlora_recovery_63"

print("Checkpoint exists:", os.path.isdir(RECOVERY_DIR))
print("\nSaved files:")
for name in sorted(os.listdir(RECOVERY_DIR)):
    path = os.path.join(RECOVERY_DIR, name)
    print(f"  {name}  ({os.path.getsize(path) / 1024**2:.2f} MB)"
          if os.path.isfile(path) else f"  {name}/")

with open(os.path.join(RECOVERY_DIR, "recovery_state.json")) as f:
    state = json.load(f)

print("\nRecovery status:")
print(json.dumps(state, indent=2))
```

```python
import os, json

print("Recovery checkpoint:",
      os.path.isdir("/content/manga_qlora_recovery_63"))

print("Split file:",
      os.path.exists("/content/manga_dataset/train_val_split_seed42.json"))

print("Scorer:",
      os.path.exists("/content/score.py"))

with open("/content/manga_dataset/train_val_split_seed42.json") as f:
    split = json.load(f)

print("Validation sequences:", len(split["validation_ids"]))
print("First 3 validation IDs:", split["validation_ids"][:3])

print("Model object in memory:", "model" in globals())
print("Processor object in memory:", "processor" in globals())
```

```python
import os
import json
import torch

CHECKPOINT_DIR = "/content/manga_qlora_recovery_63"
SPLIT_PATH = "/content/manga_dataset/train_val_split_seed42.json"

assert os.path.exists(
    os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")
), "Recovery adapter not found"

with open(SPLIT_PATH) as f:
    split = json.load(f)

validation_ids = split["validation_ids"]
train_ids = split["train_ids"]

assert len(validation_ids) == 16
assert not set(train_ids).intersection(validation_ids)

print("Checkpoint: verified")
print("Validation sequences:", len(validation_ids))
print("Training/validation overlap:", len(set(train_ids) & set(validation_ids)))
print("GPU:", torch.cuda.get_device_name(0))
print("Ready to evaluate.")
```

```python
import os

# Locate the existing baseline inference artifacts
for path in [
    "/content/baseline_predictions.jsonl",
    "/content/baseline_scores.json",
    "/content/score.py",
]:
    print(f"\n{path}:",
          "FOUND" if os.path.exists(path) else "MISSING")

# Find the original inference code saved in the notebook's history
print("\nSearch your earlier notebook cells for:")
print("model.generate(")
print("baseline_predictions.jsonl")
print("qwen_vl_utils")
```

```python
import os
import json
import re
import torch
import subprocess
from contextlib import nullcontext
from PIL import Image

# ---------- Paths and settings ----------
ROOT = "/content/manga_dataset"
IMAGE_ROOT = (
    ROOT + "/images-20261010T124807Z-1-001/images"
)
LABELS_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"
SCORE_SCRIPT = "/content/score.py"
OUT = "/content/manga_evaluation"

os.makedirs(OUT, exist_ok=True)

# ---------- Load labels and validation split ----------
with open(SPLIT_PATH) as f:
    split = json.load(f)

with open(LABELS_PATH) as f:
    label_rows = [json.loads(line) for line in f if line.strip()]

labels_by_id = {x["sequence_id"]: x for x in label_rows}
val_ids = split["validation_ids"]

assert len(val_ids) == 16
assert all(sid in labels_by_id for sid in val_ids)

references_path = OUT + "/validation_references.jsonl"

with open(references_path, "w") as f:
    for sid in val_ids:
        ref = {
            "sequence_id": sid,
            "pages": labels_by_id[sid]["pages"],
        }
        f.write(json.dumps(ref, ensure_ascii=False) + "\n")

# ---------- Resize validation images exactly as in training ----------
def load_pages(sid):
    folder = os.path.join(IMAGE_ROOT, sid)
    names = sorted(
        n for n in os.listdir(folder)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    assert len(names) == 3, f"{sid}: expected 3 pages"

    pages = []
    for name in names:
        with Image.open(os.path.join(folder, name)) as im:
            page = im.convert("RGB")
            page.thumbnail((512, 512), Image.Resampling.LANCZOS)
            pages.append(page.copy())
    return pages

PROMPT = (
    "Transcribe the dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound effects, "
    "credits, signs, and other non-dialogue text. Return only valid "
    "JSON with keys 'sequence_id' and 'pages'. Each page must be an "
    "array of objects containing 'speaker' and 'text'."
)

def parse_json(text):
    # Try a complete JSON object, including one inside a code fence.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    # Page-level fallback may return a JSON array.
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None

def clean_page(page):
    if not isinstance(page, list):
        return []
    return [
        {"speaker": str(x["speaker"]), "text": str(x["text"])}
        for x in page
        if isinstance(x, dict) and "speaker" in x and "text" in x
    ]

def generate_text(messages, use_adapter):
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    pages = [
        item["image"] for item in messages[0]["content"]
        if item.get("type") == "image"
    ]

    inputs = processor(
        text=[prompt_text],
        images=pages,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    context = nullcontext() if use_adapter else model.disable_adapter()

    with context, torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                **inputs,
                max_new_tokens=900,
                do_sample=False,
                use_cache=True,
            )

    prompt_length = inputs["input_ids"].shape[1]
    generated = output[:, prompt_length:]
    result = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, output, generated
    return result

def run_one(sid, use_adapter):
    pages = load_pages(sid)
    user_message = {
        "role": "user",
        "content": [
            *[{"type": "image", "image": p} for p in pages],
            {"type": "text", "text": PROMPT},
        ],
    }

    raw = generate_text([user_message], use_adapter)
    parsed = parse_json(raw)

    # If sequence-level JSON is malformed, try each page individually.
    if not isinstance(parsed, dict) or not isinstance(parsed.get("pages"), list):
        recovered = []
        for page in pages:
            page_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": page},
                    {"type": "text", "text": (
                        "Transcribe only dialogue from this manga page. "
                        "Return a JSON array of objects with speaker and text. "
                        "Exclude sound effects, credits, and signs."
                    )},
                ],
            }
            page_raw = generate_text([page_message], use_adapter)
            page_parsed = parse_json(page_raw)
            if isinstance(page_parsed, dict):
                page_parsed = page_parsed.get("dialogue", page_parsed.get("pages"))
            recovered.append(clean_page(page_parsed))

        result = {"sequence_id": sid, "pages": recovered}
        return result, raw, True

    predicted_pages = parsed["pages"][:3]
    while len(predicted_pages) < 3:
        predicted_pages.append([])

    result = {
        "sequence_id": sid,
        "pages": [clean_page(p) for p in predicted_pages],
    }
    return result, raw, False

# ---------- Generate both conditions ----------
model.eval()
model.config.use_cache = True

conditions = [
    ("pretrained_resized", False),
    ("finetuned_resized", True),
]

for condition, use_adapter in conditions:
    pred_path = f"{OUT}/{condition}_predictions.jsonl"
    raw_path = f"{OUT}/{condition}_raw.jsonl"
    print(f"\nEvaluating: {condition}")

    with open(pred_path, "w") as pred_file, open(raw_path, "w") as raw_file:
        for i, sid in enumerate(val_ids, start=1):
            prediction, raw, used_fallback = run_one(sid, use_adapter)
            pred_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            raw_file.write(json.dumps({
                "sequence_id": sid,
                "raw_output": raw,
                "page_fallback_used": used_fallback,
            }, ensure_ascii=False) + "\n")
            pred_file.flush()
            raw_file.flush()
            print(
                f"[{i}/16] {sid} | "
                f"page fallback: {used_fallback}"
            )

    score_path = f"{OUT}/{condition}_scores.json"
    result = subprocess.run(
        [
            "python", SCORE_SCRIPT,
            "--references", references_path,
            "--predictions", pred_path,
            "--output", score_path,
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("SCORER ERROR:")
        print(result.stderr[-3000:])
        raise RuntimeError(f"Scoring failed for {condition}")

    print("Saved predictions:", pred_path)
    print("Saved scores:", score_path)

print("\nBoth evaluation runs finished.")
print("Results directory:", OUT)
```

Training the model again

```python

# Find existing dataset and collator variables in this Colab session
for name, obj in list(globals().items()):
    if "dataset" in name.lower() or "collator" in name.lower():
        try:
            print(name, "| type:", type(obj).__name__,
                  "| length:", len(obj) if hasattr(obj, "__len__") else "n/a")
        except Exception:
            print(name, "| type:", type(obj).__name__)
```

```python
print("hello")
```

```python

# Find existing dataset and collator variables in this Colab session
for name, obj in list(globals().items()):
    if "dataset" in name.lower() or "collator" in name.lower():
        try:
            print(name, "| type:", type(obj).__name__,
                  "| length:", len(obj) if hasattr(obj, "__len__") else "n/a")
        except Exception:
            print(name, "| type:", type(obj).__name__)
```

```python

import os

print("Current directory:", os.getcwd())
print("\nFiles and folders in /content:")

for item in os.listdir("/content"):
    path = os.path.join("/content", item)
    kind = "DIR " if os.path.isdir(path) else "FILE"
    print(f"{kind}  {item}")
```

```python

import os, json

root = "/content/manga_dataset"

print("MANGA DATASET CONTENTS")
for current, dirs, files in os.walk(root):
    level = current.replace(root, "").count(os.sep)
    if level > 2:
        dirs[:] = []
        continue
    print(f"\n{current}/")
    for f in files[:12]:
        print("  ", f)
```

```python

import json

path = "/content/manga_dataset/train_val_split_seed42.json"

with open(path, "r", encoding="utf-8") as f:
    split = json.load(f)

print("Top-level type:", type(split).__name__)
if isinstance(split, dict):
    for key, value in split.items():
        print(f"\n{key}: type={type(value).__name__}")
        if isinstance(value, list):
            print("Count:", len(value))
            print("First item:", value[0] if value else "(empty)")
        elif isinstance(value, dict):
            print("Keys:", list(value.keys())[:15])
        else:
            print("Value:", value)
else:
    print("Count:", len(split))
    print("First item:", split[0] if split else "(empty)")
```

```python

import json, os

# Show one label record
with open("/content/labels (1).jsonl", "r", encoding="utf-8") as f:
    first_label = next(line for line in f if line.strip())
print("LABEL EXAMPLE:")
print(json.dumps(json.loads(first_label), indent=2)[:3500])

# Show a few image filenames
image_root = "/content/manga_dataset/images-20261010T124807Z-1-001/images"
print("\nIMAGE FILE EXAMPLES:")
for current, _, files in os.walk(image_root):
    for name in sorted(files)[:8]:
        print(os.path.join(current, name))
    break
```

```python

import os

root = "/content/manga_dataset"
found = []

for current, _, files in os.walk(root):
    for name in files:
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            found.append(os.path.join(current, name))

print("Total image files:", len(found))
for path in found[:15]:
    print(path)
```

Training again

```python

# SINGLE CELL: Rebuild data + train frozen-vision, language-only LoRA
# + generation helpers with repetition control, JSON validation, and page fallback.

import os, json, re, torch
from PIL import Image
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ---- 1. Paths and settings ----
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
LABELS_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"
OUTPUT_DIR = "/content/manga_language_only_r4_768"

# Qwen image-pixel budget; this is not a literal 768-pixel image height.
MAX_PIXELS = 768 * 28 * 28
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 2. Rebuild the training examples from existing files ----
with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

with open(LABELS_PATH, encoding="utf-8") as f:
    labels = {
        item["sequence_id"]: item
        for line in f if line.strip()
        for item in [json.loads(line)]
    }

train_ids = split["train_ids"]
assert all(sid in labels for sid in train_ids), \
    "Some training IDs are missing from the labels file."

def make_example(sid):
    paths = [os.path.join(IMAGE_ROOT, sid, f"{i:02d}.png")
             for i in (1, 2, 3)]
    assert all(os.path.isfile(p) for p in paths), \
        f"Missing page image for {sid}: {paths}"

    target = {
        "sequence_id": sid,
        "pages": labels[sid]["pages"],
    }
    return {"sequence_id": sid, "paths": paths,
            "target_text": json.dumps(target, ensure_ascii=False)}

train_examples = [make_example(sid) for sid in train_ids]
print(f"Rebuilt {len(train_examples)} training sequences.")

# ---- 3. Load processor and quantized base model ----
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PIXELS,
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

# Explicitly freeze the vision encoder.
vision_params = [
    (n, p) for n, p in model.named_parameters()
    if n.startswith("visual.") or ".visual." in n
]
assert vision_params, "Vision encoder not found; training stopped."
for _, param in vision_params:
    param.requires_grad = False

# ---- 4. Language-model-only LoRA, rank 4 ----
target_pattern = (
    r".*language_model.*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
matched = [
    n for n, m in model.named_modules()
    if re.fullmatch(target_pattern, n)
    and isinstance(m, torch.nn.Linear)
]
assert matched, "No language-model LoRA targets matched."

model = get_peft_model(
    model,
    LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=target_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
unexpected = [
    n for n in trainable
    if "language_model" not in n or "lora_" not in n
]
assert not unexpected, f"Unexpected trainable parameters: {unexpected[:10]}"
assert not any(
    p.requires_grad for n, p in model.named_parameters()
    if n.startswith("visual.") or ".visual." in n
), "Vision encoder is trainable."

model.print_trainable_parameters()
print(f"Vision frozen | matched language modules: {len(matched)}")
print(f"Processor max-pixel budget per image: {MAX_PIXELS}")

# ---- 5. Preserve your original training prompt and output format ----
PROMPT = (
    "Transcribe dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound "
    "effects, credits, signs, and other non-dialogue text. "
    "Return only valid JSON with keys sequence_id and pages. "
    "Each page must be an array of objects containing speaker "
    "and text."
)

def make_messages(pages, prompt=PROMPT):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{"type": "text", "text": prompt}],
    }]

def training_collator(examples):
    # One sequence per batch; use original PNGs and let the processor resize.
    ex = examples[0]
    pages = [Image.open(p).convert("RGB") for p in ex["paths"]]
    user_message = make_messages(pages)[0]
    full_messages = [
        user_message,
        {"role": "assistant", "content": ex["target_text"]},
    ]

    prompt_text = processor.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    image_inputs, video_inputs = process_vision_info([user_message])

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    labels_tensor = full_inputs["input_ids"].clone()
    prompt_length = prompt_inputs["input_ids"].shape[1]
    labels_tensor[:, :prompt_length] = -100
    full_inputs["labels"] = labels_tensor

    # Close source images after the processor has created tensors.
    for page in pages:
        page.close()
    return full_inputs

# ---- 6. Train once and save to a separate adapter directory ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_examples,
    data_collator=training_collator,
)

train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete:", OUTPUT_DIR)
print(train_result.metrics)

# ---- 7. Reusable generation + JSON validation + page fallback ----
def parse_json(raw):
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE
    ).strip()
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(cleaned[i:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("Generated output is not valid JSON.")

def clean_adjacent_duplicates(page):
    # Remove only immediately repeated, exactly identical dialogue entries.
    result = []
    for item in page:
        if not (result and item == result[-1]):
            result.append(item)
    return result

def generate_for_pages(pages, prompt, max_new_tokens=1800):
    messages = make_messages(pages, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

def generate_sequence_with_fallback(sequence_id, image_paths):
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    try:
        raw = generate_for_pages(pages, PROMPT)
        try:
            result = parse_json(raw)
            valid = (
                isinstance(result, dict)
                and result.get("sequence_id") == sequence_id
                and isinstance(result.get("pages"), list)
                and len(result["pages"]) == 3
                and all(
                    isinstance(page, list)
                    and all(
                        isinstance(line, dict)
                        and "speaker" in line and "text" in line
                        for line in page
                    )
                    for page in result["pages"]
                )
            )
            if not valid:
                raise ValueError("JSON does not match the required schema.")
            result["pages"] = [
                clean_adjacent_duplicates(page) for page in result["pages"]
            ]
            return {"result": result, "used_page_fallback": False,
                    "raw_output": raw}
        except ValueError:
            # Retry each page separately if the combined response is invalid.
            page_results, page_raw = [], []
            for page in pages:
                page_prompt = (
                    "Transcribe dialogue from this single English manga page "
                    "in reading order. Preserve punctuation and speaker IDs. "
                    "Exclude sound effects, credits, signs, and other "
                    "non-dialogue text. Return only a valid JSON array of "
                    "objects containing speaker and text."
                )
                raw_page = generate_for_pages([page], page_prompt)
                parsed = parse_json(raw_page)
                if not isinstance(parsed, list) or not all(
                    isinstance(line, dict)
                    and "speaker" in line and "text" in line
                    for line in parsed
                ):
                    raise ValueError("Page fallback returned invalid JSON schema.")
                page_results.append(clean_adjacent_duplicates(parsed))
                page_raw.append(raw_page)

            return {
                "result": {"sequence_id": sequence_id, "pages": page_results},
                "used_page_fallback": True,
                "raw_output": raw,
                "page_raw_outputs": page_raw,
            }
    finally:
        for page in pages:
            page.close()

print("Training and generation helpers are ready.")
print("Use generate_sequence_with_fallback(sequence_id, [page1, page2, page3]) for inference.")
```

Well increasing the resolution didn't work well.
Let's clean up.
---

```python

import gc, torch

# Release memory held by the failed training attempt.
for name in ["trainer", "model", "train_result"]:
    if name in globals():
        del globals()[name]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
    print("GPU reserved:", round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")

print("Cleanup complete. Existing files and labels are unchanged.")
```

Restarting again to clean up the memory

```python

import os, torch

required = [
    "/content/manga_dataset/train_val_split_seed42.json",
    "/content/labels (1).jsonl",
    "/content/manga_dataset/images-20261010T124807Z-1-001/images",
]

for path in required:
    print(("FOUND  " if os.path.exists(path) else "MISSING ") + path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
else:
    print("\nNo GPU detected. Enable a GPU in Runtime settings.")
```

```python

# FINAL TRAINING CELL — frozen vision + rank-4 language-only LoRA
# Uses existing files, preserves the original adapter, and limits image pixels.

import os, json, re, gc, torch
from PIL import Image
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
LABELS_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"
OUTPUT_DIR = "/content/manga_language_only_r4_401k"
MAX_PIXELS = 401408

assert torch.cuda.is_available(), "Enable a GPU before training."
assert all(os.path.exists(p) for p in [LABELS_PATH, SPLIT_PATH, IMAGE_ROOT]), \
    "A required dataset path is missing."

# 1. Rebuild training records from the existing labels and split.
with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)
with open(LABELS_PATH, encoding="utf-8") as f:
    labels = {
        item["sequence_id"]: item
        for line in f if line.strip()
        for item in [json.loads(line)]
    }

def make_example(sid):
    paths = [os.path.join(IMAGE_ROOT, sid, f"{i:02d}.png")
             for i in (1, 2, 3)]
    assert all(os.path.isfile(p) for p in paths), f"Missing page for {sid}"
    target = json.dumps(
        {"sequence_id": sid, "pages": labels[sid]["pages"]},
        ensure_ascii=False
    )
    return {"sequence_id": sid, "paths": paths, "target_text": target}

train_examples = [make_example(sid) for sid in split["train_ids"]]
assert len(train_examples) == 64
print("Training sequences:", len(train_examples))

# 2. Processor and 4-bit base model.
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PIXELS,
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

# 3. Freeze the vision encoder explicitly.
vision_params = [
    (name, param) for name, param in model.named_parameters()
    if name.startswith("visual.") or ".visual." in name
]
assert vision_params, "Vision encoder not found."
for _, param in vision_params:
    param.requires_grad = False

# 4. Attach rank-4 LoRA only to language-model layers.
target_pattern = (
    r".*language_model.*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
matched = [
    name for name, module in model.named_modules()
    if re.fullmatch(target_pattern, name)
    and isinstance(module, torch.nn.Linear)
]
assert matched, "No language-model LoRA targets matched."

model = get_peft_model(
    model,
    LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=target_pattern,
        bias="none", task_type="CAUSAL_LM",
    ),
)

trainable_names = [
    name for name, param in model.named_parameters() if param.requires_grad
]
unexpected = [
    name for name in trainable_names
    if "language_model" not in name or "lora_" not in name
]
assert not unexpected, f"Unexpected trainable parameters: {unexpected[:10]}"
assert not any(
    p.requires_grad for name, p in model.named_parameters()
    if name.startswith("visual.") or ".visual." in name
), "Vision encoder is trainable."

model.print_trainable_parameters()
print("Vision frozen; language-model modules matched:", len(matched))
print("Max pixel budget:", MAX_PIXELS)

# 5. Original training prompt.
PROMPT = (
    "Transcribe dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound "
    "effects, credits, signs, and other non-dialogue text. "
    "Return only valid JSON with keys sequence_id and pages. "
    "Each page must be an array of objects containing speaker "
    "and text."
)

def build_user_message(pages):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{"type": "text", "text": PROMPT}],
    }

# 6. Collator: mask prompt tokens so loss is computed only on the answer.
# Full conversation is tokenized once, avoiding separate image-token counts.
def training_collator(examples):
    batch = []
    for ex in examples:
        pages = [Image.open(p).convert("RGB") for p in ex["paths"]]
        user_msg = build_user_message(pages)
        full_messages = [
            user_msg,
            {"role": "assistant", "content": ex["target_text"]},
        ]

        # Process the full conversation in one call.
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info([user_msg])
        full = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # Get prompt token length using the same user message and images.
        prompt_text = processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )
        prompt = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        labels_tensor = full["input_ids"].clone()
        prompt_length = prompt["input_ids"].shape[1]
        labels_tensor[:, :prompt_length] = -100
        full["labels"] = labels_tensor

        for page in pages:
            page.close()
        batch.append(full)

    # Per-device batch size is 1; return the single prepared example.
    assert len(batch) == 1, "This collator expects batch size 1."
    return batch[0]

# 7. Train and save to a new directory.
os.makedirs(OUTPUT_DIR, exist_ok=True)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_examples,
    data_collator=training_collator,
)

train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete. New adapter:", OUTPUT_DIR)
print(train_result.metrics)

# 8. Generation helpers: repetition penalty, JSON validation, page fallback.
def parse_json(raw):
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE
    ).strip()
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char in "[{":
            try:
                result, _ = decoder.raw_decode(cleaned[i:])
                return result
            except json.JSONDecodeError:
                pass
    raise ValueError("No valid JSON found.")

def generate_raw(pages, prompt, max_new_tokens=1800):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{"type": "text", "text": prompt}],
    }
    prompt_text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

def generate_sequence_with_fallback(sequence_id, image_paths):
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    try:
        raw = generate_raw(pages, PROMPT)
        try:
            result = parse_json(raw)
            valid = (
                isinstance(result, dict)
                and result.get("sequence_id") == sequence_id
                and isinstance(result.get("pages"), list)
                and len(result["pages"]) == 3
                and all(
                    isinstance(page, list)
                    and all(
                        isinstance(line, dict)
                        and "speaker" in line and "text" in line
                        for line in page
                    )
                    for page in result["pages"]
                )
            )
            if not valid:
                raise ValueError("Wrong JSON structure.")
            return {"result": result, "used_page_fallback": False, "raw": raw}
        except ValueError:
            single_prompt = (
                "Transcribe dialogue from this single English manga page "
                "in reading order. Preserve punctuation and speaker IDs. "
                "Exclude sound effects, credits, signs, and other "
                "non-dialogue text. Return only a valid JSON array of "
                "objects containing speaker and text."
            )
            page_results = []
            for page in pages:
                parsed = parse_json(generate_raw([page], single_prompt))
                if not isinstance(parsed, list) or not all(
                    isinstance(line, dict)
                    and "speaker" in line and "text" in line
                    for line in parsed
                ):
                    raise ValueError("Page fallback returned invalid JSON.")
                page_results.append(parsed)
            return {
                "result": {"sequence_id": sequence_id, "pages": page_results},
                "used_page_fallback": True,
                "raw": raw,
            }
    finally:
        for page in pages:
            page.close()

print("Generation helpers ready.")
```

```python

# VALIDATION: evaluate the new adapter on the same 16 held-out sequences

import os, json, subprocess, sys, torch

ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
OUTPUT_DIR = "/content/manga_evaluation_r4_401k"
PRED_PATH = OUTPUT_DIR + "/predictions.jsonl"
SCORE_PATH = OUTPUT_DIR + "/scores.json"
REF_PATH = "/content/validation_references.jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

val_ids = split["validation_ids"]
assert len(val_ids) == 16, f"Expected 16 validation sequences, found {len(val_ids)}"
assert os.path.exists(REF_PATH), f"Missing references: {REF_PATH}"
assert "generate_sequence_with_fallback" in globals(), \
    "Generation helper is missing. Run the training cell first."

# Generate predictions; preserve every validation sequence in the output.
with open(PRED_PATH, "w", encoding="utf-8") as out:
    for i, sid in enumerate(val_ids, 1):
        paths = [os.path.join(IMAGE_ROOT, sid, f"{n:02d}.png")
                 for n in (1, 2, 3)]
        try:
            result = generate_sequence_with_fallback(sid, paths)
            prediction = result["result"]
            record = {
                "sequence_id": sid,
                "pages": prediction.get("pages", [[], [], []]),
            }
            print(f"{i:02d}/16  {sid}  fallback={result['used_page_fallback']}")
        except Exception as e:
            # Keep the sequence in the file and report the failure explicitly.
            record = {"sequence_id": sid, "pages": [[], [], []]}
            print(f"{i:02d}/16  {sid}  GENERATION ERROR: {type(e).__name__}: {e}")

        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nPredictions saved:", PRED_PATH)

# Run the official scorer using the common named-argument interface.
cmd = [
    sys.executable, "/content/score.py",
    "--predictions", PRED_PATH,
    "--references", REF_PATH,
    "--output", SCORE_PATH,
]
run = subprocess.run(cmd, capture_output=True, text=True)

print("\nSCORER OUTPUT:")
print(run.stdout[-6000:] if run.stdout else "(no standard output)")
if run.stderr:
    print("SCORER ERRORS:")
    print(run.stderr[-3000:])

if run.returncode == 0 and os.path.exists(SCORE_PATH):
    with open(SCORE_PATH, encoding="utf-8") as f:
        print("\nNEW VALIDATION SCORES:")
        print(json.dumps(json.load(f), indent=2))
else:
    print("\nThe scorer did not finish with those arguments.")
    help_run = subprocess.run(
        [sys.executable, "/content/score.py", "--help"],
        capture_output=True, text=True
    )
    print((help_run.stdout or help_run.stderr)[-5000:])
```

Trying to fix generation and parsing

```python

# ONE-SEQUENCE DIAGNOSTIC
# Inspect the current helpers and capture raw model output before parsing.

import inspect
import json
import os

ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"

# 1. Inspect the exact code responsible for generation and parsing.
for name in [
    "generate_raw",
    "parse_json",
    "generate_sequence_with_fallback",
]:
    fn = globals().get(name)
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    if fn is None:
        print("NOT FOUND")
    else:
        print("Signature:", inspect.signature(fn))
        try:
            print(inspect.getsource(fn))
        except (OSError, TypeError):
            print("Source unavailable; signature shown above.")

# 2. Select one validation sequence.
with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

sid = split["validation_ids"][0]
paths = [
    f"{IMAGE_ROOT}/{sid}/{page:02d}.png"
    for page in (1, 2, 3)
]

assert all(os.path.isfile(p) for p in paths), (
    f"Missing page image(s) for {sid}: {paths}"
)

print(f"\n{'=' * 20} RAW GENERATION TEST {'=' * 20}")
print("Sequence:", sid)
print("Images:", paths)

# 3. Generate once, without invoking the JSON parser or fallback.
try:
    raw = generate_raw(sid, paths)
    print("\nRaw return type:", type(raw).__name__)
    print("Raw output:")
    print(repr(raw))
except Exception as e:
    print("\nRAW GENERATION FAILED")
    print(type(e).__name__ + ":", str(e))
```

```python

# Correctly call generate_raw with loaded images and the prompt.
from PIL import Image

sid = split["validation_ids"][0]
paths = [
    f"{IMAGE_ROOT}/{sid}/{page:02d}.png"
    for page in (1, 2, 3)
]

pages = [Image.open(p).convert("RGB") for p in paths]
try:
    raw = generate_raw(pages, PROMPT, max_new_tokens=1800)
    print("RAW OUTPUT:")
    print(repr(raw))
finally:
    for page in pages:
        page.close()
```

Combined fix

```python

# ============================================================
# COMBINED FIX: PAGES-ONLY TRAINING + REPETITION CONTROL
# + ROBUST JSON HANDLING + ONE-SEQUENCE TEST
#
# Preserves existing adapters and predictions.
# Trains a fresh adapter in a new directory.
# ============================================================

import os
import re
import gc
import json
import inspect
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_vl_utils import process_vision_info

# -------------------- CONFIGURATION --------------------

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
LABEL_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"

# A NEW path: existing adapters are left untouched.
ADAPTER_OUT = "/content/manga_pages_only_r4_401k"
TRAIN_OUT = "/content/manga_pages_only_training"

MIN_PIXELS = 200704
MAX_PIXELS = 401408

assert not os.path.exists(ADAPTER_OUT), (
    f"{ADAPTER_OUT} already exists. Choose a new adapter directory "
    "so an existing result is never overwritten."
)

# Same transcription task, but the model no longer generates sequence_id.
PROMPT = (
    "Transcribe dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound "
    "effects, credits, signs, and other non-dialogue text. "
    "Return only valid JSON with the key pages. The value of "
    "pages must contain exactly three arrays, one for each input "
    "page in order. Each array must contain objects with speaker "
    "and text keys. Do not include sequence_id or any other keys."
)

# -------------------- LOAD DATA --------------------

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_records = [
        json.loads(line) for line in f if line.strip()
    ]

labels_by_id = {
    record["sequence_id"]: record["pages"]
    for record in label_records
}

train_ids = split["train_ids"]
assert len(train_ids) == 64, f"Unexpected training split: {len(train_ids)}"

train_records = []
for sid in train_ids:
    if sid not in labels_by_id:
        raise KeyError(f"Missing labels for {sid}")

    pages = labels_by_id[sid]
    if len(pages) != 3:
        raise ValueError(f"{sid}: expected 3 reference pages")

    paths = [
        f"{IMAGE_ROOT}/{sid}/{i:02d}.png"
        for i in (1, 2, 3)
    ]
    if not all(os.path.isfile(p) for p in paths):
        raise FileNotFoundError(f"Missing image for {sid}: {paths}")

    # Crucially, sequence_id is NOT part of the assistant target.
    target_text = json.dumps(
        {"pages": pages},
        ensure_ascii=False,
    )

    train_records.append({
        "sequence_id": sid,
        "image_paths": paths,
        "target_text": target_text,
    })

print("Training sequences:", len(train_records))
print("Output schema: {'pages': [page1, page2, page3]}")

# -------------------- RELEASE OLD MODEL MEMORY --------------------

# The previous adapter is already saved on disk.
# Release the currently loaded model before loading a fresh base model.
if "trainer" in globals():
    del trainer
if "model" in globals():
    del model

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------- FRESH 4-BIT BASE MODEL --------------------

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    size={
        "shortest_edge": MIN_PIXELS,
        "longest_edge": MAX_PIXELS,
    },
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# Freeze vision encoder; train language-model LoRA adapters only.
for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=(
        r".*language_model.*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    ),
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- TRAINING DATASET + COLLATOR --------------------

class MangaDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def manga_collator(batch):
    # Batch size is intentionally one to fit the T4.
    item = batch[0]
    pages = [
        Image.open(path).convert("RGB")
        for path in item["image_paths"]
    ]

    try:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": page}
                for page in pages
            ] + [{"type": "text", "text": PROMPT}],
        }

        prompt_text = processor.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = processor.apply_chat_template(
            [
                user_message,
                {
                    "role": "assistant",
                    "content": item["target_text"],
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(
            [user_message]
        )

        # Tokenize prompt and full example with identical images.
        prompt_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        full_inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()

        # Mask the user prompt so loss applies to assistant output only.
        prompt_len = int(
            prompt_inputs["attention_mask"][0].sum().item()
        )
        full_len = int(
            full_inputs["attention_mask"][0].sum().item()
        )

        if prompt_len >= full_len:
            raise ValueError(
                f"Empty assistant target for {item['sequence_id']}: "
                f"prompt={prompt_len}, full={full_len}"
            )

        labels[:, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100

        if not torch.any(labels != -100):
            raise ValueError(
                f"No trainable target tokens for {item['sequence_id']}"
            )

        full_inputs["labels"] = labels
        return dict(full_inputs)

    finally:
        for page in pages:
            page.close()


# -------------------- TRAIN --------------------


training_args = TrainingArguments(
    output_dir=TRAIN_OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    logging_steps=4,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MangaDataset(train_records),
    data_collator=manga_collator,
)

print("\nStarting fresh pages-only training...")
train_result = trainer.train()

model.config.use_cache = True
model.save_pretrained(ADAPTER_OUT)
processor.save_pretrained(ADAPTER_OUT)

print("\nNew adapter saved:", ADAPTER_OUT)
print("Training loss:", train_result.training_loss)

# -------------------- INFERENCE + PARSING HELPERS --------------------

def generate_raw(pages, prompt, max_new_tokens=1200):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": page}
            for page in pages
        ] + [{"type": "text", "text": prompt}],
    }

    prompt_text = processor.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info([message])

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_json(raw):
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()

    decoder = json.JSONDecoder()

    for i, char in enumerate(cleaned):
        if char in "[{":
            try:
                result, _ = decoder.raw_decode(cleaned[i:])
                return result
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found in model output.")


def clean_page(page):
    if not isinstance(page, list):
        raise ValueError("A page must be an array.")

    cleaned = []
    previous = None

    for line in page:
        if not isinstance(line, dict):
            raise ValueError("Dialogue entry is not an object.")

        speaker = line.get("speaker")
        text = line.get("text")

        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("Dialogue entry has an invalid speaker.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Dialogue entry has invalid text.")

        speaker = speaker.strip()
        text = text.strip()

        # Drop only identical consecutive entries on this page.
        # Do not remove legitimate repetitions separated by other lines.
        key = (
            re.sub(r"\s+", " ", speaker).casefold(),
            re.sub(r"\s+", " ", text).casefold(),
        )

        if key == previous:
            continue

        cleaned.append({"speaker": speaker, "text": text})
        previous = key

    return cleaned


def validate_pages(result):
    if not isinstance(result, dict):
        return None

    pages = result.get("pages")
    if not isinstance(pages, list) or len(pages) != 3:
        return None

    try:
        return [clean_page(page) for page in pages]
    except ValueError:
        return None


def generate_sequence_with_fallback(sequence_id, image_paths):
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    raw = ""

    try:
        # First try the original three-page task.
        raw = generate_raw(pages, PROMPT, max_new_tokens=1200)
        parsed = parse_json(raw)
        cleaned_pages = validate_pages(parsed)

        if cleaned_pages is not None:
            return {
                "result": {
                    "sequence_id": sequence_id,
                    "pages": cleaned_pages,
                },
                "used_page_fallback": False,
                "raw": raw,
            }

        # If the model misses a page or returns malformed structure,
        # transcribe pages individually using the learned pages schema.
        single_prompt = (
            "Transcribe dialogue from this single English manga page "
            "in reading order. Preserve punctuation and speaker IDs. "
            "Exclude sound effects, credits, signs, and other "
            "non-dialogue text. Return only valid JSON with the key "
            "pages, whose value is an array containing exactly one "
            "page array. Each dialogue object must contain speaker "
            "and text."
        )

        page_results = []

        for i, page in enumerate(pages):
            page_raw = generate_raw(
                [page],
                single_prompt,
                max_new_tokens=500,
            )

            try:
                page_result = parse_json(page_raw)
                one_page = page_result.get("pages") if isinstance(
                    page_result, dict
                ) else None

                if (
                    not isinstance(one_page, list)
                    or len(one_page) != 1
                ):
                    raise ValueError("Expected exactly one page array.")

                page_results.append(clean_page(one_page[0]))

            except Exception as e:
                raise ValueError(
                    f"Fallback failed for page {i + 1}: {e}. "
                    f"Raw page output: {page_raw!r}. "
                    f"Raw combined output: {raw!r}"
                ) from e

        return {
            "result": {
                "sequence_id": sequence_id,
                "pages": page_results,
            },
            "used_page_fallback": True,
            "raw": raw,
        }

    finally:
        for page in pages:
            page.close()


# -------------------- TEST ONE VALIDATION SEQUENCE --------------------

val_ids = split["validation_ids"]
test_id = val_ids[0]
test_paths = [
    f"{IMAGE_ROOT}/{test_id}/{i:02d}.png"
    for i in (1, 2, 3)
]

print("\nTesting one validation sequence:", test_id)

try:
    test_output = generate_sequence_with_fallback(
        test_id,
        test_paths,
    )
    result = test_output["result"]

    assert result["sequence_id"] == test_id
    assert len(result["pages"]) == 3

    print("Valid three-page output: YES")
    print("Used page fallback:", test_output["used_page_fallback"])
    print("Dialogue entries per page:", [
        len(page) for page in result["pages"]
    ])
    print("\nParsed result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\nRaw combined model output:")
    print(repr(test_output["raw"]))

except Exception as e:
    print("\nONE-SEQUENCE TEST FAILED")
    print(type(e).__name__ + ":", str(e))
    raise
```

```python

import gc
import torch

# Release any partial training state and cached GPU blocks.
if "trainer" in globals():
    del trainer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU total: {total_bytes / 1024**3:.2f} GiB")
    print(f"GPU free:  {free_bytes / 1024**3:.2f} GiB")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
else:
    print("CUDA is not available.")
```

```python

import inspect
import transformers
from transformers import TrainingArguments

print("Transformers version:", transformers.__version__)
print("TrainingArguments supports:")
print([
    name for name in inspect.signature(TrainingArguments).parameters
    if any(k in name.lower() for k in ("loss", "chunk", "logit", "batch"))
])
```

Restarted session.

```python

# ============================================================
# CLEAN RESTART: PAGES-ONLY TRAINING + CHUNKED LOSS + TEST
# Run this entire cell once. Do not run another training cell
# concurrently.
# ============================================================

import os, gc, json, re
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_vl_utils import process_vision_info

# -------------------- PATHS AND SETTINGS --------------------

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
LABEL_PATH = "/content/labels (1).jsonl"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"

ADAPTER_OUT = "/content/manga_pages_only_r4_200k"
TRAIN_OUT = "/content/manga_pages_only_training_200k"

assert not os.path.exists(ADAPTER_OUT), (
    f"Refusing to overwrite existing adapter: {ADAPTER_OUT}"
)
assert os.path.isfile(LABEL_PATH) and os.path.isfile(SPLIT_PATH)
assert torch.cuda.is_available(), "GPU runtime is required."

PIXEL_BUDGET = 200704
CHUNK_SIZE = 64

PROMPT = (
    "Transcribe dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound "
    "effects, credits, signs, and other non-dialogue text. "
    "Return only valid JSON with the key pages. The value of "
    "pages must contain exactly three arrays, one for each input "
    "page in order. Each array must contain objects with speaker "
    "and text keys. Do not include sequence_id or any other keys."
)

# -------------------- LOAD TRAINING DATA --------------------

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_records = [
        json.loads(line) for line in f if line.strip()
    ]

labels_by_id = {
    record["sequence_id"]: record["pages"]
    for record in label_records
}

train_records = []
for sid in split["train_ids"]:
    if sid not in labels_by_id:
        raise KeyError(f"Missing labels for {sid}")

    pages_target = labels_by_id[sid]
    if len(pages_target) != 3:
        raise ValueError(f"{sid}: expected three target pages")

    paths = [
        f"{IMAGE_ROOT}/{sid}/{i:02d}.png"
        for i in (1, 2, 3)
    ]
    if not all(os.path.isfile(p) for p in paths):
        raise FileNotFoundError(f"Missing page image for {sid}")

    train_records.append({
        "sequence_id": sid,
        "image_paths": paths,
        "target_text": json.dumps(
            {"pages": pages_target}, ensure_ascii=False
        ),
    })

print("Training sequences:", len(train_records))
print("Target format: pages only")

# -------------------- LOAD MODEL --------------------

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    size={
        "shortest_edge": PIXEL_BUDGET,
        "longest_edge": PIXEL_BUDGET,
    },
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

model = get_peft_model(
    model,
    LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        ),
    ),
)

model.print_trainable_parameters()

# -------------------- DATASET AND COLLATOR --------------------

class MangaDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def manga_collator(batch):
    item = batch[0]
    pages = [
        Image.open(path).convert("RGB")
        for path in item["image_paths"]
    ]

    try:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": page}
                for page in pages
            ] + [{"type": "text", "text": PROMPT}],
        }

        prompt_text = processor.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = processor.apply_chat_template(
            [
                user_message,
                {
                    "role": "assistant",
                    "content": item["target_text"],
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(
            [user_message]
        )

        prompt_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        full_inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = int(
            prompt_inputs["attention_mask"][0].sum().item()
        )
        full_len = int(
            full_inputs["attention_mask"][0].sum().item()
        )

        if prompt_len >= full_len:
            raise ValueError(
                f"Empty target for {item['sequence_id']}"
            )

        labels[:, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100

        if not torch.any(labels != -100):
            raise ValueError(
                f"No target tokens for {item['sequence_id']}"
            )

        full_inputs["labels"] = labels
        return dict(full_inputs)

    finally:
        for page in pages:
            page.close()

# -------------------- CHUNKED CROSS-ENTROPY --------------------
# Computes FP32 loss on small chunks instead of converting the
# entire sequence's vocabulary logits to FP32 at once.

class ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        vocab = logits.shape[-1]
        flat_logits = logits[:, :-1, :].reshape(-1, vocab)
        targets = labels[:, 1:].reshape(-1)

        valid_count = int((targets != -100).sum().item())
        if valid_count == 0:
            raise ValueError("No unmasked target tokens.")

        ctx.save_for_backward(logits, targets)
        ctx.chunk_size = int(chunk_size)
        ctx.valid_count = valid_count

        total = torch.zeros(
            (), device=logits.device, dtype=torch.float32
        )

        for start in range(0, targets.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, targets.numel())
            chunk_logits = flat_logits[start:end].float()
            chunk_targets = targets[start:end]

            total += F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )

        return total / valid_count

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        vocab = logits.shape[-1]
        flat_logits = logits[:, :-1, :].reshape(-1, vocab)

        grad_logits = torch.zeros_like(logits)
        flat_grad = grad_logits[:, :-1, :].reshape(-1, vocab)
        scale = grad_output / ctx.valid_count

        for start in range(0, targets.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, targets.numel())
            y = targets[start:end]
            valid = y != -100

            probs = torch.softmax(
                flat_logits[start:end].float(), dim=-1
            )
            probs[~valid] = 0

            if valid.any():
                rows = torch.arange(
                    end - start, device=y.device
                )[valid]
                probs[rows, y[valid]] -= 1

            flat_grad[start:end] = (
                probs * scale
            ).to(logits.dtype)

        return grad_logits, None, None


class MemoryEfficientTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False,
        num_items_in_batch=None
    ):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = ChunkedCrossEntropy.apply(
            outputs.logits, labels, CHUNK_SIZE
        )

        return (loss, outputs) if return_outputs else loss

# -------------------- TRAIN --------------------

training_args = TrainingArguments(
    output_dir=TRAIN_OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    logging_steps=4,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = MemoryEfficientTrainer(
    model=model,
    args=training_args,
    train_dataset=MangaDataset(train_records),
    data_collator=manga_collator,
)

print("\nStarting memory-efficient training...")
train_result = trainer.train()

model.config.use_cache = True
model.save_pretrained(ADAPTER_OUT)
processor.save_pretrained(ADAPTER_OUT)

print("\nTraining finished.")
print("Training loss:", train_result.training_loss)
print("Saved adapter:", ADAPTER_OUT)

# -------------------- INFERENCE HELPERS --------------------

def generate_raw(pages, prompt, max_new_tokens=1200):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{"type": "text", "text": prompt}],
    }

    prompt_text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([message])

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_json(raw):
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE
    ).strip()

    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char in "[{":
            try:
                result, _ = decoder.raw_decode(cleaned[i:])
                return result
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found.")


def clean_page(page):
    if not isinstance(page, list):
        raise ValueError("Page must be an array.")

    cleaned = []
    previous = None

    for line in page:
        if not isinstance(line, dict):
            raise ValueError("Invalid dialogue object.")

        speaker, text = line.get("speaker"), line.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("Invalid speaker.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text.")

        speaker, text = speaker.strip(), text.strip()
        key = (
            re.sub(r"\s+", " ", speaker).casefold(),
            re.sub(r"\s+", " ", text).casefold(),
        )

        if key != previous:
            cleaned.append({"speaker": speaker, "text": text})
        previous = key

    return cleaned


def generate_sequence_with_fallback(sequence_id, image_paths):
    pages = [Image.open(p).convert("RGB") for p in image_paths]

    try:
        raw = generate_raw(pages, PROMPT)
        try:
            parsed = parse_json(raw)
            predicted_pages = (
                parsed.get("pages")
                if isinstance(parsed, dict) else None
            )

            if isinstance(predicted_pages, list) and len(predicted_pages) == 3:
                cleaned = [clean_page(p) for p in predicted_pages]
                return {
                    "result": {
                        "sequence_id": sequence_id,
                        "pages": cleaned,
                    },
                    "used_page_fallback": False,
                    "raw": raw,
                }
        except (ValueError, TypeError):
            pass

        single_prompt = (
            "Transcribe dialogue from this single English manga page "
            "in reading order. Preserve punctuation and speaker IDs. "
            "Exclude sound effects, credits, signs, and other "
            "non-dialogue text. Return only valid JSON with key pages, "
            "containing exactly one page array. Each dialogue object "
            "must contain speaker and text."
        )

        page_results = []
        for index, page in enumerate(pages):
            page_raw = generate_raw(
                [page], single_prompt, max_new_tokens=500
            )
            page_result = parse_json(page_raw)
            one_page = (
                page_result.get("pages")
                if isinstance(page_result, dict) else None
            )

            if not isinstance(one_page, list) or len(one_page) != 1:
                raise ValueError(
                    f"Fallback failed on page {index + 1}: "
                    f"{page_raw!r}"
                )

            page_results.append(clean_page(one_page[0]))

        return {
            "result": {
                "sequence_id": sequence_id,
                "pages": page_results,
            },
            "used_page_fallback": True,
            "raw": raw,
        }

    finally:
        for page in pages:
            page.close()

# -------------------- ONE-SEQUENCE TEST --------------------

test_id = split["validation_ids"][0]
test_paths = [
    f"{IMAGE_ROOT}/{test_id}/{i:02d}.png"
    for i in (1, 2, 3)
]

print("\nTesting validation sequence:", test_id)
test_output = generate_sequence_with_fallback(
    test_id, test_paths
)
result = test_output["result"]

assert result["sequence_id"] == test_id
assert len(result["pages"]) == 3

print("Three-page structure: PASS")
print("Fallback used:", test_output["used_page_fallback"])
print("Dialogue entries per page:", [
    len(page) for page in result["pages"]
])
print(json.dumps(result, ensure_ascii=False, indent=2))
print("Raw combined output:", repr(test_output["raw"]))
```

restarting again

```python

import os
import torch

paths = {
    "Original adapter": "/content/manga_qlora_adapter",
    "Recovery adapter": "/content/manga_qlora_recovery_63",
    "Training dataset": "/content/manga_dataset",
    "Validation references": "/content/validation_references.jsonl",
    "Scoring script": "/content/score.py",
    "Baseline predictions": "/content/baseline_predictions.jsonl",
}

print("=== Saved project files ===")
for name, path in paths.items():
    print(f"{name}: {'FOUND' if os.path.exists(path) else 'MISSING'}")

print("\n=== GPU ===")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {total / 1024**3:.2f} GiB")
    print(f"Free memory: {free / 1024**3:.2f} GiB")
else:
    print("CUDA unavailable")
```

```python

# ============================================================
# INFERENCE-ONLY RECOVERY: LOAD + GENERATE + PARSE + TEST
# No training. No existing files overwritten.
# ============================================================

import os, json, re, gc
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "/content/manga_qlora_adapter"
ROOT = "/content/manga_dataset"
IMAGE_ROOT = ROOT + "/images-20261010T124807Z-1-001/images"
SPLIT_PATH = ROOT + "/train_val_split_seed42.json"

assert os.path.isdir(ADAPTER_PATH), "Original adapter not found."
assert torch.cuda.is_available(), "GPU unavailable."

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

# Keep the original three-page task and output schema.
PROMPT = (
    "Transcribe dialogue from these three consecutive English "
    "manga pages in reading order. Preserve page boundaries, "
    "punctuation, and consistent speaker IDs. Exclude sound "
    "effects, credits, signs, and other non-dialogue text. "
    "Return only valid JSON with keys sequence_id and pages. "
    "Each page must be an array of objects containing speaker "
    "and text."
)

print("Loading processor and 4-bit base model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Loaded adapter:", ADAPTER_PATH)

def generate_raw(pages, prompt, max_new_tokens=1200):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": page} for page in pages
        ] + [{"type": "text", "text": prompt}],
    }
    prompt_text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([message])

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_json(raw):
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE
    ).strip()
    decoder = json.JSONDecoder()

    for i, char in enumerate(cleaned):
        if char in "[{":
            try:
                result, _ = decoder.raw_decode(cleaned[i:])
                return result
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found.")


def clean_page(page):
    if not isinstance(page, list):
        raise ValueError("Page is not an array.")

    cleaned, previous = [], None
    for line in page:
        if not isinstance(line, dict):
            raise ValueError("Dialogue entry is not an object.")

        speaker, text = line.get("speaker"), line.get("text")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("Missing or invalid speaker.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Missing or invalid text.")

        speaker, text = speaker.strip(), text.strip()
        key = (
            re.sub(r"\s+", " ", speaker).casefold(),
            re.sub(r"\s+", " ", text).casefold(),
        )
        # Remove only an identical adjacent speaker/text pair.
        if key != previous:
            cleaned.append({"speaker": speaker, "text": text})
        previous = key

    return cleaned


def generate_sequence_with_fallback(sequence_id, image_paths):
    pages = [Image.open(p).convert("RGB") for p in image_paths]

    try:
        raw = generate_raw(pages, PROMPT)
        try:
            parsed = parse_json(raw)
            predicted_pages = (
                parsed.get("pages") if isinstance(parsed, dict) else None
            )

            if isinstance(predicted_pages, list) and len(predicted_pages) == 3:
                cleaned = [clean_page(p) for p in predicted_pages]
                return {
                    "result": {
                        "sequence_id": sequence_id,
                        "pages": cleaned,
                    },
                    "used_page_fallback": False,
                    "raw": raw,
                }
        except (ValueError, TypeError):
            pass

        # Fallback uses the same three-page schema with one image at a time.
        single_prompt = (
            "Transcribe dialogue from this single English manga page "
            "in reading order. Preserve punctuation and speaker IDs. "
            "Exclude sound effects, credits, signs, and other "
            "non-dialogue text. Return only valid JSON with key pages, "
            "containing exactly one page array. Each dialogue object "
            "must contain speaker and text."
        )

        page_results = []
        for index, page in enumerate(pages):
            page_raw = generate_raw(
                [page], single_prompt, max_new_tokens=500
            )
            page_result = parse_json(page_raw)
            one_page = (
                page_result.get("pages")
                if isinstance(page_result, dict) else None
            )

            if not isinstance(one_page, list) or len(one_page) != 1:
                raise ValueError(
                    f"Fallback failed on page {index + 1}: "
                    f"{page_raw!r}"
                )

            page_results.append(clean_page(one_page[0]))

        return {
            "result": {
                "sequence_id": sequence_id,
                "pages": page_results,
            },
            "used_page_fallback": True,
            "raw": raw,
        }

    finally:
        for page in pages:
            page.close()


# Test one validation sequence only.
test_id = split["validation_ids"][0]
test_paths = [
    f"{IMAGE_ROOT}/{test_id}/{i:02d}.png"
    for i in (1, 2, 3)
]
assert all(os.path.isfile(p) for p in test_paths)

print("\nTesting sequence:", test_id)
test_output = generate_sequence_with_fallback(test_id, test_paths)
result = test_output["result"]

assert result["sequence_id"] == test_id
assert len(result["pages"]) == 3

print("Three-page structure: PASS")
print("Fallback used:", test_output["used_page_fallback"])
print("Dialogue entries per page:", [
    len(page) for page in result["pages"]
])
print("\nParsed result:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\nRaw combined output:")
print(repr(test_output["raw"]))
```

Let's validate over all 16 sequences.

```python

# STEP 4: Validate the fine-tuned adapter on ORIGINAL-RESOLUTION images
# Uses the already-loaded model and existing inference/parsing helpers.
# No training or model reloading.
from pathlib import Path
IMAGE_ROOT = Path(IMAGE_ROOT)

import json
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("/content/manga_evaluation_original_adapter_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUT_DIR / "predictions.jsonl"
FAILURES_PATH = OUT_DIR / "failures.json"
SCORES_PATH = OUT_DIR / "scores.json"

validation_ids = split["validation_ids"]
results = []
failures = []
fallback_count = 0

print(f"Running validation on {len(validation_ids)} sequences")
print("Image resolution: original (not resized)")
print("Adapter: fine-tuned adapter enabled")
print("-" * 60)

for index, sequence_id in enumerate(validation_ids, start=1):
    image_paths = [
        IMAGE_ROOT / sequence_id / f"{page_num:02d}.png"
        for page_num in (1, 2, 3)
    ]

    try:
        output = generate_sequence_with_fallback(sequence_id, image_paths)
        record = output["result"]

        # Validate the output structure before accepting it.
        assert record["sequence_id"] == sequence_id
        assert len(record["pages"]) == 3
        assert all(isinstance(page, list) for page in record["pages"])

        results.append(record)
        fallback_count += int(output.get("used_page_fallback", False))

        counts = [len(page) for page in record["pages"]]
        print(
            f"[{index:02d}/{len(validation_ids)}] {sequence_id}: "
            f"SUCCESS | dialogue counts={counts} | "
            f"page fallback={output.get('used_page_fallback', False)}"
        )

    except Exception as e:
        failures.append({
            "sequence_id": sequence_id,
            "error": str(e)
        })
        print(f"[{index:02d}/{len(validation_ids)}] {sequence_id}: FAILED")

# Save every successful prediction and all failure details.
with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open(FAILURES_PATH, "w", encoding="utf-8") as f:
    json.dump(failures, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"Successful sequences: {len(results)}/{len(validation_ids)}")
print(f"Failed sequences: {len(failures)}")
print(f"Sequences using page fallback: {fallback_count}")
print(f"Predictions saved to: {PREDICTIONS_PATH}")
print(f"Failure log saved to: {FAILURES_PATH}")

if failures:
    print(
        "\nSCORING SKIPPED: not all validation sequences succeeded. "
        "Fix the inference failure before interpreting model quality."
    )
    print("Failed sequence IDs:", [x["sequence_id"] for x in failures])

else:
    print("\nAll sequences succeeded. Running the official scorer...")

    completed = subprocess.run(
        [
            sys.executable,
            "/content/score.py",
            "--predictions", str(PREDICTIONS_PATH),
            "--references", "/content/validation_references.jsonl",
            "--output", str(SCORES_PATH),
        ],
        capture_output=True,
        text=True,
    )

    print(completed.stdout)
    if completed.stderr:
        print("Scorer messages:\n", completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Scorer failed with exit code {completed.returncode}. "
            "See the messages above."
        )

    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        current = json.load(f)

    print("\nCURRENT MODEL SCORES")
    print(json.dumps(current, indent=2))

    # Historical results, kept separate from this run.
    historical = {
        "Pretrained, original resolution": {
            "text_order_score": 0.4648732167,
            "balanced_joint_f1": 0.2259968582,
        },
        "Pretrained, resized to 512": {
            "text_order_score": 0.072282,
            "balanced_joint_f1": 0.018794,
        },
        "Fine-tuned, resized to 512": {
            "text_order_score": 0.228762,
            "balanced_joint_f1": 0.103900,
        },
    }

    # Handle either direct metric keys or a nested "metrics" dictionary.
    current_metrics = current.get("metrics", current)
    historical["Fine-tuned, original resolution (CURRENT)"] = {
        metric: current_metrics.get(metric)
        for metric in ("text_order_score", "balanced_joint_f1")
    }

    print("\nCOMPARISON: HISTORICAL RESULTS + CURRENT RUN")
    print(f"{'Experiment':<43} {'Text order':>12} {'Balanced F1':>12}")
    print("-" * 70)

    for name, metrics in historical.items():
        text_score = metrics["text_order_score"]
        f1_score = metrics["balanced_joint_f1"]

        text_display = (
            f"{text_score:.4f}" if text_score is not None else "N/A"
        )
        f1_display = (
            f"{f1_score:.4f}" if f1_score is not None else "N/A"
        )
        print(f"{name:<43} {text_display:>12} {f1_display:>12}")

    print("\nValidation complete. No model weights were changed.")
```

```python

import json
import subprocess
import sys
from pathlib import Path

out_dir = Path("/content/manga_evaluation_original_adapter_fixed")
pred_path = out_dir / "predictions.jsonl"
fail_path = out_dir / "failures.json"
refs_path = Path("/content/validation_references.jsonl")

full_preds_path = out_dir / "predictions_all16_official_evaluation.jsonl"
scores_path = out_dir / "scores_all16_official_evaluation.json"

# Load the predictions from the validation run we just completed.
with open(pred_path, encoding="utf-8") as f:
    predictions = {
        row["sequence_id"]: row
        for row in (json.loads(line) for line in f if line.strip())
    }

with open(fail_path, encoding="utf-8") as f:
    failures = json.load(f)

failed_ids = {item["sequence_id"] for item in failures}

with open(refs_path, encoding="utf-8") as f:
    references = [json.loads(line) for line in f if line.strip()]

reference_ids = [r["sequence_id"] for r in references]

# Ensure the run and reference split match the expected 16 sequences.
assert len(reference_ids) == 16, (
    f"Expected 16 validation references; found {len(reference_ids)}."
)
assert len(set(reference_ids)) == 16, "Duplicate reference sequence IDs found."
assert set(predictions).isdisjoint(failed_ids), (
    "A sequence is listed as both successful and failed."
)
assert set(predictions) | failed_ids == set(reference_ids), (
    "Successful and failed IDs do not exactly cover the 16 reference sequences."
)

# Assemble a complete prediction file. Failed sequences get empty pages.
with open(full_preds_path, "w", encoding="utf-8") as f:
    for sequence_id in reference_ids:
        if sequence_id in predictions:
            record = predictions[sequence_id]
        else:
            record = {
                "sequence_id": sequence_id,
                "pages": [[], [], []]
            }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Evaluating the exact recent validation run")
print(f"Successful predictions: {len(predictions)}/16")
print(f"Failed sequences counted as empty predictions: {len(failed_ids)}")
print("Running official scorer on all 16 references...\n")

result = subprocess.run(
    [
        sys.executable,
        "/content/score.py",
        "--predictions", str(full_preds_path),
        "--references", str(refs_path),
        "--output", str(scores_path),
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.stderr:
    print("Scorer output:\n", result.stderr)

if result.returncode != 0:
    raise RuntimeError(
        "Official scoring failed. Review the scorer output above."
    )

with open(scores_path, encoding="utf-8") as f:
    scores = json.load(f)

print("\n=== OFFICIAL METRICS: ALL 16 VALIDATION SEQUENCES ===")
print(json.dumps(scores, indent=2))
print(f"\nSaved predictions: {full_preds_path}")
print(f"Saved metrics: {scores_path}")
```

```python

import json
import zipfile
from pathlib import Path

TEST_ZIP = Path("/content/test-20261010T125233Z-1-001.zip")
TEST_DIR = Path("/content/manga_test_data")

assert TEST_ZIP.exists(), f"Test archive not found: {TEST_ZIP}"

TEST_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(TEST_ZIP, "r") as z:
    z.extractall(TEST_DIR)

print("Test archive extracted to:", TEST_DIR)

# Locate the manifest and submission template.
manifests = list(TEST_DIR.rglob("sequences.json"))
templates = list(TEST_DIR.rglob("sample_submission.jsonl"))

print("\nSequence manifests:", [str(p) for p in manifests])
print("Submission templates:", [str(p) for p in templates])

assert manifests, "Could not find sequences.json in the test archive."

manifest_path = manifests[0]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("\nManifest structure:", type(manifest).__name__)

if isinstance(manifest, dict):
    print("Top-level keys:", list(manifest.keys()))
    for key, value in manifest.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} entries")
            if value:
                print("First entry:", value[0])
        else:
            print(f"{key}: {str(value)[:300]}")
elif isinstance(manifest, list):
    print("Sequence entries:", len(manifest))
    if manifest:
        print("First entry:", manifest[0])

print("\nImage files found:", sum(
    1 for p in TEST_DIR.rglob("*")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
))
```
