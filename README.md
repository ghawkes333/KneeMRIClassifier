# Knee MRI Classifier — MedGemma Fine-Tuning for ACL Tear Detection

**Course:** COMP 5630 / 6630 / 6[66]30 — Machine Learning, Spring 2026
**Institution:** Auburn University
**Team:** Gabriella Hawkes, Elizabeth Casey, Maddie Larkin

---

## 1. Project Overview

This project investigates whether a medically pre-trained vision–language
model, **MedGemma-4B-IT**, can be adapted via parameter-efficient fine-tuning
(LoRA) to perform binary classification of anterior cruciate ligament (ACL)
tears from knee MRI slices. We compare three conditions:

1. **Zero-shot baseline** — MedGemma-4B-IT prompted directly with no training.
2. **LoRA fine-tuned** — MedGemma-4B-IT adapted on MRNet training data.
3. **Per-viewpoint analysis** — the same pipeline run independently on the
   `sagittal`, `coronal`, and `axial` MRI views.

The goal is to measure how much downstream medical task performance a general
medical VLM gains from a small amount of in-domain supervised fine-tuning,
and whether the anatomical view of the MRI meaningfully changes that answer.

## 2. Dataset

- **Source:** MRNet (Stanford ML Group), mirrored on Hugging Face as
  `AUMLProject/mrnet-acl`.
- **Task:** Binary classification — `1 = ACL tear`, `0 = no tear`.
- **Splits:**
  - Train: official MRNet training shards (29 parquet files, loaded fully or
    partially via the `n_files` argument).
  - Test: official MRNet validation split (120 exams), used so our numbers
    are directly comparable to published MRNet benchmarks.
- **Inputs:** Each exam contains three 3-D MRI stacks (`sagittal`, `coronal`,
  `axial`). The notebook extracts the center slice of the chosen view and
  passes it as a PIL image to MedGemma.

### Data access note

The MRNet parquet shards declare each view as a fixed `(32, 224, 224)` tensor,
but actual exams have a variable number of slices. We bypass the Hugging Face
Arrow decoder by reading the raw PyArrow table directly (`get_sample`), which
preserves the true slice count per exam.

## 3. Repository Structure

```
KneeMRIClassifier/
├── README.md                         # this file
├── explore_models.ipynb              # main notebook: baseline + fine-tune + eval
├── baseline_results_axial.json       # zero-shot MedGemma on axial view
├── baseline_results_sagittal.json    # zero-shot MedGemma on sagittal view
├── finetuned_results_axial.json      # LoRA-fine-tuned MedGemma on axial view
└── comparison_axial.json             # side-by-side zero-shot vs. fine-tuned
```

## 4. How to Run

The notebook is designed for **Google Colab with a GPU runtime** (tested on
A100 and L4). All memory-heavy steps use 4-bit quantization via
`bitsandbytes`, so a single consumer GPU with ~16 GB VRAM is sufficient for
baseline evaluation; fine-tuning benefits from more.

### Prerequisites

1. Accept the MedGemma license at
   <https://huggingface.co/google/medgemma-4b-it>.
2. Create a Hugging Face **Read** access token at
   <https://huggingface.co/settings/tokens>.
3. (Optional) Mount Google Drive if you want checkpoint backup across Colab
   sessions.

### Dependencies

Installed by the first cell of the notebook:

```bash
pip install -q bitsandbytes accelerate peft trl
```

Additional packages used (already available in Colab): `torch`,
`transformers`, `datasets`, `huggingface_hub`, `pyarrow`, `numpy`, `pandas`,
`scikit-learn`, `Pillow`, `tqdm`.

### Running end-to-end

1. Open `explore_models.ipynb` in Colab.
2. Run cell 5 (`login()`) and paste your HF token.
3. In the **Choose viewpoint** cell, set
   ```python
   VIEWPOINT = "axial"   # or "sagittal" or "coronal"
   ```
4. Run all cells top-to-bottom. The notebook will:
   - download and inspect the dataset,
   - run and save the zero-shot baseline → `baseline_results_<VIEWPOINT>.json`,
   - LoRA-fine-tune MedGemma → adapters saved under
     `medgemma-mrnet-<lr>-<VIEWPOINT>/final/`,
   - evaluate the fine-tuned model → `finetuned_results_<VIEWPOINT>.json`,
   - write a head-to-head table → `comparison_<VIEWPOINT>.json`.

Output filenames are viewpoint-scoped, so you can re-run the notebook with a
different `VIEWPOINT` without overwriting previous results.

### Fast-dev vs. full runs

- `download_train_test_datasets(n_files=3)` loads ~117 training examples for
  quick iteration.
- `download_train_test_datasets(n_files=None)` loads all 29 shards.
- `evaluate(..., n_samples=30)` evaluates on a random 30-sample subset;
  `n_samples=None` uses the full 120-exam validation split.

## 5. Methodology

### 5.1 Zero-shot baseline

MedGemma-4B-IT is loaded in 4-bit precision through the
`image-text-to-text` pipeline. For each test exam, we extract the center
slice of the chosen view, rescale to `[0, 255]` as a grayscale PIL image,
and prompt:

> *"You are looking at a knee MRI slice. Does this image show evidence of
> an ACL (Anterior Cruciate Ligament) tear? Answer with a single word:
> 'Yes' or 'No'."*

The model's free-text reply is parsed by a robust `parse_yes_no` helper
that falls back on substring matching. Crucially, the prompt contains
**no label leakage** — the ground-truth answer is never inserted into the
context at inference time.

### 5.2 LoRA fine-tuning

We apply PEFT LoRA adapters on top of the 4-bit quantized base:

| Hyperparameter | Value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| Target modules | `all-linear` |
| `modules_to_save` | `lm_head`, `embed_tokens` |
| Task type | `CAUSAL_LM` |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 1 |
| Per-device batch | 2 |
| Grad accumulation | 8 (effective batch = 16) |
| Precision | bf16 on A100/H100, fp16 otherwise |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | enabled |

Training data is reformatted into the chat template expected by
`SFTTrainer` via `reformat_sample`, which attaches the ground-truth
`"Yes" / "No"` as the assistant turn. The `reformat_sample` helper is
used **only** for training — inference always goes through `predict_acl`,
which omits the assistant turn.

### 5.3 Evaluation

The same `evaluate()` function is applied to both the zero-shot pipeline
and the fine-tuned pipeline, so the numbers are directly comparable. We
report accuracy, precision, recall, F1, and the 2×2 confusion matrix, plus
the count of unparseable responses.

## 6. Results

Results on the axial view (full 120-exam test set):

| Metric        | Zero-shot | Fine-tuned | Δ    |
|---------------|----------:|-----------:|-----:|
| Accuracy      | 0.550     | 0.550      | 0.000 |
| Precision     | 0.000     | 0.000      | 0.000 |
| Recall        | 0.000     | 0.000      | 0.000 |
| F1            | 0.000     | 0.000      | 0.000 |
| Confusion     | `[[66,0],[54,0]]` | `[[66,0],[54,0]]` | — |

Zero-shot on the sagittal view (30-sample subset): accuracy 0.533,
precision/recall/F1 all 0.000, confusion `[[16, 0], [14, 0]]`.

**Interpretation.** Both the zero-shot and the LoRA-fine-tuned models
collapse to always predicting "No tear." The "accuracy" of 0.55 is purely
the majority-class rate on the axial validation split. A single-epoch
LoRA fine-tune on a small subset of MRNet training data (loaded with
`n_files=3`, ~117 examples) was not enough to overcome the language
model's prior toward the negative answer. We discuss this in the
accompanying report as the primary finding and suggest several
mitigations — class-balanced sampling, multi-slice input, more epochs
on the full training set, and an explicit classification head rather
than free-text generation.

## 7. Reproducibility

- All random splits use `seed=42`.
- Sampled evaluation uses `np.random.RandomState(seed).choice` for
  deterministic sample selection.
- Greedy decoding (`do_sample=False`) ensures identical outputs across
  re-runs on the same hardware.
- Results are serialized to JSON under per-viewpoint filenames.

## 8. Known Limitations

1. **Single slice per exam.** We pass only the center slice; MRNet exams
   have up to 40+ slices per view, and the actual tear may not be at the
   center.
2. **Small training subset.** Fine-tuning in the provided runs uses only
   the first three parquet shards for iteration speed.
3. **Binary free-text parsing.** The `parse_yes_no` helper returns `None`
   when the model hedges; unparseable outputs count as incorrect.
4. **Single view at a time.** Each run uses one of sagittal, coronal,
   axial. An ensemble over all three views is future work.

## 9. GenAI Disclosure

In accordance with course policy, we disclose the following uses of
generative AI tools (Claude, Gemini) during this project:

- debugging the parquet/Arrow data pipeline (fixed-shape decoder
  workaround);
- diagnosing out-of-memory errors during fine-tuning;
- boilerplate for model loading, 4-bit quantization, and the SFT training
  loop (adapted from Google's official MedGemma fine-tuning tutorial:
  <https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb>);
- scaffolding for the `evaluate()` function and the before/after
  comparison printer;
- drafting portions of the final report and this README.

All experimental decisions, dataset choices, metric interpretations, and
final reported numbers are the authors' own.

## 10. Team Contributions

*(Please fill in — graders award up to 10 points for evidence of
collaboration and individual contributions.)*

- **Gabriella Hawkes:** …
- **Elizabeth Casey:** …
- **Maddie Larkin:** …

## 11. References

- Bien, N. et al. (2018). *Deep-learning-assisted diagnosis for knee
  magnetic resonance imaging: Development and retrospective validation of
  MRNet.* PLOS Medicine.
- Google Health. *MedGemma-4B-IT.*
  <https://huggingface.co/google/medgemma-4b-it>
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models.* arXiv:2106.09685.
- Hugging Face `trl` library — `SFTTrainer` documentation.
