# Knee MRI Classifier — MedGemma Fine-Tuning for ACL Tear Detection

**Course:** COMP 5630 / 6630 / 6[66]30 — Machine Learning, Spring 2026
**Institution:** Auburn University
**Team:** Gabriella Hawkes, Elizabeth Casey, Maddie Larkin

---

## 1. Project Overview

This project investigates whether a medically pretrained vision–language
model, **MedGemma-4B-IT**, can be adapted via parameter-efficient fine-tuning
(LoRA) to perform binary classification of anterior cruciate ligament (ACL)
tears from knee MRI slices. We compare a zero-shot baseline against the same
model after LoRA fine-tuning, on the MRNet validation set.

## 2. Headline Result

On the MRNet axial validation set (n = 120):

| Metric    | Zero-shot | Fine-tuned | Δ      |
|-----------|----------:|-----------:|-------:|
| Accuracy  | 0.550     | 0.650      | +0.100 |
| Precision | 0.000     | 0.643      | +0.643 |
| Recall    | 0.000     | 0.500      | +0.500 |
| F1        | 0.000     | 0.562      | +0.562 |

Confusion matrices:

- Zero-shot: `[[66, 0], [54, 0]]` — model predicted "No" on every exam.
- Fine-tuned: `[[51, 15], [27, 27]]` — 42 positive predictions, 27 correct.

LoRA fine-tuning converted a degenerate constant predictor into a functional
binary classifier. See `comparison_axial.json` for the full metric blob and
the report for analysis.

## 3. Dataset

- **Source:** MRNet (Stanford ML Group), mirrored on Hugging Face as
  `AUMLProject/mrnet-acl`.
- **Task:** Binary classification — `1 = ACL tear`, `0 = no tear`.
- **Splits:**
  - Train: full official MRNet training shards (29 parquet files).
  - Test: official MRNet validation split (120 exams), used so our numbers
    are directly comparable to published MRNet benchmarks.
- **Inputs:** Each exam contains three 3-D MRI stacks (`sagittal`, `coronal`,
  `axial`). We extract the center slice of the chosen view and pass it as a
  PIL image to MedGemma.

### Data access note

MRNet parquet shards declare each view as a fixed `(32, 224, 224)` tensor,
but actual exams have a variable number of slices. We bypass the Hugging
Face Arrow decoder by reading the raw PyArrow table directly
(`get_sample`), which preserves the true slice count per exam.

## 4. Repository Structure

```
KneeMRIClassifier/
├── README.md                         # this file
├── explore_models.ipynb              # main notebook: baseline + fine-tune + eval
├── data_preprocessing.ipynb          # dataset prep for MRNet, KneeMRI, fastMRI, CAI2R
├── baseline_results_axial.json       # zero-shot MedGemma on axial view (n=120)
├── baseline_results_sagittal.json    # zero-shot MedGemma on sagittal view (n=30)
├── finetuned_results_axial.json      # LoRA-fine-tuned MedGemma on axial (n=120)
└── comparison_axial.json             # side-by-side zero-shot vs. fine-tuned
```

## 5. How to Run

The notebook is designed for **Google Colab with a GPU runtime** (tested on
A100 and L4). All memory-heavy steps use 4-bit quantization via
`bitsandbytes`. ~16 GB VRAM is sufficient for baseline evaluation;
fine-tuning runs comfortably on an L4 in 30-60 min or on an A100 in ~15 min.

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

Additional packages (already in Colab): `torch`, `transformers`, `datasets`,
`huggingface_hub`, `pyarrow`, `numpy`, `pandas`, `scikit-learn`, `Pillow`,
`tqdm`.

### Running end-to-end

1. Open `explore_models.ipynb` in Colab.
2. Run the `login()` cell and paste your HF token.
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

Output filenames are viewpoint-scoped, so re-running with a different
`VIEWPOINT` will not overwrite previous results.

## 6. Methodology

### 6.1 Zero-shot baseline

MedGemma-4B-IT is loaded in 4-bit precision through the
`image-text-to-text` pipeline. For each test exam we extract the center
slice of the chosen view, rescale to `[0, 255]` as a grayscale PIL image,
and prompt:

> *"You are looking at a knee MRI slice. Does this image show evidence of
> an ACL (Anterior Cruciate Ligament) tear? Answer with a single word:
> 'Yes' or 'No'."*

The model's free-text reply is parsed by a robust `parse_yes_no` helper
that falls back on substring matching. The prompt contains no label
leakage.

### 6.2 LoRA fine-tuning

| Hyperparameter | Value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4, cosine schedule, 10% warmup |
| Epochs | 3 |
| Per-device batch | 2 |
| Grad accumulation | 8 (effective batch = 16) |
| Precision | bf16 on A100/H100, fp16 otherwise |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | enabled |
| Class balancing | minority-class oversampling on train; eval untouched |

### 6.3 Evaluation

The same `evaluate()` function is applied to the zero-shot pipeline and the
fine-tuned pipeline so the numbers are directly comparable. We report
accuracy, precision, recall, F1, and the 2x2 confusion matrix, plus the
count of unparseable responses and a first-word histogram of model outputs
(a diagnostic that makes mode collapse immediately visible).

## 7. Implementation Notes

Three implementation details proved necessary to obtain non-degenerate
fine-tuned behavior. We document them here because each one, in isolation,
was sufficient to make the fine-tuned model look identical to the zero-shot
baseline.

1. **Loss masking.** The collator must mask every token up to and including
   the end of the user prompt, so only the assistant-turn answer tokens
   contribute to the loss. Computing loss over the whole sequence (prompt
   + answer) drowns out the single-token answer signal and one epoch of
   training has no measurable effect.
2. **Class balancing.** MRNet's training set is unbalanced, and MedGemma's
   language prior strongly favors "No" on yes/no medical questions.
   Oversampling the minority class on the training side (eval untouched)
   was necessary for the model to escape the prior in a small number of
   epochs.
3. **Adapter merging at inference.** `trainer.model` wrapped in
   `pipeline(...)` can, in some PEFT/TRL versions, forward through the
   base weights and ignore the LoRA adapters. We call
   `trainer.model.merge_and_unload()` before constructing the inference
   pipeline to bake the adapter deltas into the base weights.

## 8. Reproducibility

- All random splits use `seed=42`.
- Sampled evaluation uses `np.random.RandomState(seed).choice` for
  deterministic sample selection.
- Greedy decoding (`do_sample=False`) is used at inference.
- Per-run results are serialized to JSON under per-viewpoint filenames.

## 9. Known Limitations

1. **Single slice per exam.** We pass only the center slice; real MRNet
   exams have many slices and the actual tear may not be at the center.
2. **Single view evaluated end-to-end.** Fine-tuning numbers are reported
   only for the axial view. Sagittal and coronal would benefit from the
   same treatment.
3. **Single LoRA configuration.** Rank 16, attention-only targets. Higher
   rank or unfreezing the vision encoder is future work.
4. **Cross-dataset generalization not evaluated.** Preprocessing pipelines
   exist for KneeMRI, fastMRI, and CAI2R but model evaluation is on MRNet
   only.

## 10. GenAI Disclosure

We used Claude (Anthropic) and Gemini (Google) during development for:
debugging the data pipeline (parquet loading, fixed-shape decoder
workaround, PyArrow -> HuggingFace Dataset conversion), diagnosing
out-of-memory errors, model-loading boilerplate, the SFT training loop
(adapted from Google's official MedGemma fine-tuning tutorial:
<https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb>),
debugging the loss-masking and adapter-attachment failure modes described
in section 7, scaffolding for the `evaluate()` function, drafting
preprocessing scripts for the auxiliary datasets, and drafting portions
of this README and the final report. All generated code was reviewed,
modified, and tested before inclusion. All experimental decisions,
dataset choices, metric interpretations, and final reported numbers are
the authors' own.

## 11. Team Contributions

- **Gabriella Hawkes** — Model development lead. Selected MedGemma over
  CNN-from-scratch and Pillar-0 alternatives, implemented model loading
  and 4-bit quantization, ran fine-tuning experiments, debugged training
  under compute constraints.
- **Elizabeth Casey** — Data and preprocessing lead. Built loaders and
  unified-format pipelines for MRNet, KneeMRI, fastMRI, and CAI2R;
  managed the HuggingFace organization mirror; coordinated input-format
  compatibility with the modeling pipeline.
- **Maddie Larkin** — Evaluation and benchmarking lead. Defined the
  metric set, implemented the evaluation harness and confusion-matrix
  reporting, ran the zero-shot baseline and fine-tuned evaluations, and
  performed the error analysis.

All members contributed to experimental design, troubleshooting, the
final report, and code review. Code and intermediate results were shared
through a HuggingFace Organization (<https://huggingface.co/AUMLProject>)
and this GitHub repository.

## 12. References

- Bien, N. et al. (2018). *Deep-learning-assisted diagnosis for knee MRI:
  Development and retrospective validation of MRNet.* PLOS Medicine.
- Stajduhar, I. et al. (2017). *Semi-automated detection of anterior
  cruciate ligament injury from MRI.* Computer Methods and Programs in
  Biomedicine.
- Zbontar, J. et al. (2018). *fastMRI: An open dataset and benchmarks for
  accelerated MRI.* arXiv:1811.08839.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models.* arXiv:2106.09685.
- Google Health. *MedGemma-4B-IT.*
  <https://huggingface.co/google/medgemma-4b-it>
