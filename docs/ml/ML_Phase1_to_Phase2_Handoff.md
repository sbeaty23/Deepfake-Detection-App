# Machine Learning Handoff — Phase 1 → Phase 2

**Project:** Deepfake Detection App — senior capstone  
**Outgoing Phase 1 ML owner:** JP  
**Incoming Phase 2 ML owner:** Pat  
**Reference:** *Deepfake Detection Comprehensive Project Plan*, 35-week rotation, Phases 1–5  
**Status as of handoff:** Protocol/research and PyTorch exercises are available. **No deepfake-specific model has been trained or validated as part of JP's Phase 1 work.**

## 1. What JP completed for Phase 1

- Researched real vs. fully synthetic portraits, generator-held-out evaluation, preprocessing biases, and dataset availability/licensing.
- Proposed a data and evaluation protocol, model candidates, bounded tuning budget, and compute approach.
- Built introductory PyTorch exercises under [`ml/pytorch-foundations/`](../../ml/pytorch-foundations/): a Fashion-MNIST CNN, training, prediction, evaluation, and a ResNet50 *architecture demonstration*. These are **learning/starter examples, not an operational deepfake detector**.
- Documented precision, recall, F1, ROC-AUC, false-positive rate, generalization, early stopping, checkpointing, and portable CPU/CUDA/MPS training considerations.
- Discussed the initial scope with the team: **authentic human portraits versus fully AI-generated portraits**. Face swaps and edited-real images are outside the initial binary MVP.

**Detailed background:** [`JP_Phase1_ML_Full_Notes.md`](JP_Phase1_ML_Full_Notes.md). For extensive dataset/model-source research, use the separate October 28 research report if the team has access to it; this handoff does not replace that report.

### Reproduce the existing Phase 1 starter exercise

From the repository root, the exercise's own README documents:

```bash
cd ml/pytorch-foundations
python -m pip install -r requirements.txt
python train.py
python predict.py
```

`train.py` trains the **Fashion-MNIST practice CNN** and writes a checkpoint used by `predict.py`. Dataset download and compatible dependencies may be required. **These instructions are taken from the existing repository README; this handoff did not rerun them on a teammate's machine.** A passing Fashion-MNIST exercise does not prove that the real-versus-synthetic face pipeline is ready.

## 2. The sequence in the approved project plan — do not skip Phase 2

| Phase | ML owner | Assigned deliverable |
|---|---|---|
| **1 — Foundations** | **JP** | Research, model protocol, starter PyTorch example, definitions/compute needs |
| **2 — Dataset pipeline and custom CNN** | **Pat** | Configuration-driven **custom CNN baseline**, checkpoint, metrics, false-positive/false-negative analysis |
| **3 — Transfer learning** | **Abdullah** | **ResNet50** training and fair CNN comparison; serving-ready checkpoint/package |
| **4 — Explainable MVP** | **Shawn** | Calibration, Brier/reliability analysis, development-selected threshold, model report |
| **5 — Comparison and generalization** | **JP** | **EfficientNet-B0**, cross-model comparison, inference/latency, unseen-generator assessment |

The research favored EfficientNet-B0 as a strong future deployment candidate, but **that is not an instruction to replace Pat's required custom CNN baseline**. Do not represent an ImageNet-pretrained head as a deepfake-trained detector.

## 3. Immediate implementation steps for Pat (Phase 2, Weeks 4–6)

1. **Coordinate with Abdullah (Phase 2 data/forensics)** on permitted portrait sources, real/synthetic labels, primary-face cropping, duplicate/identity grouping, and the exact versioned train/validation/test manifest. The current [`forensics/`](../../forensics/) folder has ingestion, grouping and split *code and configuration examples*; an actual curated dataset and completed audit are **not confirmed** by this handoff.
2. **Agree on the labels and data interface** before coding (for example, `real` versus `synthetic`, fixed numeric encoding, image path and split fields). Preserve generator/source metadata so the same images and split assignments can be used in later phases.
3. **Create a training pipeline in `ml/`** that reads the approved split manifests, performs consistent RGB/face-crop preprocessing, and trains a modest custom CNN **from scratch**. Adapt the Fashion-MNIST starter architecture as a learning reference, not as an unchanged production model.
4. Use **configuration-controlled training** (model settings, input resolution, random seed, batch size, learning rate, epochs) and choose CPU/CUDA/MPS automatically when supported. Start with a tiny smoke run; scale only after the data path works.
5. **Save best-validation checkpoints** and a run log containing code revision, exact manifests, data/transforms, package versions, device, hyperparameters, train/validation losses, and evaluation thresholds.
6. Evaluate using **accuracy, fake precision, fake recall, F1, ROC-AUC (when computable), and false-positive rate on genuine portraits**. Report confusion matrix and several traceable false positives and false negatives. **Select checkpoints and thresholds using validation data only**; do not tune on the final held-out test set.
7. **Deliver a reproducible baseline:** a teammate can run installation, one short training command, evaluation, and inference on a sample image without private assumptions about Pat's laptop. Provide checkpoint access separately (not as large Git blobs).

**Phase 2 completion gate from the plan:** reproducible splits; duplicate checks pass; CNN baseline reproduces; corrupted/oversized images are rejected by the application pipeline; no test-data tuning. The backend upload-validation portion is owned by Shawn in Phase 2, not solely by ML.

## 4. Preliminary ML protocol — not yet measured

| Item | Proposal / caution |
|---|---|
| MVP | Binary authentic photo vs. **fully AI-generated portrait** |
| Dataset candidate | Selected **DiffusionFace** archives; verify actual labels, permissions, availability, provenance and intended use first |
| Initial data-scale idea | ~2,000 images for debugging; ~10,000 total (roughly 5,000 / 5,000) for substantive development if sources allow |
| Initial split proposal | 70% train, 15% validation, 15% test, **after deduplication and source/identity grouping** |
| Holdout | A separate test on generators or generator families excluded from training, when coverage allows |
| Image input | RGB, face-specific crop, ~224×224 as a starting point; use architecture-appropriate transforms |
| Custom CNN training | Start with a modest Adam/CrossEntropyLoss configuration; choose parameters through bounded validation experiments |
| Future transfer-learning candidates | ResNet50 in Phase 3; EfficientNet-B0 in Phase 5; frozen head before selective fine-tuning |
| Starting parameters for later experiments | Batch size ~8 (adjust for hardware), up to 10 epochs, head LR ~0.001, fine-tuning LR ~0.0001/0.00001, patience ~3 **as exploratory proposals** |
| Selection priority | **False positives on genuine photos**, balanced against fake recall; also precision, F1, ROC-AUC and latency |
| Threshold | Explore low-FPR operating points (e.g., ~1% target on validation data), **not a promised result** |
| Compute | Same scripts across CPU/CUDA/Apple MPS; benchmark locally, prefer available university GPU for longer runs |

**Important:** A configuration that works for transfer learning is *not necessarily* optimal for the Phase 2 CNN from scratch. Treat learning rates and batch sizes as initial trial values, not fixed requirements.

## 5. Terms for the Phase 1 deliverable

We label **synthetic** as the positive class for reporting unless the team agrees otherwise.

- **Precision:** `TP / (TP + FP)` — of images flagged synthetic, the fraction that actually are synthetic.
- **Recall (sensitivity):** `TP / (TP + FN)` — of all synthetic images, the fraction detected.
- **F1:** harmonic mean of precision and recall; `2PR/(P+R)` when defined.
- **False-positive rate:** `FP / (FP + TN)` — of genuine images, the fraction incorrectly flagged synthetic; an especially important safety/usability measure here.
- **ROC-AUC:** threshold-independent summary of how well scores rank synthetic above real; not the same as accuracy.
- **Calibration:** how closely predicted probabilities match observed outcomes (e.g., Brier score and reliability curves); formal calibration is scheduled for Phase 4.
- **Latency:** actual elapsed inference time per image/request, reported with environment, preprocessing inclusion, and measurement method.

## 6. Connections to other team components

- **Data/forensics (Abdullah in Phase 2):** own versioned manifests, image quality checks, cropping, duplicate/source/identity leakage prevention. See [`forensics/`](../../forensics/); keep image datasets and license-restricted assets out of Git.
- **Backend (Shawn in Phase 2):** supply model-loading interface and image validation; later serving should expose `POST /api/analyze` with label, score/probability definition, model version, processing time, and an explanation reference when implemented. The complete API is a team contract, not an already-verified integration.
- **Frontend (JP in Phase 2):** build upload/preview/error states using explicit mock fixtures until backend/ML integration is ready.
- **Later ML owners:** keep checkpoint format, class-label mapping, preprocessing/version identifiers, and evaluation code documented so ResNet50 and EfficientNet can use the *same* benchmark without leakage.

## 7. Definition of a successful handoff / unresolved decisions

**The next owner should be able to:** open the research notes; inspect the starter code; know which part is only demonstration; point to approved input manifest/split files; repeat a tiny data-to-evaluation experiment once implemented; and explain how checkpoint and class probabilities will be handed to the backend.

**Still requires verification or decision:**

- Whether the selected DiffusionFace archives actually provide sufficient eligible authentic and *fully generated* face images and what licenses/attribution apply.
- Data/compute availability and any permission to use a university GPU.
- Approved train/validation/test partitions, metadata schema and numeric class mapping.
- Whether the professor approves transfer-learning methods as planned (relevant to Phases 3 and 5).
- Whether the forensics data pipeline and the custom CNN can execute together on teammates' machines.
- Ownership and format of checkpoint distribution, inference preprocessing, and the eventual backend interface.

**Do not claim:** a trained face deepfake model, verified model accuracy, guaranteed 1% false-positive rate, completed dataset audit, or successful cross-machine reproduction until these have actually occurred.

---

**Handoff action:** Pat should start with the **Phase 2 custom CNN baseline** using splits coordinated with Abdullah. Keep the larger EfficientNet-B0/ResNet50 experiment plan as background for later owners. Review the findings together at the next team check-in, and update this file if priorities change by agreement.
