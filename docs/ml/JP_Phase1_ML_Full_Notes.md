# JP — Phase 1 Machine Learning Research & Protocol (Full Notes)

**Project:** Deepfake Detection App  
**Scope of this file:** Full working notes for Days 10–13, including research, team decisions, protocol, experiment planning, and compute planning. No sections have been intentionally summarized away.  
**Status:** Research and planning; this is not evidence of a trained deepfake detector.

> **Read this first (phase ownership):** These notes sometimes describe the custom CNN as optional and EfficientNet-B0 as the preferred starting model. The team's 35-week comprehensive plan is more specific: **Phase 2 (Pat) requires a custom CNN baseline**, **Phase 3 (Abdullah) adds ResNet50**, and **Phase 5 (JP) adds EfficientNet-B0 for comparison**. Follow that timeline unless the team explicitly agrees to change it. See `ML_Phase1_to_Phase2_Handoff.md` for the current implementation sequence.

---

## Day 10 — Deepfake Detection Research

Goal: Research potential datasets, machine learning models, and evaluation strategies for our deepfake photo detection application.

### Topics Covered

**Types of Deepfakes:** Discussed the difference between fully AI-generated portraits and manipulated real photographs, such as face swaps.

**Dataset Research:** Investigated GenFace, FaceForensics++, Celeb-DF, and DF40 as potential datasets for training and evaluating our models.

**Model Selection:** Identified three models to compare: a custom CNN, ResNet50, and EfficientNet-B0.

**Generalization:** Learned why a model trained on one AI generator might struggle to detect images from another generator. Discussed testing on unfamiliar generators to measure real-world performance.

**Dataset Bias:** Explored how models can learn misleading patterns, such as differences in JPEG/PNG formats, compression, or image resolution, instead of actual deepfake characteristics.

**Data Splitting:** Discussed separating training, validation, and testing data while preventing duplicate images and related photographs from appearing across splits.

### Key Takeaways

Our deepfake detector needs to recognize patterns that generalize beyond its training dataset. High accuracy alone does not guarantee reliable performance on unfamiliar AI-generated images.

We should compare multiple CNN architectures, carefully control dataset differences, and evaluate performance on AI generators excluded from training.

### Next Steps

Decide whether our initial model will focus on fully generated portraits, manipulated photographs, or both.

Select an appropriate dataset.

Develop our ML training and evaluation protocol (Day 11).

Plan experiments comparing the three proposed models.

## Codex Research Report — October 28, 2026

Goal: Identify realistic datasets, pretrained models, and training strategies for our deepfake detection application.

Codex produced a research report comparing 8 datasets and 9 model options, including accessibility, licensing, computational requirements, existing GitHub implementations, and evaluation strategies.

### 1. Dataset Research

The datasets investigated included DiffusionFace, GenFace, AI-Face, DFFD, DF40, FaceForensics++, Celeb-DF, and 140k Real and Fake Faces.

**Recommended starting dataset:** DiffusionFace

Contains real facial images and AI-generated or manipulated faces, including diffusion-based generation methods.

Selected archives are publicly listed on Zenodo.

We can download individual archives instead of the entire dataset.

The full archive collection is approximately 72.4 GB, but selected archives are considerably smaller.

We will focus on fully AI-generated portraits and exclude face swaps and other manipulations from our initial training scope.

The dataset is designated for research purposes only.

Other datasets worth considering:

**GenFace:** Strong collection of generated and manipulated faces, but access and licensing present challenges.

**AI-Face:** Large, diverse facial-image dataset that may be useful for future training and fairness evaluation.

**DF40:** Covers 40 forgery methods and could help evaluate generalization to unfamiliar generators. Access and storage requirements are significant.

Dataset counts, permissions, download accessibility, and image provenance still require verification before implementation.

### 2. Model Research

**Recommended primary model:** EfficientNet-B0

Pretrained on ImageNet.

Approximately 5.3 million parameters.

Smaller and generally less computationally demanding than ResNet50.

Can be adapted for binary real/fake classification through transfer learning.

**Comparison model:** ResNet50

Already explored during our PyTorch learning exercises.

Will be fine-tuned using the same dataset splits as EfficientNet-B0.

Allows us to compare detection performance and computational efficiency.

**Pretrained reference:** UniversalFakeDetect

Existing AI-generated image detector based on CLIP ViT-L/14.

Includes a trained detection head.

Can provide a reference for comparing our newly trained models against an existing detector.

Its published results do not guarantee performance on newer AI-generated profile pictures.

Our custom CNN can also serve as a simpler baseline if time permits.

### 3. Hardware and Feasibility

Research suggests the MacBook M3 Pro with PyTorch MPS should support smaller-scale training experiments.

We plan to:

Start with smaller datasets and batch sizes.

Experiment with freezing pretrained layers before fine-tuning.

Measure memory usage and training speed.

Consider university GPU resources or cloud notebooks if larger experiments become necessary.

Actual training performance on our hardware has not yet been benchmarked.

### 4. Major Research Takeaways

A high benchmark accuracy does not necessarily mean a deepfake detector will perform well on real-world profile pictures.

Important concerns include:

Generalization to unfamiliar AI generators.

Data leakage between training and testing.

Shortcut learning caused by image resolution or compression.

False positives on genuine photographs.

Robustness to JPEG compression and resizing.

Model confidence calibration and explainability.

Research status: The report provides recommendations and verified resource links, but no datasets were downloaded or models trained as part of the Codex research task.

## Day 11 — ML Training & Evaluation Protocol

Goal: Develop a proposed methodology for training, comparing, and evaluating our deepfake detection models.

We completed working drafts of all eight sections of our ML protocol.

### 1. Project Objective

Our initial model will focus on distinguishing authentic human portraits from fully AI-generated portraits.

Face swaps and manipulated real photographs may be added later, but keeping the first version focused should make training and evaluation more manageable.

### 2. Dataset Selection

We selected DiffusionFace as our preferred initial dataset candidate, pending team approval and final access/provenance checks.

Our proposed approach is to begin with:

5,000 genuine portraits.

5,000 fully AI-generated portraits.

10,000 images total for the first substantive experiment.

We may first use approximately 2,000 images to test the training pipeline, then scale toward 20,000–40,000 images if resources permit.

### 3. Dataset Splitting

Our proposed development dataset split is:


| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 7,000 |
| Validation | 15% | 1,500 |
| Testing | 15% | 1,500 |

We will aim to preserve class balance while preventing duplicate images, related source photographs, and overlapping identities from appearing across splits.

We also plan to reserve an additional dataset containing images from AI generators excluded from training to evaluate generalization.

### 4. Image Preprocessing

Before training, images will be:

Converted to RGB.

Resized and cropped according to model requirements, initially around 224 × 224 pixels.

Converted to tensors and normalized using the relevant pretrained model's recommended preprocessing.

Checked for differences in resolution, compression, and file formats that could create misleading patterns.

Controlled data augmentation may be used during training to improve generalization.

### 5. Model Selection

We will primarily compare:

**EfficientNet-B0:** Our preferred lightweight pretrained model.

**ResNet50:** Our larger pretrained comparison model.

**Custom CNN:** An optional model trained from scratch to establish a baseline.

We will also evaluate UniversalFakeDetect as an already-trained reference if integration is feasible.

The main models will use the same underlying training and evaluation splits for a fair comparison.

### 6. Training Methodology

Our initial training configuration will use:

**Loss function:** CrossEntropyLoss.

**Optimizer:** Adam.

**Batch size:** Initially 8–16.

**Epochs:** Up to 10 for initial experiments.

**Learning rate:** Approximately 0.001 for newly initialized classification layers, with smaller rates during fine-tuning.

**Transfer learning:** Start with frozen pretrained layers, then consider unfreezing later layers.

We also discussed overfitting, early stopping, and model checkpointing.

If validation performance begins declining while training performance improves, we should not automatically continue training. Instead, we can stop early and restore the best validation-selected checkpoint.

### 7. Model Evaluation

We will measure:

Accuracy.

Precision.

Recall.

F1-score.

ROC-AUC.

False positive rate.

Our main priority is reducing situations where genuine photographs are incorrectly classified as AI-generated.

We propose evaluating a classification threshold targeting approximately 1% false positive rate, selected using validation data. This is a goal to test, not guaranteed performance.

We will also evaluate performance on unseen generators, sensitivity to image compression and resizing, and the reliability of model scores.

### 8. Reproducibility and Documentation

For every experiment, we will record:

Dataset name, version, and source.

Training, validation, and testing splits.

Model architecture and pretrained weights.

Image preprocessing and augmentation.

Batch size, learning rate, optimizer, loss function, and epochs.

Random seeds and software versions.

Hardware configuration.

Model checkpoints and evaluation results.

Known limitations and observed errors.

This will allow team members to reproduce experiments and compare models consistently.

### Key Takeaways

A model can achieve extremely high training accuracy while performing poorly on unfamiliar images due to overfitting.

Transfer learning allows pretrained models to adapt existing visual features to our specific deepfake detection task.

Using identical dataset splits is essential for meaningful model comparisons.

Dataset quality, diversity, and leakage prevention are just as important as model architecture.

False positive rate is especially important for our application's use case.

Reproducibility requires documenting much more than model accuracy.

## Team Check-In — ML Decisions

Following our team discussion, we agreed on the following direction for the initial machine learning implementation.

**Project Scope:** Our initial deepfake detector will focus on distinguishing authentic human profile photographs from fully AI-generated portraits. Face swaps and other image manipulation techniques may be considered in future development.

**Dataset:** DiffusionFace is our preferred dataset, provided its available archives contain the appropriate real and fully generated portrait images. We will verify image categories, access, labels, and licensing before training.

**Model Selection:** We plan to narrow our initial comparison to two models:

EfficientNet-B0 as the primary model.

ResNet50 as the comparison model.

Both will use transfer learning and be evaluated using consistent dataset partitions. Additional pretrained detectors may be considered later if time permits.

**Evaluation Priority:** Minimizing false positives on genuine photographs will be a primary concern. We will evaluate precision, recall, accuracy, F1-score, ROC-AUC, and false positive rate.

**Training Approach:** Transfer learning is expected to be acceptable for the senior project, although formal professor confirmation may still be needed.

**Next Steps:** Finish hyperparameter and hardware planning, audit the DiffusionFace dataset, establish the training experiments, and coordinate model integration with the application backend.

## Day 12 — Hyperparameter Tuning & Experiment Planning

Goal: Develop a realistic and reproducible experiment plan for training and comparing our deepfake detection models.

### Topics Covered

**Hyperparameters:** Reviewed learning rates, batch sizes, epochs, optimizers, and how these settings influence model training.

**Learning Rates:** Discussed why excessively high learning rates can cause unstable training and why smaller learning rates are generally appropriate for fine-tuning pretrained layers.

**Batch Sizes:** Examined the tradeoff between memory requirements, gradient stability, and training efficiency.

**Early Stopping:** Reviewed how to monitor validation performance, prevent overfitting, and restore the best model checkpoint.

**Fair Experiments:** Learned to control experimental variables and document differences between configurations.

**Model Comparison:** Established a preliminary experiment plan for EfficientNet-B0 and ResNet50.

### Proposed Training Configuration


| Setting | Initial proposal |
|---|---|
| Dataset | Selected DiffusionFace portraits |
| Initial dataset size | 10,000 images |
| Pipeline debugging | 2,000 images |
| Train / validation / test | 70% / 15% / 15% |
| Batch size | 8 |
| Maximum epochs | 10 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Classifier learning rate | 0.001 |
| Fine-tuning learning rate | 0.0001 or 0.00001 |
| Early stopping | Patience of 3 epochs |

### Planned Experiments

Train the EfficientNet-B0 classification head with frozen pretrained layers.

Train the ResNet50 classification head under comparable conditions.

Fine-tune later EfficientNet-B0 layers.

Fine-tune later ResNet50 layers.

Test an additional learning rate on the strongest model.

The models will use identical dataset partitions and comparable tuning budgets.

### Evaluation Priorities

Our primary concern remains minimizing false positives on genuine photographs.

We will compare false positive rate, fake recall, precision, F1-score, ROC-AUC, training time, and memory requirements.

Classification thresholds will be selected using validation data. Independent test images and previously unseen AI generators will be used for final evaluation.

### Key Takeaways

A model with higher overall accuracy is not necessarily better for our application if it incorrectly flags substantially more genuine photographs.

Hyperparameter tuning should be systematic, computationally practical, and reproducible. Changing one variable at a time helps identify the effects of individual training settings.

**Status:** Day 12 completed. Experiment configurations are preliminary and have not yet been tested on actual deepfake data.

## Day 13 — Hardware, Compute Planning & Reproducibility

**Objective:** Design a portable and reproducible training environment that can be used by the entire senior project team, rather than relying on one developer's computer.

### Topics Covered

CPU, CUDA, and MPS: Studied how GPUs accelerate neural network training through parallel computation.

**Hardware portability:** Established that training scripts should automatically detect supported hardware.

**Memory management:** Examined how model size, image resolution, and batch size affect training memory.

**Storage planning:** Considered dataset storage, model checkpoints, and training logs.

**Training benchmarks:** Learned how to estimate full training time using smaller test runs.

**Compute environments:** Compared local computers, university GPUs, and cloud GPU options.

**Reproducibility:** Identified the configurations and metadata needed to repeat experiments.

### Proposed Team Compute Plan


| Component | Recommendation |
|---|---|
| Local development | CPU, CUDA, or Apple MPS |
| Preferred larger experiments | University GPU, if accessible |
| Alternative | Local or cloud GPU |
| Initial dataset | 2,000 images for pipeline testing |
| Development dataset | Approximately 10,000 images |
| Input resolution | 224 × 224 |
| Starting batch size | 8, adjustable for hardware |
| Maximum epochs | 10, with early stopping |
| Model architectures | EfficientNet-B0 and ResNet50 |
| Device selection | Automatic hardware detection |

### Reproducibility Requirements

Each training experiment should record the architecture, pretrained weights, dataset version, split assignments, training configuration, random seed, software environment, hardware, metrics, and saved checkpoints.

Training scripts, configuration files, and documentation should be version-controlled in GitHub. Raw datasets and large model checkpoints should be stored separately, with access and licensing requirements documented.

### Handoff Considerations

The final training pipeline should be executable by different teammates without requiring changes to the model architecture or core training logic.

The team should validate compatibility on multiple machines and use benchmark results to determine where larger experiments should run.

**Status:** Day 13 planning completed. The proposed training environment has not yet been fully implemented or benchmarked.
