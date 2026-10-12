# DeepGuard AI: Deepfake Detection System
### Protecting Images and Videos Using Deep Learning and Computer Vision
**Final-Year Computer Science & Engineering Capstone Project**

---

## 1. Project Title
**DeepGuard AI – Deepfake Detection: Protecting Images and Videos Using Deep Learning**

---

## 2. Abstract
The proliferation of deep learning-based image synthesis techniques (Generative Adversarial Networks, Autoencoders, and Diffusion Models) has made the creation of realistic hyper-manipulated visual media ("deepfakes") effortless. Such manipulations pose grave threats to public trust, democratic institutions, legal evidence, and individual privacy. This project delivers **DeepGuard AI**, a forensic deep learning system that detects manipulated human faces in both static images and temporal video streams. By leveraging transfer learning with a customized **ResNet18** convolutional backbone, localized facial extraction via **Haar Cascade classifiers**, and temporal multi-frame probability aggregation with calibrated uncertainty gating, DeepGuard AI achieves transparent, reproducible, and verifiable media authenticity verification.

---

## 3. Problem Statement
Manual inspection of modern synthetic media is unreliable because manipulation boundaries and blending seams are often imperceptible to the human eye. Automated detection systems frequently suffer from:
1. **Frame-to-video data leakage**: Models evaluated on frames from videos also present in training show falsely inflated accuracy.
2. **Single-frame brittleness**: Video authenticity determined from a single isolated frame yields high false alarm rates.
3. **Overconfident binary forced choices**: Forcing ambiguous or low-quality media into either "REAL" or "FAKE" creates dangerous false positives/negatives.

DeepGuard AI directly solves these problems through video-level dataset partitioning, temporal aggregation, and explicit uncertainty thresholds.

---

## 4. Objectives
- Localize and isolate human faces from complex backgrounds using robust computer vision.
- Fine-tune a pretrained deep convolutional neural network (ResNet18) using transfer learning for binary classification (0 = FAKE, 1 = REAL).
- Enforce strict zero-leakage video-level dataset partitioning across training, validation, and testing sets.
- Implement multi-frame temporal probability aggregation (Mean, Median, and Majority Voting) for video-level decisions.
- Introduce calibrated uncertainty thresholds (`p_fake >= 0.65` $\to$ FAKE, `p_fake <= 0.35` $\to$ REAL, otherwise UNCERTAIN).
- Provide a secure, local SQLite authentication system using PBKDF2-HMAC-SHA256 password hashing.
- Deliver an interactive, cyber-forensics Streamlit web application.

---

## 5. Key Features
- **Dual Media Modality**: Supports static images (`.jpg`, `.jpeg`, `.png`, `.webp`) and videos (`.mp4`, `.avi`, `.mov`, `.mkv`).
- **Face Localization**: Automatically isolates the primary facial region and crops with bounding margins to preserve border blending seams.
- **Uncertainty Classification**: Avoids forced false certainty by flagging borderline media as `UNCERTAIN`.
- **Temporal Analysis**: Provides per-frame probability timeline charts for video forensic inspection.
- **Academic Transparency**: Reports actual confusion matrix, precision, recall, and loss curves derived directly from test partitions without fabrication.
- **Zero Cloud Dependency**: CPU and CUDA compatible; local SQLite authentication without external API keys.

---

## 6. Technology Stack
- **Programming Language**: Python 3.11+ / Python 3.14
- **Deep Learning Framework**: PyTorch, torchvision
- **Computer Vision**: OpenCV (`opencv-python`)
- **Data & Mathematics**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib
- **Web Interface**: Streamlit
- **Security / Storage**: SQLite3, `hashlib` (PBKDF2-HMAC-SHA256), `secrets`

---

## 7. System Architecture
```
                  ┌───────────────────────────────┐
                  │    Input Media (Image/Video)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  Temporal Frame Sampling (20) │ (Videos only)
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Face Detection (Haar Cascade) │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Crop Face + 15% Border Margin │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Preprocessing: 224x224 Norm   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │      ResNet18 Backbone        │
                  │        Dropout (0.5)          │
                  │    Linear Classifier Head     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  Softmax Probability Scoring  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Multi-Frame Aggregation (Mean)│
                  └───────────────┬───────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      p_fake >= 0.65       0.35 < p < 0.65       p_fake <= 0.35
     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
     │     FAKE     │     │   UNCERTAIN   │     │     REAL     │
     └──────────────┘     └───────────────┘     └──────────────┘
```

---

## 8. Benchmark Dataset Support
DeepGuard AI natively supports the **FaceForensics++** benchmark dataset (specifically the **DeepFakes** manipulation subset):
- **REAL Videos**: Pristine human facial recordings.
- **FAKE Videos**: Faces replaced using deep autoencoder/GAN manipulation pipelines.

> **Data Placement**:
> - Real videos $\to$ `ffpp_videos/real/*.mp4`
> - Fake videos $\to$ `ffpp_videos/fake/*.mp4`

---

## 9. Preprocessing Pipeline
1. `extract_frames.py`: Uniformly samples $N$ frames (default 20) across total video duration.
2. `extract_faces.py`: Detects facial bounding box using Haar Cascade, selects largest face, applies 15% margin, crops and saves.
3. `prepare_dataset.py`: Partitions face crops **strictly by original video identifier** (70% Train, 15% Validation, 15% Test) to eliminate data leakage.
4. `check_dataset.py`: Validates dataset directory integrity and asserts zero video cross-contamination.

---

## 10. Deep Learning Model Architecture
```
Layer Name                  Output Dimensions      Parameters / Details
-----------------------------------------------------------------------
ResNet18 Backbone           [Batch, 512, 1, 1]     Pretrained on ImageNet
AdaptiveAvgPool2d           [Batch, 512]           Global Average Pooling
Dropout(p=0.5)              [Batch, 512]           Regularization
Linear(512 -> 2)            [Batch, 2]             Logits for 2 classes
-----------------------------------------------------------------------
Class 0 = FAKE
Class 1 = REAL
```

---

## 11. Training Protocol (`train_transfer.py`)
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$, $\text{weight\_decay} = 1 \times 10^{-4}$)
- **Loss Function**: Categorical Cross-Entropy Loss
- **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=2)`
- **Regularization**: Dropout ($p=0.5$) and Early Stopping ($patience=4$)
- **Data Augmentation**: Random Horizontal Flip, Random Rotation ($\pm 10^\circ$), Color Jitter
- **Best Model Checkpoint**: Saved automatically to `models/best_resnet18_deepfake.pth`

---

## 12. Model Evaluation (`evaluate.py`)
Metrics evaluated strictly on the held-out test split:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Outputs**: `results/confusion_matrix.png`, `results/metrics.txt`, `results/training_history.png`

---

## 13. Video-Level Evaluation (`evaluate_video.py`)
An isolated frame prediction is noisy. DeepGuard AI analyzes videos via:
$$\bar{P}_{\text{fake}} = \frac{1}{M} \sum_{i=1}^{M} P(\text{fake} \mid \text{face}_i)$$
where $M$ is the number of valid faces detected. Decisions are gated:
- $\bar{P}_{\text{fake}} \ge 0.65 \implies \mathbf{FAKE}$
- $\bar{P}_{\text{fake}} \le 0.35 \implies \mathbf{REAL}$
- Otherwise $\implies \mathbf{UNCERTAIN}$

---

## 14. Authentication System (`auth.py`)
Local SQLite database (`users.db`) using:
- Cryptographic salts: `secrets.token_bytes(32)` (256 bits per user)
- Key derivation: `hashlib.pbkdf2_hmac("sha256", password, salt, 100000)`
- Timing attack prevention: `secrets.compare_digest()`

---

## 15. Installation & Setup

### Step 1: Clone or Navigate to Project Directory
```powershell
cd C:\Users\shami\.gemini\antigravity\scratch\Deepfake_detection
```

### Step 2: (Optional) Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 16. Step-by-Step Execution Guide

### 1. (Optional) Generate Demonstration Sample Videos & Images
```powershell
python generate_sample_data.py
```

### 2. Extract Frames from Videos
```powershell
python extract_frames.py --frames_per_video 15
```

### 3. Extract and Crop Faces
```powershell
python extract_faces.py
```

### 4. Partition Dataset with Zero Video Leakage
```powershell
python prepare_dataset.py --train_ratio 0.70 --val_ratio 0.15
```

### 5. Check Dataset Health and Leakage
```powershell
python check_dataset.py
```

### 6. Train the ResNet18 Transfer Learning Model
```powershell
python train_transfer.py --epochs 5 --batch_size 16
```

### 7. Evaluate Model on Test Partition
```powershell
python evaluate.py
```

### 8. Run Video-Level Evaluation on a Sample Video
```powershell
python evaluate_video.py --video ffpp_videos/real/real_video_01.mp4
```

### 9. Launch the Futuristic DeepGuard AI Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- **Default Viva Login**: Username: `admin` | Password: `admin123`

---

## 17. Project Folder Structure
```
Deepfake_detection/
│
├── app.py                     # Streamlit frontend application
├── auth.py                    # SQLite PBKDF2 authentication module
├── model_def.py               # Canonical ResNet18 model definition
├── train.py                   # Training entrypoint alias
├── train_transfer.py          # Transfer learning training pipeline
├── evaluate.py                # Test set evaluation and confusion matrix
├── evaluate_video.py          # Video-level multi-frame evaluation
├── extract_frames.py          # Video frame extraction script
├── extract_faces.py           # Face detection and cropping script
├── prepare_dataset.py         # Video-partitioned dataset splitter
├── check_dataset.py           # Dataset health and leakage checker
├── dataset_loader.py          # PyTorch ImageFolder and transforms
├── generate_sample_data.py    # Demonstration sample generator
├── requirements.txt           # Python package requirements
├── README.md                  # Comprehensive academic documentation
├── VIVA_QUESTIONS.md          # Viva questions and answers guide
│
├── cascade/
│   └── haarcascade_frontalface_default.xml # OpenCV face cascade
│
├── models/
│   └── best_resnet18_deepfake.pth          # Trained model checkpoint
│
├── dataset/
│   ├── splits_manifest.json                # Split manifest
│   ├── train/ (real / fake)
│   ├── val/   (real / fake)
│   └── test/  (real / fake)
│
├── frames/ (real / fake)
├── faces/  (real / fake)
├── ffpp_videos/ (real / fake)
├── sample_images/
│
└── results/
    ├── confusion_matrix.png
    ├── metrics.txt
    ├── training_history.png
    └── training_history.json
```

---

## 18. Limitations
1. **Resolution & Compression Sensitivity**: Heavy video compression (e.g. WhatsApp / TikTok re-encoding) can destroy subtle boundary artifacts.
2. **Pose & Occlusion**: Extreme profile faces (>60° yaw) or heavy occlusions (masks, sunglasses) degrade Haar Cascade detection.
3. **Out-of-Distribution Generators**: Models trained on specific manipulation algorithms may generalize with lower confidence on radically different generation paradigms (e.g. Latent Diffusion).

---

## 19. Future Enhancements
- Integrate 3D-CNN or Vision Transformers (ViT) to model temporal temporal continuity across video frames.
- Add MTCNN or RetinaFace for robust facial landmark detection under extreme angles.
- Implement Frequency Domain Analysis (FFT / DCT) to identify spectral artifacts left by upsampling layers in GAN generators.
