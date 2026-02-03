# Penovate

A pen-based handwriting recognition system using IMU sensors and deep learning. The system captures motion data from a custom sensor-equipped pen writing on ordinary paper and recognizes lowercase English characters (a-z) in real time.

**Architecture:** CNN-BiLSTM hybrid model achieving 98.62% test accuracy on isolated character recognition.

**Implementation:** Two model variants provided:
- `CNN_BiLSTM` (hybrid_CNN_BiLSTM.ipynb) - main experimental model with optional batch normalization
- `CNN_LSTM` (model.ipynb) - alternative architecture for early testing

**Hardware:** Custom pen prototype with dual MPU-6050 IMUs, FSR sensor, Arduino Nano, and HC-05 Bluetooth module.

---

## Project Team
- **Pankaj Bhatt (THA078BEI025)**
- **Pratik Pokharel (THA078BEI027)**
- **Subham Gautam (THA078BEI042)**

---

## 1. Background

Existing handwriting recognition systems rely primarily on optical methods (OCR) or specialized touch-sensitive surfaces. These approaches require either post-processing of scanned images or dedicated hardware like graphics tablets. This project explores an alternative approach: capturing handwriting motion directly through inertial sensors embedded in a pen.

The system uses dual IMUs to capture 3D motion trajectories and an FSR to segment individual strokes. A CNN-BiLSTM network processes the resulting time-series data to classify lowercase characters. This approach enables real-time recognition on ordinary paper without requiring specialized writing surfaces.

---

## 2. System Architecture

**Hardware:** Pen prototype with Arduino Nano, dual MPU-6050 IMUs, FSR, and HC-05 Bluetooth module.

**Firmware:** Real-time sensor synchronization and data streaming at 100 Hz.

**Data Pipeline:** Butterworth filtering, stroke segmentation, feature engineering (18 channels from 12 raw sensor channels), and sequence normalization.

**Model:** CNN-BiLSTM network with 2 convolutional layers, 2-layer bidirectional LSTM, and fully connected classifier.  

---

### 2.1 Repository Structure

```
Penovate/
├── data/
│   ├── imu_dataset/          # Raw data (JSON per character)
│   └── processed_imu/        # Preprocessed NumPy arrays
├── results/
│   └── exp_*/                # Model checkpoints, logs, plots
├── scripts/
│   ├── dataset_collection.py
│   ├── process_imu_data.py
│   ├── predict_gui.py
│   └── realtime_predictor.py
├── hybrid_CNN_BiLSTM.ipynb   # Main experiments
├── model.ipynb               # Alternative model
└── requirements.txt
```

### 2.2 Hardware Implementation

| Component | Specification | Purpose |
|-----------|--------------|----------|
| Arduino Nano | ATmega328p | Sensor acquisition and streaming |
| MPU-6050 (×2) | I²C addresses 0x68, 0x69 | 6-axis IMU (accel + gyro) |
| FSR | Analog input | Stroke detection |
| HC-05 | Bluetooth 2.0 | Wireless data transmission |
| Battery | 2S Li-ion, 7.4V | Portable power |

Firmware samples sensors at 100 Hz and transmits 12-channel data packets (2 IMUs × 6 axes) over serial/Bluetooth.  

### 2.3 Data Pipeline

**Collection Phase** (`dataset_collection.py`):
1. Butterworth low-pass filter (order 2, 20 Hz cutoff)
2. FSR-based segmentation (START/END signals)
3. Relative motion features: IMU1 - IMU2 (12 → 18 channels)
4. Per-sequence z-score normalization
5. Save to JSON files (one per character)

**Training Phase** (`process_imu_data.py`):
1. Load and stratify split: 70% train, 15% val, 15% test
2. Pad/truncate to 184 timesteps (95th percentile)
3. Global normalization (mean/std from training set)
4. Augmentation: Gaussian noise (σ=0.05) + scaling (0.8-1.2×), 2 copies per sample
5. Export to NumPy arrays

---

## 3. Dataset

**Task:** Single lowercase character recognition (a-z, 26 classes)

**Sensor Channels:**
- Raw: 12 channels (2 IMUs × 6-axis each)
- Engineered: 18 channels (12 raw + 6 relative motion)
- Sampling rate: 100 Hz

**Collection:**
- ~130 samples per class from one writer
- Total: 3,380 original samples

**Split:**
- Original split: 2,366 train / 507 val / 507 test (70%/15%/15%)
- After augmenting training set (1 original + 2 copies = 3× per sample):
  - Train: 7,098 samples
  - Validation: 507 samples (not augmented)
  - Test: 507 samples (not augmented)
  - **Total: 8,112 samples**

**Sequence Length:**
- Mean: 131.6 timesteps, Std: 30.2
- Range: 25-292 timesteps
- Fixed length: 184 timesteps (95th percentile)
- Truncated: 176 sequences (5.2%)

**Storage:** JSON files per character in `data/imu_dataset/`, preprocessed arrays in `data/processed_imu/`

---

## 4. Model Architecture

**CNN-BiLSTM Hybrid:**
```
Input (B, 184, 18)
  → Conv1D(18→64, k=5) + [BatchNorm] + ReLU + MaxPool(2)
  → Conv1D(64→128, k=3) + [BatchNorm] + ReLU + MaxPool(2)
  → BiLSTM(128→128, layers=2, dropout=0.5)
  → FC(256→128) + ReLU + Dropout(0.5)
  → FC(128→26) + Softmax
```

**Implementation:** See `hybrid_CNN_BiLSTM.ipynb` for CNN_BiLSTM class, Trainer, and data loaders. Alternative implementation in `model.ipynb`.

### 4.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Epochs | 30 |
| LSTM hidden size | 128 |
| Dropout | 0.5 |
| Batch normalization | True/False (both tested) |
| Optimizer | AdamW (lr=1e-4, wd=1e-3) |
| LR scheduler | StepLR (step=8, γ=0.7) |
| Early stopping | 7 epochs (on macro-F1) |
| Seeds | 42, 123, 7 |
| Augmentation | Training only: +2 copies (3× total) with noise (σ=0.05) + scaling (0.8-1.2×) |
| Framework | PyTorch |  

---

## 5. Results

### 5.1 Experiments

| Experiment | BatchNorm | Test Loss | Test Acc | Macro F1 |
|-----------|-----------|-----------|----------|----------|
| seed42 | True | **0.0608** | **98.62%** | **0.9863** |
| seed42 | False | 0.1118 | 98.03% | 0.9803 |
| seed123 | True | 0.0652 | 98.42% | 0.9841 |
| seed123 | False | 0.1228 | 98.22% | 0.9824 |
| seed7 | True | 0.0670 | 98.62% | 0.9863 |
| seed7 | False | 0.1049 | 98.42% | 0.9843 |

**Key findings:**
- Batch normalization consistently improves performance
- Best accuracy: 98.62% (with BN, seeds 42 and 7)
- Training is stable and reproducible across seeds
- Main confusion: similar motion patterns (m/n, c/g, o/a)

**Outputs per experiment** (`results/exp_*/`):
- Model checkpoint: `best.pth`
- Metrics: `test_summary.txt`, `train_log.csv`
- Plots: confusion matrices, accuracy/loss curves
- Arrays: `y_pred_test.npy`, `y_true_test.npy`

### 5.2 Visualizations

All experiments include:
- Training/validation curves (loss, accuracy)
- Confusion matrices (counts and normalized)
- Per-class metrics in `test_summary.txt`

Best model: [results/exp_bn_bs32_seed42_drop05/](results/exp_bn_bs32_seed42_drop05/)

---

## 6. Requirements

**Software:**
- Python 3.8+
- PyTorch, NumPy, SciPy, scikit-learn, pandas, matplotlib, pyserial, tkinter
- Install: `pip install -r requirements.txt`

**Compute (Training):**
- CPU: Multi-core processor
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (CUDA-compatible)
- Storage: ~500MB

**Hardware (Pen):**
- Arduino Nano, 2× MPU-6050, FSR, HC-05, 2S Li-ion battery

---

## 7. Usage

### 7.1 Data Collection
```bash
python scripts/dataset_collection.py
```
Collects sensor data via serial (COM13, 115200 baud), applies filtering, computes features, normalizes, and saves to JSON.

### 7.2 Data Preprocessing
```bash
python scripts/process_imu_data.py
```
Loads JSON files, splits data (70/15/15), pads sequences, augments training set, normalizes globally, exports to NumPy.

### 7.3 Training
Open `hybrid_CNN_BiLSTM.ipynb` in Jupyter and run all cells. Model checkpoints and logs saved to `results/exp_*/`.

### 7.4 Inference

**GUI version:**
```bash
python scripts/predict_gui.py
```
Tkinter interface with live predictions. Edit COM port in script (default: COM6).

**Terminal version:**
```bash
python scripts/realtime_predictor.py
```
Lightweight CLI predictor.

Both require `best.pth` and `label_map.json`. Note: Scripts use BatchNorm by default.

---

## 8. Installation

```bash
git clone https://github.com/PunksB1602/Penovate.git
cd Penovate
pip install -r requirements.txt
```

---

## 9. Limitations

While the system achieves high accuracy for isolated character recognition, several limitations exist:

### 10.1 Scope Limitations
- **Single character only:** The model is trained and optimized exclusively for isolated lowercase letters (a-z). Continuous word or sentence writing is not supported.
- **Writer dependency:** Dataset collected from single writer; generalization to different writing styles not extensively tested.
- **Lowercase only:** Uppercase letters, numbers, and special characters are not included in the current dataset.

### 10.2 Technical Limitations
- **Fixed sequence length:** Sequences are padded/truncated to 184 timesteps, which may lose information for very long strokes or add noise for very short ones.
- **Confusion between similar characters:** Some character pairs show higher confusion rates:
  - Visually similar: m/n, c/e, i/l
  - Motion similar: c/g, o/a, u/v
- **Hardware constraints:** Requires calibrated IMU sensors; sensor drift over time may affect accuracy.
- **Environmental sensitivity:** Not tested for robustness to different paper surfaces, pen grip variations, or writing speeds.

### 10.3 Real-time Performance
- **Latency:** Serial communication and preprocessing introduce minimal delay (~100-200ms per character).
- **Segmentation dependency:** Relies on FSR pressure threshold; very light or heavy pressure may cause false triggers.

### 10.4 Dataset Size
- **Limited samples:** ~130 original samples per class (augmented to 273) from one writer.
- **No cross-writer validation:** Model performance on unseen writers is unknown.

---

## 10. Future Work

**Model:**
- Continuous text recognition (seq2seq, attention/Transformer architectures)
- Multi-writer training and evaluation
- Alphanumeric support (A-Z, 0-9, punctuation)
- Model compression for edge deployment

**Data:**
- Expand to multiple writers, surfaces, and grips
- Public dataset release for benchmarking
- Compare with HMM, DTW, other deep learning baselines

**Hardware:**
- Magnetometer integration
- Analog FSR readings as feature
- Miniaturization and power optimization

**Applications:**
- Text editor integration
- Signature verification
- Gesture/symbol recognition
- Accessibility tools

---

## 11. License

MIT License. See LICENSE for details.

---

## 12. Contact

- **Pankaj Bhatt** (THA078BEI025): pbecie16@gmail.com
- **Pratik Pokharel** (THA078BEI027): pratikpokhrel14@gmail.com
- **Subham Gautam** (THA078BEI042): gautamsubham65@gmail.com


