🧠 AEGISMIND: Advanced Deepfake Detection System

“Detect the unreal — Protect the real.”
A next-gen Deepfake Detection Platform developed for hackathons and real-world cybersecurity applications, powered by a ResNeXt-LSTM hybrid model for multi-modal deepfake analysis (images, videos, and live webcam feeds).

🚀 Key Highlights

🔍 Multi-Modal Detection — Detects deepfakes in images, videos, and live webcam streams

🧩 Dual-Domain Analysis — Combines spatial and frequency domain analysis for higher accuracy

⚡ Real-Time Processing — Frame-by-frame webcam and video stream analysis

🧠 AI Explainability — Integrated GradCAM heatmaps for model transparency

🧰 Error Handling — Built-in safeguards for corrupted or oversized files

🧾 Confidence Metrics — Displays confidence scores with visual indicators

🔐 Security — Uses SHA256 integrity verification for file validation

🛠️ Technology Stack
Layer	Technology	Purpose
Frontend	Streamlit + Custom CSS	Interactive and responsive UI
Backend	PyTorch (ResNeXt-LSTM)	Deepfake detection engine
Vision	OpenCV + PIL	Image/Video preprocessing
Model Explainability	Grad-CAM	AI transparency and trust
Security	hashlib (SHA256)	File integrity verification
⚙️ Installation
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Application
streamlit run app/streamlit_app_new.py


Then open the local URL (e.g., http://localhost:8501).

💻 Usage Guide
🔹 Step 1: Load the Model

Click “Load Detection Model” from the sidebar to initialize ResNeXt-LSTM.

🔹 Step 2: Choose Analysis Mode

🖼️ Image Analysis: Upload JPG/PNG files

🎞️ Video Analysis: Upload MP4/AVI videos

📹 Webcam Mode: Real-time face stream analysis

🔹 Step 3: View Results

Output: REAL or FAKE

Confidence Level: 0–100%

Visual Indicators: 🟢 Real | 🔴 Fake

GradCAM Heatmap: Highlights manipulated regions

new featurs got added in 2.0 v

🧠 Model Architecture

ResNeXt-50 Backbone (Pretrained on ImageNet)
→ LSTM Layers for temporal video frame learning
→ Dual-Domain Features (Spatial + Frequency)
→ Sigmoid Activation for binary classification (Real/Fake)

📁 Project Structure
AEGISMIND/
│
├── app/
│   ├── streamlit_app_new.py       # Streamlit main UI
│   └── assets/images/             # Logos & UI images
│
├── models/
│   ├── resnext_lstm.py            # Core ResNeXt-LSTM architecture
│   ├── gradcam_utils.py           # GradCAM visualization utilities
│   ├── image_classifier.py        # Image classification model
│   └── pretrained/                # Model weights (.pt files)
│
├── datasets/
│   ├── image_dataset.py           # Dataset loader class
│   └── (train/val/test folders)   # Organized datasets
│
├── utils/
│   └── preprocessing.py           # Preprocessing and helper functions
│
├── realtime/                      # Webcam & live feed modules
│
├── weights/                       # Model checkpoints
│
├── train_image_classifier.py      # Image model training script
├── train_classifier.py            # General model training
├── inference_corrected.py         # Final inference logic (fixed version)
├── eval.py                        # Model evaluation script
├── extract_frames.py              # Frame extraction for video input
├── model_audit.py                 # Model accuracy audit
├── optimal_threshold.py           # Threshold calibration
├── requirements.txt               # Dependencies
└── README.md                      # This documentation

📦 Training Setup (Manual Mode)
🔹 Folder Structure
/project_root/
│
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/


📏 Split rule: 70% Train | 20% Validation | 10% Test
following 10 //  20 // 30 /// rule 

🎓 Dataset Sources

Download from Kaggle
:

Dataset	Type	Description
DeepFake Detection Challenge (DFDC)	Video	Real + Fake videos
Celeb-DF v2	Video	High-quality benchmark dataset
FaceForensics++	Video/Image	Standard dataset for deepfake research
DFDC Preview	Image/Video	Lightweight DFDC version for quick tests

🔧 Command:
kaggle datasets download -d <dataset-name>


(Requires Kaggle CLI and API key authentication)

🏋️ Training Commands
🔹 Quick Training:
python train_image_classifier.py --epochs 5 --batch-size 16

🔹 Full Training:
python train_image_classifier.py


Ensure data_dir and val_dir paths are updated:

data_dir = "data/train"
val_dir = "data/val"

🧪 Threshold Optimization

If your model misclassifies all media as fake or real:

Run the threshold tuner:

python optimal_threshold.py


Adjust in inference_corrected.py:

if prob > 0.5:
    label = "Real"
else:
    label = "Fake"


Re-test to validate corrected logic.

🔒 Security & Ethics

✅ SHA256 file integrity verification
✅ No permanent storage of uploaded media
✅ Research disclaimer included
⚠️ Add user consent for webcam access
⚠️ Include false-positive/negative disclaimers in UI

🧩 Known Improvements 

 Upgrade Streamlit APIs (use_container_width → width='stretch')

 Add model caching (@st.cache_resource)

 Enhance GradCAM normalization

 Improve dataset diversity

 UI enhancements for live visualization
 
###### Summary
Category	Score	Status
UI Readiness	80%	Functional but improvable
Backend Readiness	90%	Stable, well-structured
Data Readiness	95%	Properly organized
Security	70%	Minor UI consent missing
Overall	85%	-Ready 🚀
training 13 % on a logical code 

❤️ Built With Passion by

🧑‍💻 Satya Bhargav 
///////////
# (grey Hatter)

“They said it couldn’t be done — we proved otherwise.”
## “NOTHING IS MPOSSABLE THE WORD IT SELF SAYS IM`POSSA


📜 License

For research and educational purposes.
Use responsibly. Cite “AEGISMIND” if used academically.