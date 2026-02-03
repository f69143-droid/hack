# Gender-Based Participant Counting System for SDG 5 (Gender Equality)

## Problem Statement
Gender Equality (SDG 5) requires visible measurement of participation across programs and events. This project builds an AI system that detects participants in images, classifies gender presentation, and produces transparent, aggregate counts with visual analytics to help monitor representation responsibly.

## Architecture (High-Level)

```
+-------------------+       +----------------------+       +-------------------------+
|  Image/Webcam     |  -->  |  Face Detection      |  -->  | Gender Classification   |
|  Streamlit Input  |       |  (MediaPipe)         |       | (PyTorch ResNet18)      |
+-------------------+       +----------------------+       +-------------------------+
           |                                      \                |
           |                                       \               v
           |                                        +--> Analytics + Reports
           v
     Streamlit UI (Charts, Cards, Ethics)
```

## Tech Stack
- **Python**
- **Computer Vision:** OpenCV + MediaPipe
- **ML/DL:** PyTorch
- **UI:** Streamlit
- **Visualization:** Plotly

## How the Model Works (Step-by-Step)
1. **Input**: Upload an image or take a webcam snapshot.
2. **Detection**: MediaPipe Face Detection locates faces and returns bounding boxes + confidence.
3. **Preprocessing**: Each face crop is resized and normalized.
4. **Classification**: A ResNet18 model (fine-tuned on a public gender dataset) predicts Male/Female.
5. **Counting & Analytics**: Counts and percentages are computed and visualized.
6. **Ethics**: Clear disclaimer about limitations and SDG 5 context is shown in-app.

## Dataset
Use a public gender classification dataset such as **FairFace** or **Adience**. A training script is provided to fine-tune the ResNet18 model.

## Ethical Considerations
- Gender inferred visually may not match self-identified gender.
- Dataset bias can impact performance across demographics.
- Use outputs only for aggregate insights with human oversight.

## How to Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional: Train a Better Gender Model
```bash
python model/train_model.py --data-dir /path/to/gender_dataset --epochs 5
```

## Future Improvements
- Integrate inclusive, multi-label demographic modeling (with consent).
- Add blur/occlusion handling and better low-light preprocessing.
- Add model cards and calibration metrics for transparency.
- Deploy a privacy-preserving edge inference pipeline.

## Sample Screenshots (text description)
- Upload image view with bounding boxes and gender labels.
- Result cards showing Total/Male/Female counts with FPS.
- Bar + pie charts reflecting gender distribution.
- Ethics section with SDG 5 alignment and bias disclaimer.
