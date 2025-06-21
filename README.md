# Real-Time Facial Emotion Detection 😃😡😢

This project uses deep learning and computer vision to detect human emotions in real-time through webcam input. It applies **transfer learning** on a **MobileNetV2** backbone and is trained on **FER2013** and **CK+** datasets.

## 🚀 Features
- Real-time facial emotion recognition from webcam
- Lightweight MobileNetV2 model optimized for speed
- Supports 6 emotions: Happy, Sad, Angry, Disgust, Surprised, Neutral
- Future scope: Fusion with biometric data for emotion authenticity

## 📁 Project Structure
- `/notebooks`: Jupyter notebooks for model training and experimentation
- `/models`: Trained model files (to be uploaded)
- `/datasets`: Directory for FER2013 and CK+ (not included due to size)
- `/utils`: Helper functions (e.g., face detection, preprocessing)

## 🧠 Model Details
- Base Model: `MobileNetV2(weights='imagenet', include_top=False)`
- Custom classification head for 6 emotion classes
- Trained with image augmentation and early stopping

## 📹 Demo (Coming Soon)

## 📌 Status
This project is **in progress** — training and evaluation ongoing. Final results and deployment will be shared soon.

## 🔗 Future Enhancements
- Add biometric data integration (e.g., heart rate)
- Optimize for mobile/edge deployment
