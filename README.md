# Mood Detection Project

This project detects human moods using:
1. **Facial Emotion Recognition**: Detects emotions from a camera feed.
2. **NLP-based Mood Analysis** (Planned): Uses text analysis to determine mood via NLP. This part may require Wi-Fi to connect to a phone or cloud-based LLMs.

## Features
- Real-time facial emotion recognition using a pre-trained model.
- Future integration with text-based mood analysis (NLP).

## Hardware
- Board with 1 TOPS processing power (e.g., Raspberry Pi, Jetson Nano).
- Camera (USB or CSI).
- Optional: Wi-Fi for external NLP processing.

## Software
- **OS**: Linux-based or Windows.
- **Python**: 3.6+
- **Libraries**: `opencv-python`, `tensorflow`, `numpy`
- **Models**:
  - [Haar Cascade XML](https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml)
  - [Pre-trained Emotion Model](https://github.com/atulapra/Emotion-detection/raw/master/model.h5)

## Setup

....
