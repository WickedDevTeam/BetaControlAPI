# Model Files for BetaCensor2

This directory contains all the necessary model files for BetaCensor2 to function properly.

## Directory Structure

```
models_repo/
├── nudenet/
│   ├── classifier_model.onnx
│   └── detector_v2_default_checkpoint.onnx
├── dlib/
│   └── shape_predictor_68_face_landmarks.dat
└── openpose/
    ├── pose/coco/
    ├── face/
    └── hand/
```

## Model Details

### NudeNet Models
- `classifier_model.onnx`: Used for classifying NSFW content
- `detector_v2_default_checkpoint.onnx`: Used for detecting specific body parts and regions

### dlib Model
- `shape_predictor_68_face_landmarks.dat`: Used for facial feature detection and landmark identification

### OpenPose Models (Optional)
Directory structure is prepared for OpenPose models if needed:
- `pose/coco/`: For body pose detection
- `face/`: For detailed facial landmark detection
- `hand/`: For hand pose detection

## Model Sources
- NudeNet models: https://github.com/notAI-tech/NudeNet
- dlib model: http://dlib.net/files/
- OpenPose models: https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Usage
These models are automatically used by the setup script. No manual intervention is required. 