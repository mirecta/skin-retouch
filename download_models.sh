#!/bin/sh
# Optional models for AI face masks (the plug-in works without them).
set -e
cd "$(dirname "$0")"
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/bisenet_resnet18.onnx \
  https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx
