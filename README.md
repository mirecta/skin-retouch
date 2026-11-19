# skinretouch

Clean-room, Portraiture-style skin smoothing (learning project).

## How it works
1. RGB → YCrCb; texture is smoothed on luma (Y).
2. Laplacian pyramid of Y (2–6 bands, depth from image / portrait size).
3. Each band: `gain = 1 - strength * exp(-(d/T)^2)`. Small-amplitude detail
   (skin texture) is removed, large-amplitude detail (edges) is kept.
   Fine / Medium / Large sliders set the strength per band, Threshold sets T.
4. Edge protection: estimated step height > 0.15–0.30 keeps the original pixels.
5. Chroma smoothing with a guided filter (colour uniformity).
6. Skin Healing: scale-normalized LoG blob detection finds round spots
   (blemishes) of up to face/25 px, each replaced by grey closing+opening of
   luma and the surrounding average chroma. Lines (lips, brows) are not round;
   nose and everything near eyes/mouth are excluded.
7. Masks: soft skin-tone mask (CrCb distance, latitude, fuzziness, feather).
   With models: YuNet face detection -> BiSeNet face parsing on a crop around
   each face (skin, ears, nose, neck = retouch; eyes, brows, lips, hair = keep),
   refined by a guided filter. Face size also sets 'Portrait Size'.
8. Enhancements: sharpness, softness/glow, warmth, tint, brightness, contrast.

## Models (in `models/`, not required; run `./download_models.sh`)
- `face_detection_yunet_2023mar.onnx` (230 kB) - https://github.com/opencv/opencv_zoo
- `bisenet_resnet18.onnx` (53 MB, MIT) - https://github.com/yakhyo/face-parsing/releases

## CLI
```
python3 cli.py photo.jpg --compare --save-mask
python3 cli.py photo.jpg --fine 80 --medium 50 --large 20 --threshold 30
python3 cli.py photo.jpg --healing 40 \
    --face-model models/face_detection_yunet_2023mar.onnx \
    --parse-model models/bisenet_resnet18.onnx
```

## GIMP 3
```
ln -s "$PWD/gimp/skinretouch" ~/.config/GIMP/3.0/plug-ins/skinretouch
```
Filters → Enhance → Skin Retouch…  (needs numpy + opencv in system Python).
AI masks are used automatically when the models are in `models/`.

## Tests
`python3 tests/test_core.py`

## Ideas / TODO
- Own bilateral grid implementation (what Portraiture uses) instead of the band gain.
- Live preview in the GIMP dialog.
