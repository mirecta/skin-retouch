# skinretouch

Clean-room, Portraiture-style skin smoothing (learning project), in Rust.

- `crates/skinretouch` – library (pure Rust image processing, optional ONNX models)
- `crates/skinretouch-cli` – command line tool `skinretouch`
- `crates/skinretouch-app` – desktop app with live preview (egui)
- `python/` – original Python/OpenCV prototype and GIMP 3 plug-in (reference)

## Build

```
./download_models.sh          # optional AI models into models/ (~53 MB)
cargo build --release
```
ONNX Runtime is downloaded and linked statically by the `ort` crate.
Without AI: `cargo build --release --no-default-features` (colour mask only).

## App
```
target/release/skinretouch-app [photo.jpg]
```
Open / drop a photo, tune the sliders on a downscaled preview, Export renders
the full resolution. View: Split (drag the line) / After / Before / Mask
(red = not retouched). Hold Space = before, wheel = zoom, drag = pan,
double-click = fit / 1:1.

## CLI
```
skinretouch photo.jpg --healing 40 --compare --save-mask
skinretouch photo.jpg --healing 40 --moles 60     # also remove moles
skinretouch photo.jpg --fine 80 --medium 50 --large 20 --threshold 30 -o out.png
skinretouch --help
```

Models are looked up in `$SKINRETOUCH_MODELS`, `./models` and `models/` next to
(or up to 3 levels above) the executable.

## How it works
1. RGB → YCrCb; texture is smoothed on luma (Y).
2. Laplacian pyramid of Y (2–6 bands, depth from the face / image size).
3. Each band: `gain = 1 - strength * exp(-(d/T)^2)`. Small-amplitude detail
   (skin texture) is removed, large-amplitude detail (edges) is kept.
   Fine / Medium / Large set the strength per band, Threshold sets T.
4. Edge protection: estimated step height > 0.15–0.30 keeps the original pixels.
5. Chroma smoothing with a (fast) guided filter (colour uniformity).
6. Skin Healing: scale-normalized LoG blob detection finds round spots up to
   face/25 px. Like healing on a frequency-separated image, only the low
   frequencies of a spot are replaced by the mean of the skin around it
   (spots left out of the mean), so brightness and pore texture are kept.
   Only spots that stand out from their surroundings are touched; lines
   (lips, brows), the nose and everything outside the skin mask are skipped.
   Spots are searched in luma (dark/bright) and in redness (Cr: red dots,
   marks after pimples). A second pass, "Moles", removes bigger dark spots
   (up to face/12: moles, birthmarks) - separate because they are often
   wanted. Big faces are analysed on a shrunken copy (the correction is smooth).
7. Masks: soft skin-tone mask (CrCb distance, latitude, fuzziness, feather).
   With models: YuNet face detection → BiSeNet face parsing on a crop around
   each face (skin, ears, nose, neck = retouch; eyes, brows, lips, hair = keep),
   refined by a guided filter. Face size also sets "Portrait Size".
   YuNet ships with a fixed 640×640 input; the loader rewrites the ONNX
   graph in memory to dynamic input size (like OpenCV does).
8. Enhancements: sharpness, softness/glow, warmth, tint, brightness, contrast.

## Models (not in git)
- `face_detection_yunet_2023mar.onnx` (230 kB) - https://github.com/opencv/opencv_zoo
- `bisenet_resnet18.onnx` (53 MB, MIT) - https://github.com/yakhyo/face-parsing/releases

## Tests
`cargo test --release` (synthetic skin: texture removed, edges kept, blemishes healed).

## Ideas / TODO
- Own bilateral grid (what Portraiture uses) instead of the band gain.
- GIMP / PhotoCraft plug-in on top of the library.
- GPU (ONNX Runtime CUDA for the models, compute shaders for the filters).
