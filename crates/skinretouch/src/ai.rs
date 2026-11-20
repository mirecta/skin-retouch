//! Neural face masks through ONNX Runtime.
//!
//! * [`FaceDetector`]: YuNet (OpenCV Zoo, 230 kB) - face boxes.
//! * [`FaceParser`]: BiSeNet trained on CelebAMask-HQ (19 classes, MIT,
//!   github.com/yakhyo/face-parsing) - which pixels are skin.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use crate::mask::{Face, ParseMaps};
use crate::ops;
use crate::plane::{Plane, Rgb};

pub const YUNET_FILE: &str = "face_detection_yunet_2023mar.onnx";
pub const BISENET_FILE: &str = "bisenet_resnet18.onnx";

fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let env = ort::init()
        .with_name("skinretouch")
        .build()
        .map_err(|e| anyhow!("onnxruntime: {e}"))?;
    Ok(ENV.get_or_init(|| env))
}

fn builder() -> Result<ort::session::builder::SessionBuilder> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    Session::builder(environment()?)
        .map_err(|e| anyhow!("onnxruntime: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("onnxruntime: {e}"))
}

fn load(path: &Path) -> Result<Session> {
    builder()?
        .commit_from_file(path)
        .map_err(|e| anyhow!("cannot load model {}: {e}", path.display()))
}

fn load_bytes(bytes: &[u8]) -> Result<Session> {
    builder()?
        .commit_from_memory(bytes)
        .map_err(|e| anyhow!("cannot load model: {e}"))
}

/// Fixed input size of a model's first input, if it is not dynamic.
fn fixed_input_hw(s: &Session) -> Option<(usize, usize)> {
    let shape = s.inputs().first()?.dtype().tensor_shape()?;
    match shape[..] {
        [_, _, h, w] if h > 0 && w > 0 => Some((h as usize, w as usize)),
        _ => None,
    }
}

// ---------------------------------------------------------------- YuNet

pub struct FaceDetector {
    session: Session,
    input: String,
    fixed: Option<(usize, usize)>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
    /// Longest image side the detector works on (bigger images are shrunk).
    pub max_side: usize,
}

impl FaceDetector {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // YuNet ships with a fixed 640x640 input; OpenCV reshapes it freely,
        // ONNX Runtime needs the dimensions to be declared dynamic.
        let bytes =
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        if let Some(patched) = onnx_patch::dynamic_io(&bytes)
            && let Ok(session) = load_bytes(&patched)
        {
            let mut det = Self::with_session(session)?;
            if det.fixed.is_none() && det.detect(&Rgb::new(200, 150, [0.5; 3])).is_ok() {
                return Ok(det);
            }
        }
        Self::with_session(load(path)?)
    }

    fn with_session(session: Session) -> Result<Self> {
        let input = session
            .inputs()
            .first()
            .context("model has no input")?
            .name()
            .to_string();
        let fixed = fixed_input_hw(&session);
        Ok(Self {
            session,
            input,
            fixed,
            score_threshold: 0.7,
            nms_threshold: 0.3,
            max_side: 1024,
        })
    }

    /// Input size the network works at: None = any (dynamic), else (h, w).
    pub fn fixed_input(&self) -> Option<(usize, usize)> {
        self.fixed
    }

    pub fn detect(&mut self, rgb: &Rgb) -> Result<Vec<Face>> {
        let (w, h) = (rgb.w(), rgb.h());
        let scale = match self.fixed {
            Some((fh, fw)) => (fw as f32 / w as f32).min(fh as f32 / h as f32),
            None => (self.max_side as f32 / w.max(h) as f32).min(1.0),
        };
        let (sw, sh) = (
            ((w as f32 * scale) as usize).max(1),
            ((h as f32 * scale) as usize).max(1),
        );
        let (pw, ph) = match self.fixed {
            Some((fh, fw)) => (fw, fh),
            None => (sw.div_ceil(32) * 32, sh.div_ceil(32) * 32),
        };
        // BGR, 0..255, NCHW, zero padding at right/bottom
        let mut blob = vec![0.0f32; 3 * pw * ph];
        for (ci, src) in [2usize, 1, 0].into_iter().enumerate() {
            let small = ops::resize_linear(&rgb.c[src], sw, sh);
            let dst = &mut blob[ci * pw * ph..(ci + 1) * pw * ph];
            for y in 0..sh {
                for x in 0..sw {
                    dst[y * pw + x] = (small.get(x, y).clamp(0.0, 1.0) * 255.0).round();
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, ph, pw], blob)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .map_err(|e| anyhow!("face detection failed: {e}"))?;

        let mut cands = Vec::new();
        for stride in [8usize, 16, 32] {
            let get = |name: &str| -> Result<&[f32]> {
                let v = outputs
                    .get(format!("{name}_{stride}"))
                    .with_context(|| format!("missing output {name}_{stride}"))?;
                Ok(v.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?.1)
            };
            let (cls, obj, bbox) = (get("cls")?, get("obj")?, get("bbox")?);
            let cols = pw / stride;
            let s = stride as f32;
            for (i, (&c, &o)) in cls.iter().zip(obj).enumerate() {
                let score = (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();
                if score < self.score_threshold {
                    continue;
                }
                let (r, col) = ((i / cols) as f32, (i % cols) as f32);
                let b = &bbox[i * 4..i * 4 + 4];
                let (cx, cy) = ((col + b[0]) * s, (r + b[1]) * s);
                let (bw, bh) = (b[2].exp() * s, b[3].exp() * s);
                cands.push((score, [cx - bw / 2.0, cy - bh / 2.0, bw, bh]));
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut keep: Vec<(f32, [f32; 4])> = Vec::new();
        for c in cands {
            if keep.iter().all(|k| iou(&k.1, &c.1) <= self.nms_threshold) {
                keep.push(c);
            }
        }
        Ok(keep
            .into_iter()
            .map(|(score, b)| Face {
                x: (b[0] / scale) as i64,
                y: (b[1] / scale) as i64,
                w: (b[2] / scale) as i64,
                h: (b[3] / scale) as i64,
                score,
            })
            .collect())
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = (a[0] + a[2]).min(b[0] + b[2]);
    let y1 = (a[1] + a[3]).min(b[1] + b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

// ---------------------------------------------------------------- BiSeNet

/// CelebAMask-HQ classes.
pub const CLASSES: [&str; 19] = [
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
];
/// Classes that get retouched: skin, ears, nose, neck.
pub const SKIN: [usize; 5] = [1, 7, 8, 10, 14];
pub const NOSE: usize = 10;
/// Facial features that must stay crisp: brows, eyes, mouth, lips.
pub const FEATURES: [usize; 7] = [2, 3, 4, 5, 11, 12, 13];
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct FaceParser {
    session: Session,
    input: String,
    output: String,
    size: usize,
    /// Crop side relative to the face size.
    pub expand: f32,
}

impl FaceParser {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let session = load(path.as_ref())?;
        let input = session
            .inputs()
            .first()
            .context("model has no input")?
            .name()
            .to_string();
        let output = if session.outputs().iter().any(|o| o.name() == "output") {
            "output".to_string()
        } else {
            session
                .outputs()
                .first()
                .context("model has no output")?
                .name()
                .to_string()
        };
        let size = fixed_input_hw(&session).map_or(512, |(h, _)| h);
        Ok(Self {
            session,
            input,
            output,
            size,
            expand: 2.0,
        })
    }

    /// Square crop (x, y, side) around a face; may extend past the image.
    pub fn crop_box(&self, f: &Face) -> (i64, i64, usize) {
        let side = (f.w.max(f.h) as f32 * self.expand) as i64;
        let (cx, cy) = (f.x + f.w / 2, f.y + f.h / 2 - (f.h as f32 * 0.05) as i64);
        (cx - side / 2, cy - side / 2, side.max(1) as usize)
    }

    /// Class probabilities (19 planes of size x size) for a crop box.
    pub fn probs(&mut self, rgb: &Rgb, (x, y, s): (i64, i64, usize)) -> Result<Vec<Plane>> {
        let n = self.size;
        let (w, h) = (rgb.w() as i64, rgb.h() as i64);
        let mut blob = Vec::with_capacity(3 * n * n);
        for c in 0..3 {
            let mut crop = Plane::zeros(s, s);
            for cy in y.max(0)..(y + s as i64).min(h) {
                for cx in x.max(0)..(x + s as i64).min(w) {
                    crop.set(
                        (cx - x) as usize,
                        (cy - y) as usize,
                        rgb.c[c].get(cx as usize, cy as usize),
                    );
                }
            }
            let small = ops::resize_area(&crop, n, n);
            blob.extend(small.data.iter().map(|v| (v - MEAN[c]) / STD[c]));
        }
        let tensor = Tensor::from_array(([1usize, 3, n, n], blob)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input.as_str() => tensor])
            .map_err(|e| anyhow!("face parsing failed: {e}"))?;
        let (_, logits) = outputs[self.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        let k = logits.len() / (n * n);
        // softmax over classes, per pixel
        let mut out: Vec<Plane> = (0..k).map(|_| Plane::zeros(n, n)).collect();
        let probs: Vec<Vec<f32>> = (0..n * n)
            .into_par_iter()
            .map(|i| {
                let m = (0..k)
                    .map(|c| logits[c * n * n + i])
                    .fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = (0..k).map(|c| (logits[c * n * n + i] - m).exp()).collect();
                let s: f32 = e.iter().sum();
                e.into_iter().map(|v| v / s).collect()
            })
            .collect();
        for (i, p) in probs.iter().enumerate() {
            for c in 0..k {
                out[c].data[i] = p[c];
            }
        }
        Ok(out)
    }

    /// Skin / coverage / nose / feature maps for the whole image.
    pub fn parse(&mut self, rgb: &Rgb, faces: &[Face]) -> Result<ParseMaps> {
        let (w, h) = (rgb.w(), rgb.h());
        let mut maps = ParseMaps {
            skin: Plane::zeros(w, h),
            cover: Plane::zeros(w, h),
            nose: Plane::zeros(w, h),
            features: Plane::zeros(w, h),
        };
        for f in faces {
            let bx @ (x, y, s) = self.crop_box(f);
            let p = self.probs(rgb, bx)?;
            let n = self.size;
            let skin = Plane::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| {
                        ((SKIN.iter().map(|&c| p[c].data[i]).sum::<f32>() - 0.2) / 0.5)
                            .clamp(0.0, 1.0)
                    })
                    .collect(),
            );
            let skin = ops::resize_linear(&skin, s, s);
            let nose = ops::resize_linear(&p[NOSE], s, s);
            let feat = Plane::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| FEATURES.iter().map(|&c| p[c].data[i]).sum::<f32>().min(1.0))
                    .collect(),
            );
            let feat = ops::resize_linear(&feat, s, s);
            // feather the crop border so it blends with the colour mask outside
            let ramp = |i: usize| (i.min(s - 1 - i) as f32 / (s as f32 * 0.08)).min(1.0);
            for yy in y.max(0)..(y + s as i64).min(h as i64) {
                let (ly, iy) = ((yy - y) as usize, yy as usize);
                for xx in x.max(0)..(x + s as i64).min(w as i64) {
                    let (lx, ix) = ((xx - x) as usize, xx as usize);
                    let border = ramp(lx).min(ramp(ly));
                    let o = iy * w + ix;
                    let l = ly * s + lx;
                    maps.skin.data[o] = maps.skin.data[o].max(skin.data[l] * border);
                    maps.cover.data[o] = maps.cover.data[o].max(border);
                    maps.nose.data[o] = maps.nose.data[o].max(nose.data[l]);
                    maps.features.data[o] = maps.features.data[o].max(feat.data[l]);
                }
            }
        }
        Ok(maps)
    }
}

/// Default places to look for model files: `$SKINRETOUCH_MODELS`,
/// `./models`, `models/` next to the executable and up to 3 levels above it.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("SKINRETOUCH_MODELS") {
        v.push(PathBuf::from(d));
    }
    v.push(PathBuf::from("models"));
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1).take(4) {
            v.push(dir.join("models"));
        }
    }
    v
}

pub fn find_model(file: &str) -> Option<PathBuf> {
    model_dirs()
        .into_iter()
        .map(|d| d.join(file))
        .find(|p| p.is_file())
}

/// Minimal protobuf rewriting of an ONNX model (no onnx crate needed).
mod onnx_patch {
    fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *b.get(*i)?;
            *i += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    /// A raw field: number, wire type, encoded bytes (incl. key), payload.
    struct Field<'a> {
        num: u64,
        wire: u8,
        raw: &'a [u8],
        payload: &'a [u8],
    }

    fn fields(b: &[u8]) -> Option<Vec<Field<'_>>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let start = i;
            let key = varint(b, &mut i)?;
            let (num, wire) = (key >> 3, (key & 7) as u8);
            let pstart;
            match wire {
                0 => {
                    pstart = i;
                    varint(b, &mut i)?;
                }
                1 => {
                    pstart = i;
                    i += 8;
                }
                2 => {
                    let len = varint(b, &mut i)? as usize;
                    pstart = i;
                    i += len;
                }
                5 => {
                    pstart = i;
                    i += 4;
                }
                _ => return None,
            }
            if i > b.len() {
                return None;
            }
            out.push(Field {
                num,
                wire,
                raw: &b[start..i],
                payload: &b[pstart..i],
            });
        }
        Some(out)
    }

    fn put_bytes(out: &mut Vec<u8>, num: u64, payload: &[u8]) {
        put_varint(out, (num << 3) | 2);
        put_varint(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }

    /// Field rewriter: outer None = parse error, inner None = drop the field.
    type Rewriter<'f> = dyn FnMut(&[u8]) -> Option<Option<Vec<u8>>> + 'f;

    /// Re-encode message `b`, replacing length-delimited field `num` by `f(payload)`.
    fn rewrite(b: &[u8], num: u64, f: &mut Rewriter) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(b.len());
        for fl in fields(b)? {
            if fl.num == num && fl.wire == 2 {
                if let Some(p) = f(fl.payload)? {
                    put_bytes(&mut out, num, &p);
                }
            } else {
                out.extend_from_slice(fl.raw);
            }
        }
        Some(out)
    }

    /// Dimension { dim_value = 1 (varint) | dim_param = 2 (string) }:
    /// spatial sizes (>= 16) become named dynamic dimensions.
    fn dims_dynamic(shape: &[u8], counter: &mut u32) -> Option<Vec<u8>> {
        rewrite(shape, 1, &mut |dim| {
            let fl = fields(dim)?;
            let value = fl
                .iter()
                .find(|f| f.num == 1 && f.wire == 0)
                .and_then(|f| varint(f.payload, &mut 0));
            match value {
                Some(v) if v >= 16 => {
                    *counter += 1;
                    let mut d = Vec::new();
                    put_bytes(&mut d, 2, format!("d{counter}").as_bytes());
                    Some(Some(d))
                }
                _ => Some(Some(dim.to_vec())),
            }
        })
    }

    /// ValueInfoProto.type(2) -> TypeProto.tensor_type(1) -> shape(2)
    fn value_info_dynamic(vi: &[u8], counter: &mut u32) -> Option<Vec<u8>> {
        rewrite(vi, 2, &mut |ty| {
            Some(Some(rewrite(ty, 1, &mut |tt| {
                Some(Some(rewrite(tt, 2, &mut |sh| {
                    Some(Some(dims_dynamic(sh, counter)?))
                })?))
            })?))
        })
    }

    /// ModelProto.graph(7): make inputs(11)/outputs(12) dynamic, drop value_info(13).
    pub fn dynamic_io(model: &[u8]) -> Option<Vec<u8>> {
        let mut counter = 0u32;
        rewrite(model, 7, &mut |graph| {
            let mut out = Vec::with_capacity(graph.len());
            for fl in fields(graph)? {
                match (fl.num, fl.wire) {
                    (11 | 12, 2) => put_bytes(
                        &mut out,
                        fl.num,
                        &value_info_dynamic(fl.payload, &mut counter)?,
                    ),
                    (13, 2) => {}
                    _ => out.extend_from_slice(fl.raw),
                }
            }
            Some(Some(out))
        })
    }
}
