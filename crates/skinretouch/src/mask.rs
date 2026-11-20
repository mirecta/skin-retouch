//! Skin masks.
//!
//! * colour mask: soft distance from a skin-tone centre in the CrCb plane
//!   (Portraiture's Latitude / Fuzziness / Feathering);
//! * face boxes (from a detector) sample the skin tone from the face itself
//!   and can restrict work to the people in the photo;
//! * face parsing maps (from a segmentation network) replace the colour mask
//!   inside the parsed face crops.

use crate::color::to_ycc;
use crate::ops;
use crate::plane::{Plane, Rgb};

/// Typical skin centre in YCrCb (0..1 scale) across many skin tones.
pub const DEFAULT_CR: f32 = 150.0 / 255.0;
pub const DEFAULT_CB: f32 = 110.0 / 255.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskParams {
    /// 0..100 how wide the accepted colour range is.
    pub latitude: f32,
    /// 0..100 softness of the colour range edge.
    pub fuzziness: f32,
    /// 0..100 spatial blur of the mask.
    pub feather: f32,
    /// Ignore very dark pixels (hair, shadows).
    pub min_luma: f32,
    /// Ignore blown highlights.
    pub max_luma: f32,
}

impl Default for MaskParams {
    fn default() -> Self {
        Self {
            latitude: 50.0,
            fuzziness: 50.0,
            feather: 30.0,
            min_luma: 0.12,
            max_luma: 0.98,
        }
    }
}

/// Face bounding box in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
    pub score: f32,
}

impl Face {
    pub fn size(&self) -> f32 {
        self.w.max(self.h) as f32
    }

    pub fn scaled(&self, s: f32) -> Face {
        let f = |v: i64| (v as f32 * s).round() as i64;
        Face {
            x: f(self.x),
            y: f(self.y),
            w: f(self.w),
            h: f(self.h),
            score: self.score,
        }
    }
}

/// Output of face parsing, full image size.
#[derive(Clone, Debug)]
pub struct ParseMaps {
    /// Skin probability (skin, ears, nose, neck), feathered at crop borders.
    pub skin: Plane,
    /// 1 inside parsed crops (feathered at their border), 0 elsewhere.
    pub cover: Plane,
    /// Nose probability (Skin Healing skips nostrils and nose shading).
    pub nose: Plane,
    /// Eyes, brows, mouth and lips probability (kept crisp, never healed).
    pub features: Plane,
}

/// Median of a slice (mutates order).
fn median(v: &mut [f32]) -> f32 {
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// Robust (median) skin tone from the central cheek/forehead area of each face.
pub fn sample_skin_tone(ycc: &[Plane; 3], faces: &[Face]) -> (f32, f32) {
    let (w, h) = (ycc[0].w as i64, ycc[0].h as i64);
    let (mut cr, mut cb) = (Vec::new(), Vec::new());
    for f in faces {
        let (x0, y0) = ((f.x + f.w / 4).max(0), (f.y + f.h / 4).max(0));
        let (x1, y1) = (
            (f.x + 3 * f.w / 4).min(w - 1),
            (f.y + 3 * f.h / 4).min(h - 1),
        );
        for y in y0..=y1 {
            for x in x0..=x1 {
                cr.push(ycc[1].get(x as usize, y as usize));
                cb.push(ycc[2].get(x as usize, y as usize));
            }
        }
    }
    if cr.len() < 50 {
        return (DEFAULT_CR, DEFAULT_CB);
    }
    (median(&mut cr), median(&mut cb))
}

pub fn color_mask(ycc: &[Plane; 3], mp: &MaskParams, centre: (f32, f32)) -> Plane {
    let (cr0, cb0) = centre;
    let radius = 0.02 + 0.10 * (mp.latitude / 100.0);
    let soft = 0.005 + 0.08 * (mp.fuzziness / 100.0);
    let (lo, hi) = (mp.min_luma, mp.max_luma);
    let m = ycc[0].zip3(&ycc[1], &ycc[2], |y, cr, cb| {
        let d = ((cr - cr0).powi(2) + (cb - cb0).powi(2)).sqrt();
        let m = ((radius + soft - d) / (2.0 * soft)).clamp(0.0, 1.0);
        m * ((y - lo) / 0.05).clamp(0.0, 1.0) * ((hi - y) / 0.02).clamp(0.0, 1.0)
    });
    if mp.feather > 0.0 {
        let sigma = (m.w.min(m.h) as f32 * 0.01 * mp.feather / 100.0).max(0.5);
        return ops::gaussian_blur(&m, sigma);
    }
    m
}

/// Soft elliptical region around detected faces (covers neck and ears too).
pub fn face_region(w: usize, h: usize, faces: &[Face], grow: f32) -> Plane {
    let mut m = Plane::zeros(w, h);
    for f in faces {
        let (cx, cy) = ((f.x + f.w / 2) as f32, (f.y + f.h / 2) as f32);
        let (ax, ay) = (
            (f.w as f32 * grow / 2.0).trunc(),
            (f.h as f32 * grow / 2.0 * 1.15).trunc(),
        );
        ops::fill_ellipse(&mut m, cx, cy, ax, ay, 1.0);
    }
    if faces.is_empty() {
        return m;
    }
    ops::gaussian_blur(&m, (w.min(h) as f32 * 0.02).max(2.0))
}

/// Masks for one image: where to retouch and where Skin Healing may work.
pub struct Masks {
    pub retouch: Plane,
    pub heal: Plane,
}

/// Build the retouch and healing masks.
pub fn build_mask(
    rgb: &Rgb,
    mp: &MaskParams,
    faces: &[Face],
    parse: Option<&ParseMaps>,
    faces_only: bool,
) -> Masks {
    let ycc = to_ycc(rgb);
    let centre = if faces.is_empty() {
        (DEFAULT_CR, DEFAULT_CB)
    } else {
        sample_skin_tone(&ycc, faces)
    };
    let mut m = color_mask(&ycc, mp, centre);
    if !faces.is_empty() && faces_only {
        m = m.zip(&face_region(rgb.w(), rgb.h(), faces, 1.6), |a, b| a * b);
    }
    let Some(parse) = parse.filter(|_| !faces.is_empty()) else {
        return Masks {
            heal: m.clone(),
            retouch: m,
        };
    };
    // edge-aware refinement snaps the coarse network mask to image edges
    let mut sizes: Vec<f32> = faces.iter().map(Face::size).collect();
    let r = ((median(&mut sizes) / 80.0) as usize).max(4);
    let mut skin =
        ops::guided_filter_color(&rgb.c, &parse.skin, r, 1e-4).map(|v| v.clamp(0.0, 1.0));
    if mp.feather > 0.0 {
        skin = ops::gaussian_blur(
            &skin,
            (rgb.w().min(rgb.h()) as f32 * 0.004 * mp.feather / 100.0).max(0.5),
        );
    }
    // inside parsed face crops trust the network, outside keep the colour mask
    let m = skin.zip3(&parse.cover, &m, |s, c, m| s * c + m * (1.0 - c));
    // The network marks only the eyeball as "eye"; lashes, eyeliner and the
    // lid crease count as skin. Keep a thin band around the features crisp,
    // and keep healing (which looks for dark round spots) well away from them.
    let face = median(&mut sizes);
    let near = |sigma: f32, gain: f32| {
        ops::gaussian_blur(&parse.features, sigma).map(|v| (v * gain).clamp(0.0, 1.0))
    };
    let (thin, wide) = rayon::join(|| near(face / 150.0, 2.5), || near(face / 30.0, 4.0));
    let m = m.zip(&thin, |m, t| m * (1.0 - t));
    let nose = ops::dilate(&parse.nose, &ops::rect_se(7)); // = 3x3, 3 iterations
    let heal = m.zip3(&nose, &wide, |m, n, f| {
        m * (1.0 - (n * 2.0).clamp(0.0, 1.0)) * (1.0 - f)
    });
    Masks { retouch: m, heal }
}
