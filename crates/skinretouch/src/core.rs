//! Core skin-smoothing algorithm (Portraiture-style, clean-room).
//!
//! 1. Convert to YCrCb. Texture lives mostly in luma (Y).
//! 2. Build a Laplacian pyramid of Y. Each band holds details of one size:
//!    band 0-1 = fine (pores, noise), band 2 = medium (blemishes),
//!    band 3+ = large (uneven skin, shadows).
//! 3. Attenuate each band with an amplitude-dependent gain
//!    `gain(d) = 1 - strength * exp(-(d / T)^2)`: small-amplitude detail
//!    (skin texture) is removed, large-amplitude detail (real edges) passes.
//! 4. Reconstruct, smooth the chroma a little (colour uniformity), then blend
//!    with the original through the skin mask.

use crate::bilateral;
use crate::color::{from_ycc, to_ycc};
use crate::ops;
use crate::plane::{Plane, Rgb};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetouchParams {
    /// 0..100 smoothing of the finest detail (pores, noise).
    pub fine: f32,
    /// 0..100 smoothing of medium detail.
    pub medium: f32,
    /// 0..100 smoothing of large detail (uneven skin).
    pub large: f32,
    /// 0..100 amplitude regarded as "skin texture".
    pub threshold: f32,
    /// Face height in pixels; 0 = auto (from detected faces or image size).
    pub portrait_size: f32,
    /// 0..100 chroma smoothing.
    pub uniformity: f32,
    /// 0..100 extra protection of strong edges.
    pub edge_protect: f32,
    /// 0..100 final blend with the original.
    pub amount: f32,
    /// 0..100 removal of small round spots (blemishes, red dots).
    pub healing: f32,
    /// 0..100 removal of bigger dark spots (moles, birthmarks).
    pub moles: f32,
}

impl Default for RetouchParams {
    fn default() -> Self {
        Self {
            fine: 60.0,
            medium: 40.0,
            large: 15.0,
            threshold: 20.0,
            portrait_size: 0.0,
            uniformity: 20.0,
            edge_protect: 50.0,
            amount: 100.0,
            healing: 0.0,
            moles: 0.0,
        }
    }
}

/// Reference size used when no face size is known.
pub fn ref_size(w: usize, h: usize, portrait_size: f32) -> f32 {
    if portrait_size > 0.0 {
        portrait_size
    } else {
        w.min(h) as f32 * 0.6
    }
}

/// Pyramid depth so that the coarsest band (~ref/16 px) matches large skin structures.
pub fn num_levels(w: usize, h: usize, portrait_size: f32) -> usize {
    let r = ref_size(w, h, portrait_size).max(32.0);
    (r / 16.0).log2().round().clamp(2.0, 6.0) as usize
}

pub fn laplacian_pyramid(y: &Plane, levels: usize) -> Vec<Plane> {
    let mut bands = Vec::with_capacity(levels + 1);
    let mut cur = y.clone();
    for _ in 0..levels {
        let down = ops::pyr_down(&cur);
        let up = ops::pyr_up(&down, cur.w, cur.h);
        bands.push(cur.zip(&up, |a, b| a - b));
        cur = down;
    }
    bands.push(cur);
    bands
}

pub fn reconstruct(bands: &[Plane]) -> Plane {
    let mut cur = bands.last().expect("non-empty").clone();
    for band in bands[..bands.len() - 1].iter().rev() {
        cur = ops::pyr_up(&cur, band.w, band.h).zip(band, |a, b| a + b);
    }
    cur
}

/// Map Fine/Medium/Large sliders onto the pyramid bands (0..1 each).
pub fn band_strengths(levels: usize, p: &RetouchParams) -> Vec<f32> {
    let (f, m, l) = (p.fine / 100.0, p.medium / 100.0, p.large / 100.0);
    (0..levels)
        .map(|i| {
            let t = i as f32 / (levels.max(2) - 1) as f32;
            let s = if t <= 0.5 {
                f + (m - f) * (t / 0.5)
            } else {
                m + (l - m) * ((t - 0.5) / 0.5)
            };
            s.clamp(0.0, 1.0)
        })
        .collect()
}

/// 0..1 map of strong, coherent edges (eyes, lips, hairline, jaw).
///
/// The gradient is converted to an estimated step height in luma units, so
/// the map is absolute: low-contrast blemishes are not protected, real
/// features with contrast above `hi` are fully protected.
pub fn edge_map(y: &Plane, sigma: f32, lo: f32, hi: f32) -> Plane {
    let blur = ops::gaussian_blur(y, sigma);
    let (gx, gy) = rayon::join(
        || ops::sobel(&blur, 1, 0, 1.0),
        || ops::sobel(&blur, 0, 1, 1.0),
    );
    // Sobel 3x3 = 8x derivative; a blurred step of height h peaks at h/(2.5 sigma)
    let e = gx.zip(&gy, |a, b| {
        let step = (a * a + b * b).sqrt() / 8.0 * 2.5 * sigma;
        ((step - lo) / (hi - lo)).clamp(0.0, 1.0)
    });
    ops::gaussian_blur(&e, sigma)
}

/// Spatial sigma of the finest smoothing layer and number of layers.
///
/// Portraiture's "Portrait Size": the finest layer works at ~face/200
/// (pores), every next layer doubles, the coarsest reaches ~face/20
/// (uneven skin tone, blotches).
pub fn layer_scales(w: usize, h: usize, portrait_size: f32) -> (f32, usize) {
    let r = ref_size(w, h, portrait_size).max(32.0);
    let s0 = (r / 200.0).max(1.0);
    let levels = ((r / 20.0 / s0).log2().round() as usize + 1).clamp(2, 6);
    (s0, levels)
}

/// Range sigma of layer `i` for the Threshold slider (0..100): skin
/// texture below this contrast is smoothed, real edges above it are kept.
pub fn layer_threshold(threshold: f32, i: usize) -> f32 {
    (0.01 + threshold / 100.0 * 0.15) * (1.0 + 0.25 * i as f32)
}

/// Multi-scale bilateral decomposition: base_0 = y,
/// base_{i+1} = bilateral(base_i, s0 * 2^i, threshold_i),
/// detail_i = base_i - base_{i+1}, so y = base_n + sum(detail_i).
/// Each detail layer holds texture of one size and of low contrast;
/// edges stronger than the threshold stay in the base (no halos).
/// `strengths[i]` (0..1) says how much of detail layer i is removed.
pub fn smooth_layers(y: &Plane, s0: f32, strengths: &[f32], threshold: f32) -> Plane {
    let mut out = y.clone();
    let mut base = y.clone();
    let last = strengths.iter().rposition(|&s| s > 0.0);
    let Some(last) = last else { return out };
    for (i, &s) in strengths.iter().enumerate().take(last + 1) {
        let next = bilateral::bilateral(&base, s0 * (1 << i) as f32, layer_threshold(threshold, i));
        if s > 0.0 {
            out = out.zip3(&base, &next, |o, b, n| o - s * (b - n));
        }
        base = next;
    }
    out
}

pub fn smooth_luma(y: &Plane, p: &RetouchParams) -> Plane {
    let (s0, levels) = layer_scales(y.w, y.h, p.portrait_size);
    let strengths = band_strengths(levels, p);
    let out = smooth_layers(y, s0, &strengths, p.threshold);
    if p.edge_protect > 0.0 {
        let e = edge_map(y, (y.w.min(y.h) as f32 / 300.0).max(1.5), 0.15, 0.30);
        let k = p.edge_protect / 100.0;
        return out.zip3(&e, y, |o, e, y| {
            let w = e * k;
            o * (1.0 - w) + y * w
        });
    }
    out
}

/// A detected spot: centre and radius in pixels, and its strength.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub x: usize,
    pub y: usize,
    pub r: f32,
    pub strength: f32,
}

/// Scale-normalized LoG blob detection.
///
/// A spot is a local extremum of s^2 LoG at its own scale, round (Hessian
/// eigenvalues of similar size) and with contrast above `t`.
/// `signs`: +1 finds dark spots, -1 bright ones.
pub fn find_spots(
    y: &Plane,
    k: usize,
    t: f32,
    signs: &[f32],
    inner: Option<&Plane>,
    max_spots: usize,
) -> Vec<Spot> {
    let (s0, s1) = ((k as f32 / 10.0).max(1.0), k as f32 / 4.5);
    let mut spots = Vec::new();
    for i in 0..4 {
        let s = s0 * (s1 / s0).powf(i as f32 / 3.0);
        let yb = ops::gaussian_blur(y, s);
        let ixx = ops::sobel(&yb, 2, 0, 0.25);
        let iyy = ops::sobel(&yb, 0, 2, 0.25);
        let ixy = ops::sobel(&yb, 1, 1, 0.25);
        let s2 = s * s;
        let log = ixx.zip(&iyy, |a, b| (a + b) * s2); // >0 dark spot, <0 bright speck
        let round = ixx.zip3(&iyy, &ixy, |a, b, c| {
            (a * b - c * c) / ((a + b).powi(2) / 4.0 + 1e-12)
        });
        let se = ops::ellipse_se(((s * 2.0) as usize | 1).max(3));
        for &sign in signs {
            let l = log.map(|v| v * sign);
            let dl = ops::dilate(&l, &se);
            for idx in 0..l.data.len() {
                let v = l.data[idx];
                if v > t
                    && v >= dl.data[idx]
                    && round.data[idx] > 0.5
                    && inner.is_none_or(|m| m.data[idx] > 0.5)
                {
                    spots.push(Spot {
                        x: idx % y.w,
                        y: idx / y.w,
                        r: s * 1.414 * 1.4,
                        strength: v,
                    });
                }
            }
        }
    }
    spots.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    spots.truncate(max_spots);
    spots
}

/// Result of [`heal_spots`]: healed luma and, per healing pass, the heal
/// weight and spot size (for the matching chroma correction).
pub struct Healed {
    pub y: Plane,
    pub passes: Vec<(Plane, usize)>,
}

/// What one healing pass looks for.
#[derive(Clone, Copy, Debug)]
pub struct SpotKind {
    /// Largest spot diameter as a fraction of the face size.
    pub size: f32,
    /// Luma: +1 = dark spots, -1 = bright specks.
    pub luma_signs: &'static [f32],
    /// Also look for red spots (redness = Cr channel).
    pub red: bool,
    /// Minimal spot strength at slider 100 and at slider 0.
    pub t_min: f32,
    pub t_max: f32,
    /// Spots with more contrast than this are features (nostril, eye corner).
    pub max_contrast: f32,
    /// How far from mask borders (eyes, lips, brows) to stay, in spot sizes.
    pub margin: f32,
}

/// Skin Healing: small blemishes, pimples, red dots, dark pores.
pub const BLEMISHES: SpotKind = SpotKind {
    size: 1.0 / 25.0,
    luma_signs: &[1.0, -1.0],
    red: true,
    t_min: 0.006,
    t_max: 0.036,
    max_contrast: 0.22,
    margin: 1.0,
};

/// Mole removal: bigger, darker round spots (moles, birthmarks).
pub const MOLES: SpotKind = SpotKind {
    size: 1.0 / 12.0,
    luma_signs: &[1.0],
    red: false,
    t_min: 0.012,
    t_max: 0.07,
    max_contrast: 0.6,
    margin: 0.5,
};

/// Redness spots are weaker than luma spots; scale Cr up to compare.
const RED_GAIN: f32 = 2.5;

/// Skin Healing and mole removal.
///
/// Works like healing on a frequency-separated image: only the *low
/// frequencies* of a spot are replaced, by the mean of the skin around it
/// (normalized convolution with the spots left out), so brightness is kept
/// and the pore texture stays. Spots must be round (lines like lip edge,
/// nostril, brow are not), stand out from their surroundings and lie inside
/// the skin mask.
pub fn heal_spots(
    y: &Plane,
    cr: &Plane,
    p: &RetouchParams,
    mask: Option<&Plane>,
) -> Option<Healed> {
    let face = ref_size(y.w, y.h, p.portrait_size);
    let mut cur = y.clone();
    let mut passes = Vec::new();
    for (kind, strength) in [(BLEMISHES, p.healing), (MOLES, p.moles)] {
        if strength <= 0.0 {
            continue;
        }
        let k = ((face * kind.size).clamp(5.0, 101.0) as usize) | 1;
        let (delta, w) = heal_pass(&cur, cr, k, &kind, strength, mask);
        cur = cur.zip3(&delta, &w, |y, d, w| y + d * w);
        passes.push((w, k));
    }
    (!passes.is_empty()).then_some(Healed { y: cur, passes })
}

/// Largest spot size (px) healing works at; bigger faces are shrunk first.
const HEAL_MAX_K: f32 = 17.0;

/// One pass: low-frequency correction and heal weight, full size.
fn heal_pass(
    y: &Plane,
    cr: &Plane,
    k: usize,
    kind: &SpotKind,
    strength: f32,
    mask: Option<&Plane>,
) -> (Plane, Plane) {
    // The correction is smooth, so big spots are analysed on a shrunken
    // copy (spot size ~HEAL_MAX_K px): same look, a fraction of the time.
    let f = k as f32 / HEAL_MAX_K;
    if f <= 1.25 {
        return heal_weights(y, cr, k, kind, strength, mask);
    }
    let (sw, sh) = (
        ((y.w as f32 / f).round() as usize).max(1),
        ((y.h as f32 / f).round() as usize).max(1),
    );
    let small = |p: &Plane| ops::resize_area(p, sw, sh);
    let ms = mask.map(small);
    let ks = ((k as f32 / f).round() as usize) | 1;
    let (d, w) = heal_weights(&small(y), &small(cr), ks, kind, strength, ms.as_ref());
    rayon::join(
        || ops::resize_linear(&d, y.w, y.h),
        || ops::resize_linear(&w, y.w, y.h),
    )
}

/// Local mean of `p` with the `keep == 0` areas left out; falls back to a
/// wide plain blur where nearly everything around is left out.
fn surround(p: &Plane, keep: &Plane, sigma: f32) -> Plane {
    let (num, den) = rayon::join(
        || ops::gaussian_blur(&p.zip(keep, |a, b| a * b), sigma),
        || ops::gaussian_blur(keep, sigma),
    );
    let wide = ops::gaussian_blur(p, sigma * 2.0);
    num.zip3(&den, &wide, |n, d, w| {
        let a = (d / 0.15).clamp(0.0, 1.0);
        (n / d.max(1e-6)) * a + w * (1.0 - a)
    })
}

/// Low-frequency correction (surrounding skin minus spot) and per-pixel
/// heal weight for spots up to `k` px.
fn heal_weights(
    y: &Plane,
    cr: &Plane,
    k: usize,
    kind: &SpotKind,
    strength: f32,
    mask: Option<&Plane>,
) -> (Plane, Plane) {
    let se = ops::ellipse_se(k);
    // close first: a blemish is redder than skin and punches a hole in the
    // mask (only holes up to spot size; eyes and mouth stay excluded)
    let inner = mask.map(|m| {
        let filled = ops::close(m, &se).map(|v| ((v - 0.6) / 0.3).clamp(0.0, 1.0));
        ops::erode(
            &filled,
            &ops::ellipse_se(((k as f32 * kind.margin) as usize) | 1),
        )
    });

    let t = kind.t_min + (kind.t_max - kind.t_min) * (1.0 - strength / 100.0);
    let mut spots: Vec<(Spot, bool)> = find_spots(y, k, t, kind.luma_signs, inner.as_ref(), 4000)
        .into_iter()
        .map(|s| (s, false))
        .collect();
    let redness = kind.red.then(|| cr.map(|v| v * RED_GAIN));
    if let Some(red) = &redness {
        // red = bright in Cr
        spots.extend(
            find_spots(red, k, t, &[-1.0], inner.as_ref(), 4000)
                .into_iter()
                .map(|s| (s, true)),
        );
    }

    // surrounding skin: local mean with all candidate spots left out
    let mut discs = Plane::zeros(y.w, y.h);
    for (s, _) in &spots {
        ops::fill_circle_aa(&mut discs, s.x as f32, s.y as f32, s.r * 1.2, 1.0);
    }
    // only skin counts as surroundings (not hair, brows or background)
    let keep = match mask {
        Some(m) => discs.zip(m, |d, m| (1.0 - d) * m.clamp(0.0, 1.0)),
        None => discs.map(|d| 1.0 - d),
    };
    let sigma_ref = k as f32 / 2.5;
    // low frequencies: what the spot looks like without pores
    let sigma_low = ((k as f32 / 10.0).max(1.0) / 2.0).max(0.7);
    let (sur, low) = rayon::join(
        || surround(y, &keep, sigma_ref),
        || ops::gaussian_blur(y, sigma_low),
    );
    let red_maps = redness.as_ref().map(|r| {
        rayon::join(
            || surround(r, &keep, sigma_ref),
            || ops::gaussian_blur(r, sigma_low),
        )
    });

    // keep spots that really stand out from the skin around them
    let min_contrast = 0.4 * t;
    let mut w = Plane::zeros(y.w, y.h);
    for (s, is_red) in &spots {
        let i = s.y * y.w + s.x;
        let c = match (&red_maps, is_red) {
            (Some((rs, rl)), true) => rl.data[i] - rs.data[i],
            _ => (low.data[i] - sur.data[i]).abs(),
        };
        if c < min_contrast || c > kind.max_contrast {
            continue; // flat texture, or very contrasty = nostril / eye corner
        }
        ops::fill_circle_aa(&mut w, s.x as f32, s.y as f32, s.r * 1.2, 1.0);
    }
    let gain = (strength / 50.0).min(1.0);
    let w = ops::gaussian_blur(&w, (k as f32 / 12.0).max(1.0))
        .map(|v| (v * 1.3).clamp(0.0, 1.0) * gain);
    let w = match &inner {
        Some(inner) => w.zip(inner, |w, i| w * (i * 2.0).clamp(0.0, 1.0)),
        None => w,
    };
    (sur.zip(&low, |s, l| s - l), w)
}

/// Fill healed spots with the average chroma of the skin around them
/// (normalized convolution: spots are left out of the average).
pub fn heal_chroma(c: &Plane, h: &Healed) -> Plane {
    let mut c = c.clone();
    for (weight, k) in &h.passes {
        let keep = weight.map(|w| 1.0 - w);
        let sigma = (*k as f32 / 2.0).max(1.0);
        let (num, den) = rayon::join(
            || ops::gaussian_blur(&c.zip(&keep, |a, b| a * b), sigma),
            || ops::gaussian_blur(&keep, sigma),
        );
        let reference = num.zip(&den, |n, d| n / (d + 1e-6));
        c = c.zip3(&reference, weight, |c, r, w| c + (r - c) * w);
    }
    c
}

pub fn smooth_chroma(cc: &Plane, guide: &Plane, strength: f32) -> Plane {
    if strength <= 0.0 {
        return cc.clone();
    }
    let r = ((guide.w.min(guide.h) as f32 * 0.01) as usize).max(2);
    let sm = ops::guided_filter(guide, cc, r, 1e-4);
    cc.zip(&sm, |c, s| c + (s - c) * strength)
}

/// Retouch an RGB image (0..1). `mask` (0..1) says where to apply it,
/// `heal_mask` where Skin Healing may work (defaults to `mask`).
pub fn retouch(
    rgb: &Rgb,
    p: &RetouchParams,
    mask: Option<&Plane>,
    heal_mask: Option<&Plane>,
) -> Rgb {
    let [y, cr, cb] = to_ycc(rgb);
    let healed = heal_spots(&y, &cr, p, heal_mask.or(mask));
    let yh = healed.as_ref().map_or(&y, |h| &h.y);
    let u = p.uniformity / 100.0;
    let (y2, (cr2, cb2)) = rayon::join(
        || smooth_luma(yh, p),
        || {
            rayon::join(
                || {
                    let c = healed
                        .as_ref()
                        .map_or_else(|| cr.clone(), |h| heal_chroma(&cr, h));
                    smooth_chroma(&c, yh, u)
                },
                || {
                    let c = healed
                        .as_ref()
                        .map_or_else(|| cb.clone(), |h| heal_chroma(&cb, h));
                    smooth_chroma(&c, yh, u)
                },
            )
        },
    );
    let out = from_ycc(&[y2, cr2, cb2]);
    let a = p.amount / 100.0;
    let blend = |o: &Plane, s: &Plane| match mask {
        Some(m) => o.zip3(s, m, |o, s, m| (s + (o - s) * m * a).clamp(0.0, 1.0)),
        None => o.zip(s, |o, s| (s + (o - s) * a).clamp(0.0, 1.0)),
    };
    Rgb {
        c: [0, 1, 2].map(|k| blend(&out.c[k], &rgb.c[k])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_roundtrip() {
        let mut s = 1u32;
        let y = Plane::from_vec(
            257,
            300,
            (0..257 * 300)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 8) as f32 / (1 << 24) as f32
                })
                .collect(),
        );
        let r = reconstruct(&laplacian_pyramid(&y, 4));
        assert!(r.zip(&y, |a, b| (a - b).abs()).max() < 1e-5);
    }

    #[test]
    fn strengths_interpolate() {
        let p = RetouchParams {
            fine: 100.0,
            medium: 50.0,
            large: 0.0,
            ..Default::default()
        };
        assert_eq!(band_strengths(3, &p), vec![1.0, 0.5, 0.0]);
    }
}
