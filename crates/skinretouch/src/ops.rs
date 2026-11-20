//! Image operations used by the pipeline (the subset of OpenCV we need).
//!
//! Borders follow OpenCV's default `BORDER_REFLECT_101` (…cb|abcd|cb…) for
//! filters; morphology ignores pixels outside the image, like OpenCV does.

use crate::plane::Plane;
use rayon::prelude::*;

/// Mirror an index into 0..n without repeating the edge pixel.
#[inline]
pub fn reflect101(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

// ---------------------------------------------------------------- convolution

/// Horizontal 1D convolution with an odd, centred kernel.
fn conv_rows(p: &Plane, k: &[f32]) -> Plane {
    let r = k.len() / 2;
    let w = p.w;
    let mut out = Plane::zeros(p.w, p.h);
    out.data.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0.0f32; w + 2 * r],
        |pad, (y, dst)| {
            let src = p.row(y);
            for (i, v) in pad.iter_mut().enumerate() {
                *v = src[reflect101(i as isize - r as isize, w)];
            }
            for (x, d) in dst.iter_mut().enumerate() {
                let win = &pad[x..x + k.len()];
                *d = win.iter().zip(k).map(|(a, b)| a * b).sum();
            }
        },
    );
    out
}

/// Vertical 1D convolution with an odd, centred kernel.
fn conv_cols(p: &Plane, k: &[f32]) -> Plane {
    let r = k.len() as isize / 2;
    let mut out = Plane::zeros(p.w, p.h);
    out.data
        .par_chunks_mut(p.w)
        .enumerate()
        .for_each(|(y, dst)| {
            for (j, &kv) in k.iter().enumerate() {
                let src = p.row(reflect101(y as isize + j as isize - r, p.h));
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += kv * s;
                }
            }
        });
    out
}

/// Separable filter: `kx` along rows, `ky` along columns.
pub fn sep_filter(p: &Plane, kx: &[f32], ky: &[f32]) -> Plane {
    conv_cols(&conv_rows(p, kx), ky)
}

/// OpenCV's Gaussian kernel for float images (size = round(8 sigma + 1) | 1).
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let n = ((sigma * 8.0 + 1.0).round() as usize) | 1;
    let c = (n / 2) as f32;
    let mut k: Vec<f32> = (0..n)
        .map(|i| (-((i as f32 - c).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Gaussian blur. Exact kernel for small sigma; for big sigma three box
/// blurs (error < 1 %, but cost independent of sigma).
pub fn gaussian_blur(p: &Plane, sigma: f32) -> Plane {
    if sigma <= 0.0 {
        return p.clone();
    }
    if sigma <= 6.0 {
        let k = gaussian_kernel(sigma);
        return sep_filter(p, &k, &k);
    }
    // Box sizes whose cascade has the requested variance (Kovesi 2010).
    let n = 3.0f32;
    let wi = ((12.0 * sigma * sigma / n) + 1.0).sqrt().floor();
    let wl = if (wi as i32) % 2 == 0 { wi - 1.0 } else { wi };
    let wu = wl + 2.0;
    let m =
        ((12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / (-4.0 * wl - 4.0)).round();
    let mut out = p.clone();
    for i in 0..3 {
        let size = if (i as f32) < m { wl } else { wu };
        out = box_blur(&out, ((size - 1.0) / 2.0) as usize);
    }
    out
}

/// Running-sum mean along rows, window 2r+1, reflect-101 border.
fn box_rows(p: &Plane, r: usize) -> Plane {
    let w = p.w;
    let n = (2 * r + 1) as f32;
    let mut out = Plane::zeros(p.w, p.h);
    out.data.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0.0f32; w + 2 * r],
        |pad, (y, dst)| {
            let src = p.row(y);
            for (i, v) in pad.iter_mut().enumerate() {
                *v = src[reflect101(i as isize - r as isize, w)];
            }
            let mut s: f64 = pad[..2 * r + 1].iter().map(|&v| v as f64).sum();
            dst[0] = s as f32 / n;
            for x in 1..w {
                s += pad[x + 2 * r] as f64 - pad[x - 1] as f64;
                dst[x] = s as f32 / n;
            }
        },
    );
    out
}

pub fn transpose(p: &Plane) -> Plane {
    const B: usize = 64;
    let mut out = Plane::zeros(p.h, p.w);
    let (w, h) = (p.w, p.h);
    // each parallel task fills B output rows (= B input columns)
    out.data
        .par_chunks_mut(h * B)
        .enumerate()
        .for_each(|(bi, chunk)| {
            let x0 = bi * B;
            let rows = chunk.len() / h;
            for y0 in (0..h).step_by(B) {
                for x in x0..x0 + rows {
                    for y in y0..(y0 + B).min(h) {
                        chunk[(x - x0) * h + y] = p.data[y * w + x];
                    }
                }
            }
        });
    out
}

/// Normalized box filter (mean over a (2r+1)^2 window).
pub fn box_blur(p: &Plane, r: usize) -> Plane {
    if r == 0 {
        return p.clone();
    }
    transpose(&box_rows(&transpose(&box_rows(p, r)), r))
}

const PYR: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Blur with the 5-tap binomial kernel and drop every second row/column.
pub fn pyr_down(p: &Plane) -> Plane {
    let b = sep_filter(p, &PYR, &PYR);
    let (w, h) = (p.w.div_ceil(2), p.h.div_ceil(2));
    let mut out = Plane::zeros(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        let src = b.row(2 * y);
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src[2 * x];
        }
    });
    out
}

/// Inverse of [`pyr_down`]: zero-insert to (w, h) and interpolate.
pub fn pyr_up(p: &Plane, w: usize, h: usize) -> Plane {
    let mut up = Plane::zeros(w, h);
    for y in 0..p.h.min(h.div_ceil(2)) {
        for x in 0..p.w.min(w.div_ceil(2)) {
            up.set(2 * x, 2 * y, p.get(x, y));
        }
    }
    let k = PYR.map(|v| v * 2.0);
    sep_filter(&up, &k, &k)
}

/// Sobel derivative (order dx, dy in 0..=2, 3x3 aperture) times `scale`.
pub fn sobel(p: &Plane, dx: u8, dy: u8, scale: f32) -> Plane {
    let k = |order: u8| -> [f32; 3] {
        match order {
            0 => [1.0, 2.0, 1.0],
            1 => [-1.0, 0.0, 1.0],
            _ => [1.0, -2.0, 1.0],
        }
    };
    let kx = k(dx).map(|v| v * scale);
    sep_filter(p, &kx, &k(dy))
}

// ---------------------------------------------------------------- morphology

/// Structuring element as half-widths per row offset: (dy, a) means columns
/// -a..=a of row dy. Same rasterization as `cv2.getStructuringElement`.
pub fn ellipse_se(k: usize) -> Vec<(isize, usize)> {
    let r = (k / 2) as isize;
    let c = (k / 2) as f32;
    let inv_r2 = if r > 0 { 1.0 / (r * r) as f32 } else { 0.0 };
    (-r..=r)
        .map(|dy| {
            let a = (c * (((r * r - dy * dy) as f32) * inv_r2).sqrt()).round() as usize;
            (dy, a)
        })
        .collect()
}

pub fn rect_se(k: usize) -> Vec<(isize, usize)> {
    let r = (k / 2) as isize;
    (-r..=r).map(|dy| (dy, k / 2)).collect()
}

/// 1D running max (or min) over a window -a..=a; outside pixels are ignored.
/// van Herk / Gil-Werman, O(1) per pixel.
fn run_extreme(
    src: &[f32],
    a: usize,
    is_max: bool,
    g: &mut Vec<f32>,
    hb: &mut Vec<f32>,
    out: &mut [f32],
) {
    let op = |x: f32, y: f32| if is_max { x.max(y) } else { x.min(y) };
    let neutral = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let n = src.len();
    let win = 2 * a + 1;
    let len = n + 2 * a;
    let len_pad = len.div_ceil(win) * win;
    g.clear();
    g.resize(len_pad, neutral);
    hb.clear();
    hb.resize(len_pad, neutral);
    g[a..a + n].copy_from_slice(src);
    hb[a..a + n].copy_from_slice(src);
    for b in (0..len_pad).step_by(win) {
        for i in b + 1..b + win {
            g[i] = op(g[i], g[i - 1]);
        }
        for i in (b..b + win - 1).rev() {
            hb[i] = op(hb[i], hb[i + 1]);
        }
    }
    // window over padded [i, i+win) has its centre at source pixel i
    for (i, o) in out.iter_mut().enumerate() {
        *o = op(hb[i], g[i + win - 1]);
    }
}

fn morph(p: &Plane, se: &[(isize, usize)], is_max: bool) -> Plane {
    let neutral = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut out = Plane::zeros(p.w, p.h);
    out.data.par_chunks_mut(p.w).enumerate().for_each_init(
        || (vec![0.0f32; p.w], Vec::new(), Vec::new()),
        |(tmp, g, hb), (y, dst)| {
            dst.fill(neutral);
            for &(dy, a) in se {
                let sy = y as isize + dy;
                if sy < 0 || sy >= p.h as isize {
                    continue;
                }
                run_extreme(p.row(sy as usize), a, is_max, g, hb, tmp);
                for (d, &t) in dst.iter_mut().zip(tmp.iter()) {
                    *d = if is_max { d.max(t) } else { d.min(t) };
                }
            }
        },
    );
    out
}

pub fn dilate(p: &Plane, se: &[(isize, usize)]) -> Plane {
    morph(p, se, true)
}

pub fn erode(p: &Plane, se: &[(isize, usize)]) -> Plane {
    morph(p, se, false)
}

pub fn close(p: &Plane, se: &[(isize, usize)]) -> Plane {
    erode(&dilate(p, se), se)
}

pub fn open(p: &Plane, se: &[(isize, usize)]) -> Plane {
    dilate(&erode(p, se), se)
}

// ---------------------------------------------------------------- resize

/// Per output index: (first source index, weights) for a 1D resampling.
fn area_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            let (a, b) = (i as f64 * scale, (i + 1) as f64 * scale);
            let first = a.floor() as usize;
            let last = (b.ceil() as usize).min(src);
            let w: Vec<f32> = (first..last)
                .map(|j| ((b.min(j as f64 + 1.0) - a.max(j as f64)) / scale) as f32)
                .collect();
            (first, w)
        })
        .collect()
}

fn linear_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f32 / dst as f32;
    (0..dst)
        .map(|i| {
            let s = ((i as f32 + 0.5) * scale - 0.5).clamp(0.0, (src - 1) as f32);
            let x0 = (s.floor() as usize).min(src.saturating_sub(2));
            let f = s - x0 as f32;
            if src == 1 {
                (0, vec![1.0])
            } else {
                (x0, vec![1.0 - f, f])
            }
        })
        .collect()
}

fn resample(
    p: &Plane,
    w: usize,
    h: usize,
    wx: &[(usize, Vec<f32>)],
    wy: &[(usize, Vec<f32>)],
) -> Plane {
    // rows first: p.h x w
    let mut tmp = Plane::zeros(w, p.h);
    tmp.data.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        let src = p.row(y);
        for (d, (f, ws)) in dst.iter_mut().zip(wx) {
            *d = ws.iter().enumerate().map(|(j, &k)| src[f + j] * k).sum();
        }
    });
    let mut out = Plane::zeros(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        let (f, ws) = &wy[y];
        for (j, &k) in ws.iter().enumerate() {
            for (d, s) in dst.iter_mut().zip(tmp.row(f + j)) {
                *d += k * s;
            }
        }
    });
    out
}

/// Bilinear resize with half-pixel centres (cv2.INTER_LINEAR).
pub fn resize_linear(p: &Plane, w: usize, h: usize) -> Plane {
    resample(p, w, h, &linear_weights(p.w, w), &linear_weights(p.h, h))
}

/// Pixel-area averaging for shrinking (cv2.INTER_AREA); bilinear when growing.
pub fn resize_area(p: &Plane, w: usize, h: usize) -> Plane {
    let wx = if w < p.w {
        area_weights(p.w, w)
    } else {
        linear_weights(p.w, w)
    };
    let wy = if h < p.h {
        area_weights(p.h, h)
    } else {
        linear_weights(p.h, h)
    };
    resample(p, w, h, &wx, &wy)
}

// ---------------------------------------------------------------- guided filter

/// Subsampling factor of the fast guided filter (He & Sun 2015): the linear
/// coefficients are smooth, so they can be computed on a smaller image.
fn fast_factor(r: usize) -> usize {
    (r / 4).clamp(1, 8)
}

fn shrink(p: &Plane, s: usize) -> Plane {
    if s == 1 {
        p.clone()
    } else {
        resize_area(p, p.w.div_ceil(s), p.h.div_ceil(s))
    }
}

fn grow(p: &Plane, w: usize, h: usize) -> Plane {
    if p.w == w && p.h == h {
        p.clone()
    } else {
        resize_linear(p, w, h)
    }
}

/// Mean linear coefficients (a, b) of the grey guided filter.
fn guided_coef(guide: &Plane, src: &Plane, r: usize, eps: f32) -> (Plane, Plane) {
    let mi = box_blur(guide, r);
    let mp = box_blur(src, r);
    let ii = box_blur(&guide.zip(guide, |a, b| a * b), r);
    let ip = box_blur(&guide.zip(src, |a, b| a * b), r);
    let a = Plane::from_vec(
        guide.w,
        guide.h,
        (0..guide.data.len())
            .into_par_iter()
            .map(|i| {
                let var = ii.data[i] - mi.data[i] * mi.data[i];
                let cov = ip.data[i] - mi.data[i] * mp.data[i];
                cov / (var + eps)
            })
            .collect(),
    );
    let b = mp.zip3(&a, &mi, |p, a, i| p - a * i);
    rayon::join(|| box_blur(&a, r), || box_blur(&b, r))
}

/// Guided filter (He et al.) with a single-channel guide.
pub fn guided_filter(guide: &Plane, src: &Plane, r: usize, eps: f32) -> Plane {
    let s = fast_factor(r);
    let (ma, mb) = guided_coef(&shrink(guide, s), &shrink(src, s), (r / s).max(1), eps);
    let (ma, mb) = rayon::join(
        || grow(&ma, guide.w, guide.h),
        || grow(&mb, guide.w, guide.h),
    );
    ma.zip3(&mb, guide, |a, b, i| a * i + b)
}

/// Mean linear coefficients (a[3], b) of the colour guided filter.
fn guided_coef_color(guide: &[Plane; 3], src: &Plane, r: usize, eps: f32) -> ([Plane; 3], Plane) {
    let bx = |p: &Plane| box_blur(p, r);
    let m: Vec<Plane> = guide.par_iter().map(bx).collect();
    let mp = bx(src);
    // covariance of the guide: (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let var: Vec<Plane> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let c = bx(&guide[a].zip(&guide[b], |x, y| x * y));
            c.zip3(&m[a], &m[b], |c, x, y| c - x * y)
        })
        .collect();
    let cov: Vec<Plane> = (0..3)
        .into_par_iter()
        .map(|k| bx(&guide[k].zip(src, |x, y| x * y)).zip3(&m[k], &mp, |c, x, y| c - x * y))
        .collect();
    let n = src.data.len();
    let coef: Vec<[f32; 3]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let s = [
                var[0].data[i] + eps,
                var[1].data[i],
                var[2].data[i],
                var[3].data[i] + eps,
                var[4].data[i],
                var[5].data[i] + eps,
            ];
            // inverse of the symmetric 3x3 matrix [[s0 s1 s2] [s1 s3 s4] [s2 s4 s5]]
            let i00 = s[3] * s[5] - s[4] * s[4];
            let i01 = s[2] * s[4] - s[1] * s[5];
            let i02 = s[1] * s[4] - s[2] * s[3];
            let i11 = s[0] * s[5] - s[2] * s[2];
            let i12 = s[1] * s[2] - s[0] * s[4];
            let i22 = s[0] * s[3] - s[1] * s[1];
            let det = s[0] * i00 + s[1] * i01 + s[2] * i02;
            let c = [cov[0].data[i], cov[1].data[i], cov[2].data[i]];
            [
                (i00 * c[0] + i01 * c[1] + i02 * c[2]) / det,
                (i01 * c[0] + i11 * c[1] + i12 * c[2]) / det,
                (i02 * c[0] + i12 * c[1] + i22 * c[2]) / det,
            ]
        })
        .collect();
    let a: [Plane; 3] =
        [0, 1, 2].map(|k| Plane::from_vec(src.w, src.h, coef.par_iter().map(|c| c[k]).collect()));
    let b = Plane::from_vec(
        src.w,
        src.h,
        (0..n)
            .into_par_iter()
            .map(|i| mp.data[i] - (0..3).map(|k| a[k].data[i] * m[k].data[i]).sum::<f32>())
            .collect(),
    );
    let ma: Vec<Plane> = a.par_iter().map(bx).collect();
    let [m0, m1, m2]: [Plane; 3] = ma.try_into().expect("three planes");
    ([m0, m1, m2], bx(&b))
}

/// Guided filter with a colour (3-channel) guide.
pub fn guided_filter_color(guide: &[Plane; 3], src: &Plane, r: usize, eps: f32) -> Plane {
    let s = fast_factor(r);
    let small = [0, 1, 2].map(|k| shrink(&guide[k], s));
    let (ma, mb) = guided_coef_color(&small, &shrink(src, s), (r / s).max(1), eps);
    let (w, h) = (src.w, src.h);
    let ma: Vec<Plane> = ma.par_iter().map(|p| grow(p, w, h)).collect();
    let mb = grow(&mb, w, h);
    Plane::from_vec(
        w,
        h,
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                mb.data[i]
                    + (0..3)
                        .map(|k| ma[k].data[i] * guide[k].data[i])
                        .sum::<f32>()
            })
            .collect(),
    )
}

// ---------------------------------------------------------------- drawing

/// Anti-aliased filled disc blended towards `v` (like cv2.circle LINE_AA, -1).
pub fn fill_circle_aa(p: &mut Plane, cx: f32, cy: f32, r: f32, v: f32) {
    let x0 = (cx - r - 1.0).floor().max(0.0) as usize;
    let y0 = (cy - r - 1.0).floor().max(0.0) as usize;
    let x1 = ((cx + r + 1.0).ceil() as usize).min(p.w.saturating_sub(1));
    let y1 = ((cy + r + 1.0).ceil() as usize).min(p.h.saturating_sub(1));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let cov = (r + 0.5 - d).clamp(0.0, 1.0);
            if cov > 0.0 {
                let o = p.get(x, y);
                p.set(x, y, o + (v - o) * cov);
            }
        }
    }
}

/// Filled axis-aligned ellipse.
pub fn fill_ellipse(p: &mut Plane, cx: f32, cy: f32, ax: f32, ay: f32, v: f32) {
    if ax <= 0.0 || ay <= 0.0 {
        return;
    }
    let y0 = (cy - ay).floor().max(0.0) as usize;
    let y1 = ((cy + ay).ceil().max(0.0) as usize).min(p.h.saturating_sub(1));
    for y in y0..=y1 {
        let t = 1.0 - ((y as f32 - cy) / ay).powi(2);
        if t < 0.0 {
            continue;
        }
        let dx = ax * t.sqrt();
        let xa = (cx - dx).round().max(0.0) as usize;
        let xb = ((cx + dx).round().max(0.0) as usize).min(p.w.saturating_sub(1));
        for x in xa..=xb {
            p.set(x, y, v);
        }
    }
}

/// Filled rectangle [x0, x1] x [y0, y1], clipped.
pub fn fill_rect(p: &mut Plane, x0: i64, y0: i64, x1: i64, y1: i64, v: f32) {
    let cl = |a: i64, n: usize| a.clamp(0, n as i64 - 1) as usize;
    if x1 < 0 || y1 < 0 || x0 >= p.w as i64 || y0 >= p.h as i64 {
        return;
    }
    for y in cl(y0, p.h)..=cl(y1, p.h) {
        for x in cl(x0, p.w)..=cl(x1, p.w) {
            p.set(x, y, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: usize, h: usize, seed: u64) -> Plane {
        let mut s = seed;
        Plane::from_vec(
            w,
            h,
            (0..w * h)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 40) as f32 / (1u64 << 24) as f32
                })
                .collect(),
        )
    }

    #[test]
    fn transpose_roundtrip() {
        let p = noise(131, 77, 3);
        assert_eq!(transpose(&transpose(&p)), p);
        assert_eq!(transpose(&p).get(5, 9), p.get(9, 5));
    }

    #[test]
    fn box_blur_keeps_constant_and_mean() {
        let p = Plane::new(50, 40, 0.3);
        let b = box_blur(&p, 7);
        assert!(b.data.iter().all(|v| (v - 0.3).abs() < 1e-5));
    }

    #[test]
    fn morph_matches_naive() {
        let p = noise(40, 30, 7);
        let se = ellipse_se(7);
        let d = dilate(&p, &se);
        for y in 0..p.h {
            for x in 0..p.w {
                let mut m = f32::NEG_INFINITY;
                for &(dy, a) in &se {
                    let sy = y as isize + dy;
                    if sy < 0 || sy >= p.h as isize {
                        continue;
                    }
                    for dx in -(a as isize)..=a as isize {
                        let sx = x as isize + dx;
                        if sx >= 0 && sx < p.w as isize {
                            m = m.max(p.get(sx as usize, sy as usize));
                        }
                    }
                }
                assert_eq!(d.get(x, y), m);
            }
        }
    }

    #[test]
    fn ellipse_se_shape() {
        // cv2.getStructuringElement(MORPH_ELLIPSE, (5,5)) rows: 00100 11111 11111 11111 00100
        assert_eq!(
            ellipse_se(5),
            vec![(-2, 0), (-1, 2), (0, 2), (1, 2), (2, 0)]
        );
    }

    #[test]
    fn gaussian_box_approx_close_to_exact() {
        let p = noise(200, 150, 11);
        let k = gaussian_kernel(8.0);
        let exact = sep_filter(&p, &k, &k);
        let approx = gaussian_blur(&p, 8.0);
        let err = exact.zip(&approx, |a, b| (a - b).abs()).max();
        assert!(err < 0.01, "err {err}");
    }

    #[test]
    fn guided_filters_reproduce_linear_function_of_guide() {
        // q = a*I + b locally, so a source that *is* a linear function of the
        // guide must come out (almost) unchanged when eps is small.
        let g = [noise(60, 50, 5), noise(60, 50, 6), noise(60, 50, 7)];
        let s = g[0].zip3(&g[1], &g[2], |r, gg, b| 0.5 * r - 0.3 * gg + 0.2 * b + 0.1);
        let q = guided_filter_color(&g, &s, 4, 1e-6);
        assert!(q.zip(&s, |a, b| (a - b).abs()).max() < 1e-3);
        let s1 = g[0].map(|v| 0.7 * v + 0.2);
        let q1 = guided_filter(&g[0], &s1, 4, 1e-6);
        assert!(q1.zip(&s1, |a, b| (a - b).abs()).max() < 1e-3);
    }

    #[test]
    fn resize_area_mean() {
        let p = noise(100, 80, 1);
        let r = resize_area(&p, 50, 40);
        assert!((r.mean() - p.mean()).abs() < 1e-4);
        assert!(
            (r.get(0, 0) - (p.get(0, 0) + p.get(1, 0) + p.get(0, 1) + p.get(1, 1)) / 4.0).abs()
                < 1e-5
        );
    }
}
