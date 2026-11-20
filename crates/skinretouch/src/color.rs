//! RGB <-> YCrCb (same coefficients as OpenCV's float conversion, delta 0.5).

use crate::plane::{Plane, Rgb};
use rayon::prelude::*;

#[inline]
pub fn rgb_to_ycc(r: f32, g: f32, b: f32) -> [f32; 3] {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    [y, (r - y) * 0.713 + 0.5, (b - y) * 0.564 + 0.5]
}

#[inline]
pub fn ycc_to_rgb(y: f32, cr: f32, cb: f32) -> [f32; 3] {
    let (cr, cb) = (cr - 0.5, cb - 0.5);
    [y + 1.403 * cr, y - 0.714 * cr - 0.344 * cb, y + 1.773 * cb]
}

/// Planes [Y, Cr, Cb].
pub fn to_ycc(img: &Rgb) -> [Plane; 3] {
    convert(&img.c, rgb_to_ycc)
}

pub fn from_ycc(ycc: &[Plane; 3]) -> Rgb {
    Rgb {
        c: convert(ycc, ycc_to_rgb),
    }
}

/// Luma only.
pub fn luma(img: &Rgb) -> Plane {
    img.c[0].zip3(&img.c[1], &img.c[2], |r, g, b| {
        0.299 * r + 0.587 * g + 0.114 * b
    })
}

fn convert(src: &[Plane; 3], f: fn(f32, f32, f32) -> [f32; 3]) -> [Plane; 3] {
    let (w, h) = (src[0].w, src[0].h);
    let mut out = [Plane::zeros(w, h), Plane::zeros(w, h), Plane::zeros(w, h)];
    let [o0, o1, o2] = &mut out;
    o0.data
        .par_iter_mut()
        .zip(o1.data.par_iter_mut())
        .zip(o2.data.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((a, b), c))| {
            let v = f(src[0].data[i], src[1].data[i], src[2].data[i]);
            (*a, *b, *c) = (v[0], v[1], v[2]);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for &(r, g, b) in &[
            (0.85, 0.65, 0.55),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.1, 0.9, 0.3),
        ] {
            let [y, cr, cb] = rgb_to_ycc(r, g, b);
            let [r2, g2, b2] = ycc_to_rgb(y, cr, cb);
            assert!((r - r2).abs() < 2e-3 && (g - g2).abs() < 2e-3 && (b - b2).abs() < 2e-3);
        }
    }
}
