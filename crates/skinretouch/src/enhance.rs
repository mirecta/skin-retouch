//! Finishing "Enhancements" (applied after smoothing, optionally masked).

use crate::color::{from_ycc, to_ycc};
use crate::ops;
use crate::plane::{Plane, Rgb};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnhanceParams {
    /// 0..100 unsharp mask on luma.
    pub sharpness: f32,
    /// 0..100 soft glow.
    pub softness: f32,
    /// -100..100
    pub warmth: f32,
    /// -100..100 (+ = magenta, - = green)
    pub tint: f32,
    /// -100..100
    pub brightness: f32,
    /// -100..100
    pub contrast: f32,
}

impl EnhanceParams {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
}

pub fn enhance(rgb: &Rgb, e: &EnhanceParams, mask: Option<&Plane>) -> Rgb {
    if e.is_identity() {
        return rgb.clone();
    }
    let sigma = (rgb.w().min(rgb.h()) as f32 / 600.0).max(1.0);
    let mut out = rgb.clone();

    if e.sharpness > 0.0 {
        let [y, cr, cb] = to_ycc(&out);
        let k = e.sharpness / 100.0 * 1.5;
        let y = y.zip(&ops::gaussian_blur(&y, sigma), |y, b| y + (y - b) * k);
        out = from_ycc(&[y, cr, cb]);
    }

    if e.softness > 0.0 {
        let glow = out.map(|p| ops::gaussian_blur(p, sigma * 8.0));
        let k = e.softness / 100.0 * 0.5;
        out = Rgb {
            c: [0, 1, 2].map(|i| {
                out.c[i].zip(&glow.c[i], |o, g| {
                    let screen = 1.0 - (1.0 - o) * (1.0 - g);
                    o + (screen - o) * k
                })
            }),
        };
    }

    let (wa, ti) = (e.warmth / 100.0 * 0.08, e.tint / 100.0 * 0.05);
    let offset = [wa + ti * 0.5, -ti, -wa + ti * 0.5].map(|v| v + e.brightness / 100.0 * 0.2);
    let k = 1.0 + e.contrast / 100.0 * 0.5;
    let out = Rgb {
        c: [0, 1, 2].map(|i| {
            let o = offset[i];
            out.c[i].map(|v| (((v + o) - 0.5) * k + 0.5).clamp(0.0, 1.0))
        }),
    };
    match mask {
        Some(m) => Rgb {
            c: [0, 1, 2].map(|i| out.c[i].zip3(&rgb.c[i], m, |o, s, m| s + (o - s) * m)),
        },
        None => out,
    }
}
