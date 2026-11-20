//! Fast bilateral filter on a bilateral grid (Chen, Paris & Durand 2007).
//!
//! The steps follow the names found in Portraiture's binary:
//! 0. `bufferZero`      – allocate an empty grid (x, y, intensity),
//! 1. `Downsample`      – splat every pixel into its grid cell (value, weight),
//! 2. `ConvolutionIter` – blur the grid along x, y and intensity,
//! 3. `Upsample` +
//! 4. `Interpolate`     – read the grid back at each pixel (trilinear) and
//!    divide value by weight.
//!
//! Blurring in the intensity direction is what makes it edge-aware: pixels
//! that differ by much more than `sigma_r` live in different grid cells and
//! never mix.

use crate::plane::Plane;
use rayon::prelude::*;

/// Grid layout: [gy][gx][gz] of (sum of values, sum of weights).
struct Grid {
    gw: usize,
    gh: usize,
    gd: usize,
    data: Vec<[f32; 2]>,
}

impl Grid {
    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (y * self.gw + x) * self.gd + z
    }
}

/// Bilateral filter with spatial sigma `sigma_s` (px) and range sigma
/// `sigma_r` (intensity units). Cost is O(pixels) independent of sigma_s.
pub fn bilateral(img: &Plane, sigma_s: f32, sigma_r: f32) -> Plane {
    let (w, h) = (img.w, img.h);
    let ss = sigma_s.max(1.0);
    let sr = sigma_r.max(1e-4);
    let (lo, hi) = img
        .data
        .par_iter()
        .fold(|| (f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)))
        .reduce(
            || (f32::MAX, f32::MIN),
            |(a, b), (c, d)| (a.min(c), b.max(d)),
        );
    let hi = hi.max(lo + 1e-6);

    // Step 0: empty grid with one cell of padding on each side
    let gw = (((w - 1) as f32 / ss).round() as usize) + 3;
    let gh = (((h - 1) as f32 / ss).round() as usize) + 3;
    let gd = (((hi - lo) / sr).round() as usize) + 3;
    let mut grid = Grid {
        gw,
        gh,
        gd,
        data: vec![[0.0; 2]; gw * gh * gd],
    };

    // Step 1: splat to the nearest cell; every grid row gy only receives
    // pixel rows that round to it, so rows can be filled in parallel
    let slab = gw * gd;
    grid.data
        .par_chunks_mut(slab)
        .enumerate()
        .for_each(|(gy, cells)| {
            if gy == 0 {
                return;
            }
            let y0 = ((gy as f32 - 1.5) * ss).ceil().max(0.0) as usize;
            let y1 = (((gy as f32 - 0.5) * ss).ceil() as usize).min(h);
            for y in y0..y1 {
                if (y as f32 / ss).round() as usize + 1 != gy {
                    continue;
                }
                for (x, &v) in img.row(y).iter().enumerate() {
                    let gx = (x as f32 / ss).round() as usize + 1;
                    let gz = ((v - lo) / sr).round() as usize + 1;
                    let c = &mut cells[gx * gd + gz];
                    c[0] += v;
                    c[1] += 1.0;
                }
            }
        });

    // Step 2: separable [1 4 6 4 1] blur along z, x and y (~ Gaussian, sigma 1 cell)
    blur_grid(&mut grid);

    // Steps 3 + 4: trilinear read-back
    let mut out = Plane::zeros(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = y as f32 / ss + 1.0;
        let y0 = (fy.floor() as usize).min(grid.gh - 2);
        let ty = fy - y0 as f32;
        for (x, o) in row.iter_mut().enumerate() {
            let v = img.data[y * w + x];
            let fx = x as f32 / ss + 1.0;
            let fz = (v - lo) / sr + 1.0;
            let x0 = (fx.floor() as usize).min(grid.gw - 2);
            let z0 = (fz.floor() as usize).min(grid.gd - 2);
            let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
            let mut acc = [0.0f32; 2];
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    let base = grid.idx(x0 + dx, y0 + dy, z0);
                    let (a, b) = (grid.data[base], grid.data[base + 1]);
                    let k = wy * wx;
                    acc[0] += k * (a[0] * (1.0 - tz) + b[0] * tz);
                    acc[1] += k * (a[1] * (1.0 - tz) + b[1] * tz);
                }
            }
            *o = if acc[1] > 1e-8 { acc[0] / acc[1] } else { v };
        }
    });
    out
}

const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

fn blur_grid(g: &mut Grid) {
    let (gw, gh, gd) = (g.gw, g.gh, g.gd);
    // z: contiguous runs of gd cells
    g.data.par_chunks_mut(gd).for_each_init(
        || vec![[0.0f32; 2]; gd],
        |tmp, run| {
            tmp.copy_from_slice(run);
            for (z, o) in run.iter_mut().enumerate() {
                *o = blur_at(tmp, z, 1);
            }
        },
    );
    // x: within each gy slab, stride gd
    g.data.par_chunks_mut(gw * gd).for_each_init(
        || vec![[0.0f32; 2]; gw * gd],
        |tmp, slab| {
            tmp.copy_from_slice(slab);
            for x in 0..gw {
                for z in 0..gd {
                    slab[x * gd + z] = blur_at(&tmp[z..], x, gd);
                }
            }
        },
    );
    // y: whole slabs
    let src = g.data.clone();
    let slab = gw * gd;
    g.data
        .par_chunks_mut(slab)
        .enumerate()
        .for_each(|(y, out)| {
            for (i, o) in out.iter_mut().enumerate() {
                let mut acc = [0.0f32; 2];
                for (j, &k) in K.iter().enumerate() {
                    let yy = y as isize + j as isize - 2;
                    if yy >= 0 && (yy as usize) < gh {
                        let c = src[yy as usize * slab + i];
                        acc[0] += k * c[0];
                        acc[1] += k * c[1];
                    }
                }
                *o = acc;
            }
        });
}

/// [1 4 6 4 1] at index i of a strided sequence (zero outside).
#[inline]
fn blur_at(s: &[[f32; 2]], i: usize, stride: usize) -> [f32; 2] {
    let n = s.len().div_ceil(stride);
    let mut acc = [0.0f32; 2];
    for (j, &k) in K.iter().enumerate() {
        let t = i as isize + j as isize - 2;
        if t >= 0 && (t as usize) < n {
            let c = s[t as usize * stride];
            acc[0] += k * c[0];
            acc[1] += k * c[1];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_constant() {
        let p = Plane::new(64, 48, 0.4);
        let b = bilateral(&p, 4.0, 0.05);
        assert!(b.data.iter().all(|v| (v - 0.4).abs() < 1e-4));
    }

    #[test]
    fn smooths_noise_keeps_step() {
        // left half 0.2, right half 0.8, small noise everywhere
        let (w, h) = (128, 64);
        let mut s = 7u32;
        let p = Plane::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let n = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.04;
                    (if i % w < w / 2 { 0.2 } else { 0.8 }) + n
                })
                .collect(),
        );
        let b = bilateral(&p, 6.0, 0.1);
        let std = |q: &Plane, x0: usize, x1: usize| {
            let v: Vec<f32> = (10..54)
                .flat_map(|y| (x0..x1).map(move |x| q.get(x, y)))
                .collect();
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(
            std(&b, 10, 50) < std(&p, 10, 50) * 0.4,
            "noise not smoothed"
        );
        // the step stays sharp: neighbours across the edge keep their levels
        let (l, r) = (b.get(w / 2 - 1, 32), b.get(w / 2, 32));
        assert!(l < 0.25 && r > 0.75, "edge blurred: {l} {r}");
    }
}
