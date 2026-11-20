//! Single-channel float image and RGB image made of three planes.

use rayon::prelude::*;

/// One channel, row-major, f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize, v: f32) -> Self {
        Self {
            w,
            h,
            data: vec![v; w * h],
        }
    }

    pub fn zeros(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.0)
    }

    pub fn from_vec(w: usize, h: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), w * h);
        Self { w, h, data }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.w + x] = v;
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    pub fn same_size(&self, o: &Plane) -> bool {
        self.w == o.w && self.h == o.h
    }

    /// Element-wise map into a new plane.
    pub fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Plane {
        Plane::from_vec(
            self.w,
            self.h,
            self.data.par_iter().map(|&v| f(v)).collect(),
        )
    }

    /// Element-wise combination of two planes of the same size.
    pub fn zip(&self, o: &Plane, f: impl Fn(f32, f32) -> f32 + Sync) -> Plane {
        assert!(self.same_size(o));
        Plane::from_vec(
            self.w,
            self.h,
            self.data
                .par_iter()
                .zip(&o.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        )
    }

    pub fn zip3(&self, o: &Plane, p: &Plane, f: impl Fn(f32, f32, f32) -> f32 + Sync) -> Plane {
        assert!(self.same_size(o) && self.same_size(p));
        Plane::from_vec(
            self.w,
            self.h,
            self.data
                .par_iter()
                .zip(&o.data)
                .zip(&p.data)
                .map(|((&a, &b), &c)| f(a, b, c))
                .collect(),
        )
    }

    pub fn max(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn mean(&self) -> f32 {
        (self.data.iter().map(|&v| v as f64).sum::<f64>() / self.data.len().max(1) as f64) as f32
    }

    /// Copy of the rectangle [x0, x0+w) x [y0, y0+h) (must be inside).
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Plane {
        let mut out = Vec::with_capacity(w * h);
        for y in y0..y0 + h {
            out.extend_from_slice(&self.data[y * self.w + x0..y * self.w + x0 + w]);
        }
        Plane::from_vec(w, h, out)
    }
}

/// RGB image as three float planes, nominal range 0..1.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgb {
    pub c: [Plane; 3],
}

impl Rgb {
    pub fn w(&self) -> usize {
        self.c[0].w
    }

    pub fn h(&self) -> usize {
        self.c[0].h
    }

    pub fn new(w: usize, h: usize, rgb: [f32; 3]) -> Self {
        Self {
            c: rgb.map(|v| Plane::new(w, h, v)),
        }
    }

    /// From interleaved RGB f32 data.
    pub fn from_interleaved(w: usize, h: usize, data: &[f32]) -> Self {
        assert_eq!(data.len(), w * h * 3);
        let ch = |k: usize| Plane::from_vec(w, h, data.par_chunks_exact(3).map(|p| p[k]).collect());
        Self {
            c: [ch(0), ch(1), ch(2)],
        }
    }

    pub fn to_interleaved(&self) -> Vec<f32> {
        let [r, g, b] = &self.c;
        let mut out = vec![0.0; self.w() * self.h() * 3];
        out.par_chunks_exact_mut(3).enumerate().for_each(|(i, p)| {
            p[0] = r.data[i];
            p[1] = g.data[i];
            p[2] = b.data[i];
        });
        out
    }

    pub fn from_image(img: &image::Rgb32FImage) -> Self {
        Self::from_interleaved(img.width() as usize, img.height() as usize, img.as_raw())
    }

    pub fn to_image(&self) -> image::Rgb32FImage {
        image::Rgb32FImage::from_raw(self.w() as u32, self.h() as u32, self.to_interleaved())
            .expect("size matches")
    }

    pub fn map(&self, f: impl Fn(&Plane) -> Plane + Sync) -> Rgb {
        let (a, (b, c)) = rayon::join(
            || f(&self.c[0]),
            || rayon::join(|| f(&self.c[1]), || f(&self.c[2])),
        );
        Rgb { c: [a, b, c] }
    }

    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Rgb {
        Rgb {
            c: [0, 1, 2].map(|k| self.c[k].crop(x0, y0, w, h)),
        }
    }

    pub fn clamp01(&self) -> Rgb {
        self.map(|p| p.map(|v| v.clamp(0.0, 1.0)))
    }
}
