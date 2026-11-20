//! Synthetic checks: skin texture is removed, strong edges survive,
//! blemishes are healed. Same scenarios as python/tests/test_core.py.

use skinretouch::{Analysis, Plane, RetouchParams, Rgb, Settings, ops, process};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.next(), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// 512x512 skin patch: noise (pores), 25 darker blemishes, a dark "eye".
fn synthetic() -> Rgb {
    let (w, h) = (512, 512);
    let mut rng = Rng(42);
    let mut img = Rgb::new(w, h, [0.85, 0.65, 0.55]);
    for i in 0..w * h {
        let n = rng.normal() * 0.03;
        for c in 0..3 {
            img.c[c].data[i] += n;
        }
    }
    for _ in 0..25 {
        let x = 20.0 + rng.next() * (w - 40) as f32;
        let y = 20.0 + rng.next() * (h - 40) as f32;
        let r = 3.0 + (rng.next() * 4.0).floor();
        for (c, v) in [0.75, 0.5, 0.45].into_iter().enumerate() {
            let mut disc = Plane::zeros(w, h);
            ops::fill_ellipse(&mut disc, x.round(), y.round(), r, r, 1.0);
            img.c[c] = img.c[c].zip(&disc, |a, d| if d > 0.0 { v } else { a });
        }
    }
    let mut img = img.map(|p| ops::gaussian_blur(p, 0.7));
    for (c, v) in [0.15, 0.1, 0.08].into_iter().enumerate() {
        for y in 200..260 {
            for x in 300..420 {
                img.c[c].set(x, y, v);
            }
        }
    }
    img.clamp01()
}

/// Mean over channels of the standard deviation in a rectangle.
fn std_rect(img: &Rgb, x0: usize, x1: usize, y0: usize, y1: usize) -> f32 {
    let mut total = 0.0;
    for p in &img.c {
        let v: Vec<f32> = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| p.get(x, y)))
            .collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        total += (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
    }
    total / 3.0
}

/// Mean over channels of the peak-to-peak along row y, columns x0..x1.
fn edge(img: &Rgb, y: usize, x0: usize, x1: usize) -> f32 {
    img.c
        .iter()
        .map(|p| {
            let v = (x0..x1).map(|x| p.get(x, y));
            v.clone().fold(f32::MIN, f32::max) - v.fold(f32::MAX, f32::min)
        })
        .sum::<f32>()
        / 3.0
}

fn mean_rect(p: &Plane, x0: usize, x1: usize, y0: usize, y1: usize) -> f32 {
    let s: f32 = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| p.get(x, y)))
        .sum();
    s / ((x1 - x0) * (y1 - y0)) as f32
}

fn run(img: &Rgb, rp: RetouchParams) -> skinretouch::Output {
    let s = Settings {
        retouch: rp,
        ..Default::default()
    };
    process(img, &Analysis::default(), &s)
}

#[test]
fn texture_removed_edges_kept() {
    let img = synthetic();
    let out = run(
        &img,
        RetouchParams {
            fine: 80.0,
            medium: 60.0,
            large: 20.0,
            threshold: 30.0,
            ..Default::default()
        },
    );
    let mask = out.mask.unwrap();
    let (s_in, s_out) = (
        std_rect(&img, 20, 280, 20, 180),
        std_rect(&out.image, 20, 280, 20, 180),
    );
    let (e_in, e_out) = (edge(&img, 230, 290, 310), edge(&out.image, 230, 290, 310));
    let (m_skin, m_eye) = (
        mean_rect(&mask, 20, 280, 20, 180),
        mean_rect(&mask, 330, 390, 215, 245),
    );
    println!(
        "flat std {s_in:.4} -> {s_out:.4}; edge {e_in:.3} -> {e_out:.3}; mask skin {m_skin:.2} eye {m_eye:.2}"
    );
    assert!(s_out < s_in * 0.6, "skin texture not reduced enough");
    assert!(e_out > e_in * 0.9, "edge was blurred");
    assert!(m_skin > 0.8 && m_eye < 0.2, "bad mask");
}

#[test]
fn healing_removes_blemishes() {
    let img = synthetic();
    let base = run(
        &img,
        RetouchParams {
            threshold: 30.0,
            ..Default::default()
        },
    )
    .image;
    let healed = run(
        &img,
        RetouchParams {
            threshold: 30.0,
            healing: 70.0,
            ..Default::default()
        },
    )
    .image;
    let (s0, s1) = (
        std_rect(&base, 20, 280, 20, 180),
        std_rect(&healed, 20, 280, 20, 180),
    );
    let e = edge(&healed, 230, 290, 310);
    println!("healing: flat std {s0:.4} -> {s1:.4}, eye edge {e:.3}");
    assert!(s1 < s0 * 0.8, "blemishes not healed");
    assert!(e > 0.5, "eye edge damaged");
}

#[test]
fn amount_zero_is_identity() {
    let img = synthetic();
    let out = run(
        &img,
        RetouchParams {
            amount: 0.0,
            ..Default::default()
        },
    )
    .image;
    let d = img.c[0].zip(&out.c[0], |a, b| (a - b).abs()).max();
    assert!(d < 1e-5);
}

/// Flat skin with pores, a dark mole (r=10) and a red mark (r=6).
fn skin_with_mole_and_red_mark() -> Rgb {
    let (w, h) = (400, 300);
    let mut rng = Rng(9);
    let mut img = Rgb::new(w, h, [0.85, 0.65, 0.55]);
    for i in 0..w * h {
        let n = rng.normal() * 0.02;
        for c in 0..3 {
            img.c[c].data[i] += n;
        }
    }
    let mut paint = |cx: f32, cy: f32, r: f32, rgb: [f32; 3]| {
        let mut disc = Plane::zeros(w, h);
        ops::fill_circle_aa(&mut disc, cx, cy, r, 1.0);
        let disc = ops::gaussian_blur(&disc, 1.5);
        for (c, v) in rgb.into_iter().enumerate() {
            img.c[c] = img.c[c].zip(&disc, |a, d| a + (v - a) * d);
        }
    };
    paint(100.0, 150.0, 10.0, [0.45, 0.30, 0.25]); // mole
    paint(300.0, 150.0, 6.0, [0.92, 0.55, 0.50]); // red mark
    img.clamp01()
}

/// Mean |difference| to the clean skin colour in a disc.
fn spot_left(img: &Rgb, cx: usize, cy: usize, r: usize) -> f32 {
    let base = [0.85, 0.65, 0.55];
    let mut s = 0.0;
    let mut n = 0;
    for y in cy - r..=cy + r {
        for x in cx - r..=cx + r {
            if (x as f32 - cx as f32).hypot(y as f32 - cy as f32) <= r as f32 {
                s += (0..3)
                    .map(|c| (img.c[c].get(x, y) - base[c]).abs())
                    .sum::<f32>()
                    / 3.0;
                n += 1;
            }
        }
    }
    s / n as f32
}

#[test]
fn healing_red_marks_and_moles() {
    let img = skin_with_mole_and_red_mark();
    let rp = |healing, moles| RetouchParams {
        healing,
        moles,
        portrait_size: 250.0,
        ..Default::default()
    };
    let s = |rp| Settings {
        retouch: rp,
        use_mask: false,
        ..Default::default()
    };
    let out = |h, m| process(&img, &Analysis::default(), &s(rp(h, m))).image;
    let (none, heal, both) = (out(0.0, 0.0), out(60.0, 0.0), out(60.0, 60.0));
    let (mole0, red0) = (spot_left(&none, 100, 150, 6), spot_left(&none, 300, 150, 4));
    let (mole1, red1) = (spot_left(&heal, 100, 150, 6), spot_left(&heal, 300, 150, 4));
    let (mole2, red2) = (spot_left(&both, 100, 150, 6), spot_left(&both, 300, 150, 4));
    println!(
        "mole {mole0:.3} -> heal {mole1:.3} -> moles {mole2:.3}; red {red0:.3} -> {red1:.3} -> {red2:.3}"
    );
    assert!(red1 < red0 * 0.5, "red mark not healed");
    assert!(mole1 > mole0 * 0.7, "healing alone should keep the mole");
    assert!(mole2 < mole0 * 0.35, "mole not removed");
}
