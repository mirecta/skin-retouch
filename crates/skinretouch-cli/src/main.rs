//! skinretouch in.jpg [-o out.jpg] [--compare] [--save-mask]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use skinretouch::{
    EnhanceParams, MaskParams, Models, RetouchParams, Settings, analyze, io, process,
};

#[derive(Parser)]
#[command(version, about = "Portraiture-style skin retouch")]
struct Args {
    input: PathBuf,
    /// Output file (default: <input>_retouched.<ext>)
    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 60.0)]
    fine: f32,
    #[arg(long, default_value_t = 40.0)]
    medium: f32,
    #[arg(long, default_value_t = 15.0)]
    large: f32,
    #[arg(long, default_value_t = 20.0)]
    threshold: f32,
    /// Face height in px; 0 = auto
    #[arg(long, default_value_t = 0.0)]
    portrait_size: f32,
    #[arg(long, default_value_t = 20.0)]
    uniformity: f32,
    #[arg(long, default_value_t = 50.0)]
    edge_protect: f32,
    #[arg(long, default_value_t = 100.0)]
    amount: f32,
    /// 0..100 remove small spots (blemishes)
    #[arg(long, default_value_t = 0.0)]
    healing: f32,
    /// 0..100 remove moles / birthmarks (bigger dark spots)
    #[arg(long, default_value_t = 0.0)]
    moles: f32,

    #[arg(long, default_value_t = 50.0)]
    latitude: f32,
    #[arg(long, default_value_t = 50.0)]
    fuzziness: f32,
    #[arg(long, default_value_t = 30.0)]
    feather: f32,
    /// Retouch the whole image, not only skin
    #[arg(long)]
    no_mask: bool,
    /// Do not use the face detection / face parsing models
    #[arg(long)]
    no_ai: bool,
    /// Limit the mask to an area around detected faces
    #[arg(long)]
    faces_only: bool,

    #[arg(long, default_value_t = 0.0)]
    sharpness: f32,
    #[arg(long, default_value_t = 0.0)]
    softness: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    warmth: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    tint: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    brightness: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    contrast: f32,

    /// Also write <name>_compare.jpg (before | after)
    #[arg(long)]
    compare: bool,
    /// Also write <name>_mask.png
    #[arg(long)]
    save_mask: bool,
}

fn sibling(base: &Path, suffix: &str, ext: &str) -> PathBuf {
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    base.with_file_name(format!("{stem}{suffix}.{ext}"))
}

fn main() -> Result<()> {
    let a = Args::parse();
    let settings = Settings {
        retouch: RetouchParams {
            fine: a.fine,
            medium: a.medium,
            large: a.large,
            threshold: a.threshold,
            portrait_size: a.portrait_size,
            uniformity: a.uniformity,
            edge_protect: a.edge_protect,
            amount: a.amount,
            healing: a.healing,
            moles: a.moles,
        },
        mask: MaskParams {
            latitude: a.latitude,
            fuzziness: a.fuzziness,
            feather: a.feather,
            ..Default::default()
        },
        enhance: EnhanceParams {
            sharpness: a.sharpness,
            softness: a.softness,
            warmth: a.warmth,
            tint: a.tint,
            brightness: a.brightness,
            contrast: a.contrast,
        },
        use_mask: !a.no_mask,
        use_ai: !a.no_ai,
        faces_only: a.faces_only,
    };

    let mut img = io::load(&a.input)?;
    let t = Instant::now();
    let mut models = if a.no_ai {
        Models::default()
    } else {
        Models::find_default()
    };
    let analysis = analyze(&img.rgb, &mut models)?;
    eprintln!(
        "{}x{}, {}, {} face(s), analysis {:.2}s",
        img.rgb.w(),
        img.rgb.h(),
        models.describe(),
        analysis.faces.len(),
        t.elapsed().as_secs_f32()
    );
    for f in &analysis.faces {
        eprintln!(
            "  face x={} y={} w={} h={} score={:.2}",
            f.x, f.y, f.w, f.h, f.score
        );
    }
    let t = Instant::now();
    let out = process(&img.rgb, &analysis, &settings);
    eprintln!("retouch {:.2}s", t.elapsed().as_secs_f32());

    let ext = a
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_string();
    let out_path = a
        .output
        .clone()
        .unwrap_or_else(|| sibling(&a.input, "_retouched", &ext));
    let base = a.output.as_deref().unwrap_or(&a.input);
    let before = std::mem::replace(&mut img.rgb, out.image);
    io::save(&img, &out_path)?;
    println!("wrote {}", out_path.display());

    if a.compare {
        let (w, h) = (before.w(), before.h());
        let mut both = skinretouch::Rgb::new(2 * w, h, [0.0; 3]);
        for c in 0..3 {
            for y in 0..h {
                let dst = &mut both.c[c].data[y * 2 * w..(y + 1) * 2 * w];
                dst[..w].copy_from_slice(before.c[c].row(y));
                dst[w..].copy_from_slice(img.rgb.c[c].row(y));
            }
        }
        let p = sibling(base, "_compare", "jpg");
        io::save(
            &io::Image {
                rgb: both,
                high_bit_depth: false,
                icc: img.icc.clone(),
            },
            &p,
        )?;
        println!("wrote {}", p.display());
    }
    if a.save_mask
        && let Some(m) = out.mask
    {
        let p = sibling(base, "_mask", "png");
        let rgb = skinretouch::Rgb {
            c: [m.clone(), m.clone(), m],
        };
        io::save(
            &io::Image {
                rgb,
                high_bit_depth: false,
                icc: None,
            },
            &p,
        )?;
        println!("wrote {}", p.display());
    }
    Ok(())
}
