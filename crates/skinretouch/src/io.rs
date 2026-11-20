//! Loading and saving photos (EXIF orientation, ICC profile, 8/16 bit).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tiff::TiffEncoder;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader};

use crate::plane::Rgb;

pub struct Image {
    pub rgb: Rgb,
    /// Source had more than 8 bits per channel.
    pub high_bit_depth: bool,
    /// Embedded colour profile, written back on save.
    pub icc: Option<Vec<u8>>,
}

pub fn load(path: impl AsRef<Path>) -> Result<Image> {
    let path = path.as_ref();
    let mut dec = ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("unsupported image {}", path.display()))?;
    let orientation = dec.orientation().ok();
    let icc = dec.icc_profile().ok().flatten();
    let mut img = DynamicImage::from_decoder(dec)
        .with_context(|| format!("cannot decode {}", path.display()))?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    let high_bit_depth = img.color().bytes_per_pixel() / img.color().channel_count() > 1;
    Ok(Image {
        rgb: Rgb::from_image(&img.into_rgb32f()),
        high_bit_depth,
        icc,
    })
}

/// Save by extension: JPEG (quality 95, 8 bit), PNG / TIFF (16 bit when the
/// source had more than 8 bits).
pub fn save(img: &Image, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let f32img = DynamicImage::ImageRgb32F(img.rgb.clamp01().to_image());
    let out = || -> Result<BufWriter<File>> {
        Ok(BufWriter::new(File::create(path).with_context(|| {
            format!("cannot write {}", path.display())
        })?))
    };
    let deep = img.high_bit_depth;
    let pixels = |deep: bool| {
        if deep {
            DynamicImage::ImageRgb16(f32img.to_rgb16())
        } else {
            DynamicImage::ImageRgb8(f32img.to_rgb8())
        }
    };
    match ext.as_str() {
        "jpg" | "jpeg" => {
            let mut enc = JpegEncoder::new_with_quality(out()?, 95);
            if let Some(icc) = &img.icc {
                let _ = enc.set_icc_profile(icc.clone());
            }
            pixels(false).write_with_encoder(enc)?;
        }
        "png" => {
            let mut enc = PngEncoder::new(out()?);
            if let Some(icc) = &img.icc {
                let _ = enc.set_icc_profile(icc.clone());
            }
            pixels(deep).write_with_encoder(enc)?;
        }
        "tif" | "tiff" => pixels(deep).write_with_encoder(TiffEncoder::new(out()?))?,
        _ => pixels(false)
            .save(path)
            .with_context(|| format!("cannot write {}", path.display()))?,
    }
    Ok(())
}
