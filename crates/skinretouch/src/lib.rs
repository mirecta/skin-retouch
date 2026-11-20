//! Clean-room, Portraiture-style skin retouching.
//!
//! Typical use:
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use skinretouch::{Models, Settings, analyze, io, process};
//! let mut img = io::load("portrait.jpg")?;
//! let mut models = Models::find_default();      // AI masks if the models are found
//! let analysis = analyze(&img.rgb, &mut models)?; // once per image (slow part)
//! let out = process(&img.rgb, &analysis, &Settings::default()); // per slider change
//! img.rgb = out.image;
//! io::save(&img, "portrait_retouched.jpg")?;
//! # Ok(()) }
//! ```

pub mod bilateral;
pub mod color;
pub mod core;
pub mod enhance;
pub mod io;
pub mod mask;
pub mod ops;
pub mod plane;

#[cfg(feature = "ai")]
pub mod ai;

pub use crate::core::RetouchParams;
pub use crate::enhance::EnhanceParams;
pub use crate::mask::{Face, MaskParams, ParseMaps};
pub use crate::plane::{Plane, Rgb};

/// All user-facing settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub retouch: RetouchParams,
    pub mask: MaskParams,
    pub enhance: EnhanceParams,
    /// Retouch only skin (mask); false = whole image.
    pub use_mask: bool,
    /// Use detected faces and face parsing when available.
    pub use_ai: bool,
    /// Limit the mask to an area around detected faces.
    pub faces_only: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            retouch: RetouchParams::default(),
            mask: MaskParams::default(),
            enhance: EnhanceParams::default(),
            use_mask: true,
            use_ai: true,
            faces_only: false,
        }
    }
}

/// Neural networks used by [`analyze`]; each one is optional.
#[derive(Default)]
pub struct Models {
    #[cfg(feature = "ai")]
    pub detector: Option<ai::FaceDetector>,
    #[cfg(feature = "ai")]
    pub parser: Option<ai::FaceParser>,
}

impl Models {
    /// Load the models from the default locations (see [`ai::model_dirs`]);
    /// models that are missing or fail to load are left out.
    pub fn find_default() -> Self {
        #[cfg(feature = "ai")]
        {
            let detector =
                ai::find_model(ai::YUNET_FILE).and_then(|p| ai::FaceDetector::new(p).ok());
            let parser = ai::find_model(ai::BISENET_FILE).and_then(|p| ai::FaceParser::new(p).ok());
            Self { detector, parser }
        }
        #[cfg(not(feature = "ai"))]
        Self::default()
    }

    /// Short human-readable status, e.g. "face detection + face parsing".
    pub fn describe(&self) -> String {
        #[cfg(feature = "ai")]
        {
            let det = |d: &ai::FaceDetector| match d.fixed_input() {
                Some((h, w)) => format!("face detection ({w}x{h})"),
                None => "face detection".to_string(),
            };
            match (&self.detector, &self.parser) {
                (Some(d), Some(_)) => format!("{} + face parsing", det(d)),
                (Some(d), None) => det(d),
                _ => "no AI models (colour mask only)".into(),
            }
        }
        #[cfg(not(feature = "ai"))]
        "built without AI support (colour mask only)".into()
    }
}

/// Image-dependent results that do not depend on the sliders.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub faces: Vec<Face>,
    pub parse: Option<ParseMaps>,
}

impl Analysis {
    /// Median face size in pixels (Portraiture's "Portrait Size").
    pub fn face_size(&self) -> Option<f32> {
        let mut s: Vec<f32> = self.faces.iter().map(Face::size).collect();
        if s.is_empty() {
            return None;
        }
        s.sort_by(f32::total_cmp);
        Some(s[s.len() / 2])
    }
}

/// Run the neural networks (face detection, face parsing) on an image.
#[allow(unused_variables, unused_mut)]
pub fn analyze(rgb: &Rgb, models: &mut Models) -> anyhow::Result<Analysis> {
    let mut a = Analysis::default();
    #[cfg(feature = "ai")]
    {
        if let Some(det) = models.detector.as_mut() {
            a.faces = det.detect(rgb)?;
        }
        if let Some(parser) = models.parser.as_mut()
            && !a.faces.is_empty()
        {
            a.parse = Some(parser.parse(rgb, &a.faces)?);
        }
    }
    Ok(a)
}

/// Result of [`process`].
pub struct Output {
    pub image: Rgb,
    /// Retouch mask (None when the whole image was retouched).
    pub mask: Option<Plane>,
}

/// Full pipeline: masks -> smoothing + healing -> enhancements.
pub fn process(rgb: &Rgb, analysis: &Analysis, s: &Settings) -> Output {
    let empty = Analysis::default();
    let a = if s.use_ai { analysis } else { &empty };
    let masks = s
        .use_mask
        .then(|| mask::build_mask(rgb, &s.mask, &a.faces, a.parse.as_ref(), s.faces_only));
    let mut rp = s.retouch;
    if rp.portrait_size <= 0.0
        && let Some(size) = a.face_size()
    {
        rp.portrait_size = size;
    }
    let out = core::retouch(
        rgb,
        &rp,
        masks.as_ref().map(|m| &m.retouch),
        masks.as_ref().map(|m| &m.heal),
    );
    let mask = masks.map(|m| m.retouch);
    let image = enhance::enhance(&out, &s.enhance, mask.as_ref());
    Output { image, mask }
}
