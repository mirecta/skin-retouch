//! Skin retouch desktop app: open a photo, tune the sliders on a live
//! preview, export the full-resolution result.

mod worker;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use eframe::egui::{
    self, Color32, ColorImage, Pos2, Rect, Sense, TextureHandle, TextureOptions, Vec2,
};
use skinretouch::{EnhanceParams, MaskParams, Plane, RetouchParams, Rgb, Settings};
use worker::{Cmd, Msg};

const EXTENSIONS: [&str; 6] = ["jpg", "jpeg", "png", "tif", "tiff", "webp"];

#[derive(Clone, Copy, PartialEq)]
enum View {
    Split,
    After,
    Before,
    Mask,
}

struct Doc {
    path: PathBuf,
    full_size: (usize, usize),
    before: TextureHandle,
    after: Option<TextureHandle>,
    mask: Option<TextureHandle>,
    size: Vec2,
}

struct App {
    tx: Sender<Cmd>,
    rx: Receiver<Msg>,
    settings: Settings,
    sent: Option<Settings>,
    generation: u64,
    shown_generation: u64,
    doc: Option<Doc>,
    view: View,
    split: f32,
    /// Current drag moves the split line (decided when the drag starts).
    drag_split: bool,
    zoom: f32, // 0 = fit
    pan: Vec2,
    preview_max: usize,
    status: String,
    models: String,
    busy: bool,
}

fn to_color_image(rgb: &Rgb) -> ColorImage {
    let (w, h) = (rgb.w(), rgb.h());
    let mut px = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let q = |p: &Plane| (p.data[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        px.push(Color32::from_rgb(q(&rgb.c[0]), q(&rgb.c[1]), q(&rgb.c[2])));
    }
    ColorImage {
        size: [w, h],
        pixels: px,
        source_size: Vec2::new(w as f32, h as f32),
    }
}

/// Retouched image with the mask shown as a red overlay where it is 0.
fn mask_image(rgb: &Rgb, mask: &Plane) -> ColorImage {
    let mut img = to_color_image(rgb);
    for (p, &m) in img.pixels.iter_mut().zip(&mask.data) {
        let k = 1.0 - m.clamp(0.0, 1.0);
        let mix = |c: u8, t: f32| (c as f32 * (1.0 - 0.6 * k) + t * 0.6 * k) as u8;
        *p = Color32::from_rgb(mix(p.r(), 220.0), mix(p.g(), 30.0), mix(p.b(), 60.0));
    }
    img
}

impl App {
    fn new(cc: &eframe::CreationContext, open: Option<PathBuf>) -> Self {
        let (tx, worker_rx) = channel();
        let (worker_tx, rx) = channel();
        let ctx = cc.egui_ctx.clone();
        worker::spawn(worker_rx, worker_tx, move || ctx.request_repaint());
        let mut app = Self {
            tx,
            rx,
            settings: Settings::default(),
            sent: None,
            generation: 0,
            shown_generation: 0,
            doc: None,
            view: View::Split,
            split: 0.5,
            drag_split: false,
            zoom: 0.0,
            pan: Vec2::ZERO,
            preview_max: 1600,
            status: "Open a photo (File → Open, or drop it on the window)".into(),
            models: "loading models…".into(),
            busy: false,
        };
        if let Some(p) = open {
            app.open(p);
        }
        app
    }

    fn open(&mut self, path: PathBuf) {
        self.status = format!("Loading {}…", path.display());
        self.busy = true;
        self.sent = None;
        let _ = self.tx.send(Cmd::Load {
            path,
            preview_max: self.preview_max,
        });
    }

    fn open_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &EXTENSIONS)
            .pick_file()
        {
            self.open(p);
        }
    }

    fn export_dialog(&mut self) {
        let Some(doc) = &self.doc else { return };
        let stem = doc.path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = doc
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg");
        let mut dlg = rfd::FileDialog::new()
            .set_file_name(format!("{stem}_retouched.{ext}"))
            .add_filter("Images", &EXTENSIONS);
        if let Some(dir) = doc.path.parent() {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.save_file() {
            self.status = format!("Exporting {} at full resolution…", path.display());
            self.busy = true;
            let _ = self.tx.send(Cmd::Export {
                path,
                settings: self.settings,
            });
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::ModelsReady(s) => self.models = s,
                Msg::Loaded {
                    path,
                    full_w,
                    full_h,
                    preview,
                    faces,
                    secs,
                } => {
                    let before = ctx.load_texture(
                        "before",
                        to_color_image(&preview),
                        TextureOptions::LINEAR,
                    );
                    let size = Vec2::new(preview.w() as f32, preview.h() as f32);
                    self.doc = Some(Doc {
                        path,
                        full_size: (full_w, full_h),
                        before,
                        after: None,
                        mask: None,
                        size,
                    });
                    self.zoom = 0.0;
                    self.pan = Vec2::ZERO;
                    self.status = format!("{full_w}x{full_h}, {faces} face(s) found, {secs:.1}s");
                    self.busy = false;
                    keep_arc(preview);
                }
                Msg::Rendered {
                    generation,
                    image,
                    mask,
                    secs,
                } => {
                    if generation < self.shown_generation {
                        continue;
                    }
                    self.shown_generation = generation;
                    if let Some(doc) = &mut self.doc {
                        let after = to_color_image(&image);
                        match &mut doc.after {
                            Some(t) => t.set(after, TextureOptions::LINEAR),
                            None => {
                                doc.after =
                                    Some(ctx.load_texture("after", after, TextureOptions::LINEAR))
                            }
                        }
                        let mask_img = match &mask {
                            Some(m) => mask_image(&image, m),
                            None => to_color_image(&image),
                        };
                        match &mut doc.mask {
                            Some(t) => t.set(mask_img, TextureOptions::LINEAR),
                            None => {
                                doc.mask =
                                    Some(ctx.load_texture("mask", mask_img, TextureOptions::LINEAR))
                            }
                        }
                        if !self.busy {
                            self.status = format!("preview {:.0} ms", secs * 1000.0);
                        }
                    }
                }
                Msg::Exported { path, secs } => {
                    self.busy = false;
                    self.status = format!("Saved {} ({secs:.1}s)", path.display());
                }
                Msg::Error(e) => {
                    self.busy = false;
                    self.status = format!("Error: {e}");
                }
            }
        }
    }

    fn request_render(&mut self) {
        if self.doc.is_some() && self.sent != Some(self.settings) {
            self.sent = Some(self.settings);
            self.generation += 1;
            let _ = self.tx.send(Cmd::Render {
                generation: self.generation,
                settings: self.settings,
            });
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.heading("Presets");
        ui.horizontal_wrapped(|ui| {
            for (name, (f, m, l, t, heal)) in [
                ("Natural", (40.0, 25.0, 10.0, 15.0, 30.0)),
                ("Normal", (60.0, 40.0, 15.0, 20.0, 40.0)),
                ("Medium", (75.0, 55.0, 20.0, 25.0, 50.0)),
                ("Strong", (90.0, 70.0, 30.0, 35.0, 60.0)),
            ] {
                if ui.button(name).clicked() {
                    (
                        s.retouch.fine,
                        s.retouch.medium,
                        s.retouch.large,
                        s.retouch.threshold,
                        s.retouch.healing,
                    ) = (f, m, l, t, heal);
                }
            }
            if ui.button("Reset all").clicked() {
                *s = Settings::default();
            }
        });

        let slider = |ui: &mut egui::Ui,
                      v: &mut f32,
                      r: std::ops::RangeInclusive<f32>,
                      label: &str,
                      tip: &str| {
            ui.add(egui::Slider::new(v, r).text(label))
                .on_hover_text(tip);
        };
        let rp: &mut RetouchParams = &mut s.retouch;
        egui::CollapsingHeader::new("Detail smoothing")
            .default_open(true)
            .show(ui, |ui| {
                slider(ui, &mut rp.fine, 0.0..=100.0, "Fine", "pores, noise");
                slider(
                    ui,
                    &mut rp.medium,
                    0.0..=100.0,
                    "Medium",
                    "small unevenness",
                );
                slider(
                    ui,
                    &mut rp.large,
                    0.0..=100.0,
                    "Large",
                    "large unevenness, shadows",
                );
                slider(
                    ui,
                    &mut rp.threshold,
                    0.0..=100.0,
                    "Threshold",
                    "contrast still regarded as skin texture",
                );
                slider(
                    ui,
                    &mut rp.healing,
                    0.0..=100.0,
                    "Skin healing",
                    "remove small round spots (blemishes)",
                );
                slider(
                    ui,
                    &mut rp.moles,
                    0.0..=100.0,
                    "Moles",
                    "remove moles and birthmarks (bigger dark spots)",
                );
                slider(
                    ui,
                    &mut rp.uniformity,
                    0.0..=100.0,
                    "Uniformity",
                    "even out skin colour",
                );
                slider(
                    ui,
                    &mut rp.edge_protect,
                    0.0..=100.0,
                    "Edge protect",
                    "keep eyes, lips, hairline sharp",
                );
                slider(
                    ui,
                    &mut rp.amount,
                    0.0..=100.0,
                    "Amount",
                    "blend with the original",
                );
                ui.horizontal(|ui| {
                    let mut auto = rp.portrait_size <= 0.0;
                    if ui.checkbox(&mut auto, "Auto portrait size").changed() {
                        rp.portrait_size = if auto { 0.0 } else { 400.0 };
                    }
                });
                if rp.portrait_size > 0.0 {
                    slider(
                        ui,
                        &mut rp.portrait_size,
                        50.0..=3000.0,
                        "Portrait size",
                        "face height in pixels of the preview",
                    );
                }
            });

        let mp: &mut MaskParams = &mut s.mask;
        egui::CollapsingHeader::new("Skin mask")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut s.use_mask, "Retouch skin only");
                ui.checkbox(&mut s.use_ai, "Use AI face masks")
                    .on_hover_text(self.models.clone());
                ui.checkbox(&mut s.faces_only, "Only around faces");
                slider(
                    ui,
                    &mut mp.latitude,
                    0.0..=100.0,
                    "Latitude",
                    "width of the accepted skin colour range",
                );
                slider(
                    ui,
                    &mut mp.fuzziness,
                    0.0..=100.0,
                    "Fuzziness",
                    "softness of the colour range edge",
                );
                slider(
                    ui,
                    &mut mp.feather,
                    0.0..=100.0,
                    "Feather",
                    "blur of the mask edge",
                );
            });

        let ep: &mut EnhanceParams = &mut s.enhance;
        egui::CollapsingHeader::new("Enhancements")
            .default_open(false)
            .show(ui, |ui| {
                slider(ui, &mut ep.sharpness, 0.0..=100.0, "Sharpness", "");
                slider(ui, &mut ep.softness, 0.0..=100.0, "Softness", "soft glow");
                slider(ui, &mut ep.warmth, -100.0..=100.0, "Warmth", "");
                slider(
                    ui,
                    &mut ep.tint,
                    -100.0..=100.0,
                    "Tint",
                    "+ magenta, − green",
                );
                slider(ui, &mut ep.brightness, -100.0..=100.0, "Brightness", "");
                slider(ui, &mut ep.contrast, -100.0..=100.0, "Contrast", "");
            });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Preview size");
            for v in [1200usize, 1600, 2400, 100_000] {
                let label = if v > 10_000 {
                    "full".to_string()
                } else {
                    v.to_string()
                };
                if ui.selectable_label(self.preview_max == v, label).clicked()
                    && self.preview_max != v
                {
                    self.preview_max = v;
                    if let Some(p) = self.doc.as_ref().map(|d| d.path.clone()) {
                        self.open(p);
                    }
                }
            }
        });
        ui.small(format!("AI: {}", self.models));
    }

    fn image_view(&mut self, ui: &mut egui::Ui) {
        let Some(doc) = &self.doc else {
            ui.centered_and_justified(|ui| ui.label("Drop a photo here"));
            return;
        };
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let fit = (rect.width() / doc.size.x).min(rect.height() / doc.size.y);
        let scale = if self.zoom > 0.0 { self.zoom } else { fit };

        // zoom around the cursor, pan by dragging (outside the split handle)
        if let Some(pos) = resp.hover_pos() {
            let z = ui.input(|i| i.zoom_delta() * (1.0 + i.smooth_scroll_delta().y * 0.002));
            if (z - 1.0).abs() > 1e-4 {
                let new = (scale * z).clamp(fit.min(0.05), 16.0);
                let centre = rect.center() + self.pan;
                self.pan += (pos - centre) * (1.0 - new / scale);
                self.zoom = new;
            }
        }
        let scale = if self.zoom > 0.0 { self.zoom } else { fit };
        let img_rect = Rect::from_center_size(rect.center() + self.pan, doc.size * scale);
        let split_x = img_rect.left() + img_rect.width() * self.split;
        const GRAB: f32 = 12.0;
        let near_split = self.view == View::Split
            && resp
                .hover_pos()
                .is_some_and(|p| (p.x - split_x).abs() < GRAB);
        // decide once, where the button went down: line or image
        if resp.drag_started() {
            let origin = ui.input(|i| i.pointer.press_origin());
            self.drag_split = self.view == View::Split
                && (ui.input(|i| i.modifiers.shift)
                    || origin.is_some_and(|p| (p.x - split_x).abs() < GRAB));
        }
        if resp.dragged() {
            if self.drag_split {
                if let Some(p) = resp.interact_pointer_pos() {
                    self.split = ((p.x - img_rect.left()) / img_rect.width()).clamp(0.0, 1.0);
                }
            } else {
                self.pan += resp.drag_delta();
            }
        }
        if resp.drag_stopped() {
            self.drag_split = false;
        }
        if resp.double_clicked() {
            self.zoom = if self.zoom > 0.0 { 0.0 } else { 1.0 };
            self.pan = Vec2::ZERO;
        }
        if near_split || self.drag_split {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }

        let painter = ui.painter_at(rect);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let holding_before = ui.input(|i| i.key_down(egui::Key::Space));
        let after = doc.after.as_ref().unwrap_or(&doc.before);
        let view = if holding_before {
            View::Before
        } else {
            self.view
        };
        match view {
            View::Before => {
                painter.image(doc.before.id(), img_rect, uv, Color32::WHITE);
            }
            View::After => {
                painter.image(after.id(), img_rect, uv, Color32::WHITE);
            }
            View::Mask => {
                painter.image(
                    doc.mask.as_ref().unwrap_or(after).id(),
                    img_rect,
                    uv,
                    Color32::WHITE,
                );
            }
            View::Split => {
                let t = self.split;
                let left = Rect::from_min_max(img_rect.min, Pos2::new(split_x, img_rect.max.y));
                let right = Rect::from_min_max(Pos2::new(split_x, img_rect.min.y), img_rect.max);
                painter.image(
                    doc.before.id(),
                    left,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(t, 1.0)),
                    Color32::WHITE,
                );
                painter.image(
                    after.id(),
                    right,
                    Rect::from_min_max(Pos2::new(t, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                painter.vline(
                    split_x,
                    img_rect.y_range(),
                    egui::Stroke::new(2.0, Color32::WHITE),
                );
                let font = egui::FontId::proportional(14.0);
                let y = rect.top().max(img_rect.top()) + 6.0;
                painter.text(
                    Pos2::new(split_x - 8.0, y),
                    egui::Align2::RIGHT_TOP,
                    "before",
                    font.clone(),
                    Color32::WHITE,
                );
                painter.text(
                    Pos2::new(split_x + 8.0, y),
                    egui::Align2::LEFT_TOP,
                    "after",
                    font,
                    Color32::WHITE,
                );
            }
        }
    }
}

/// The preview is owned by the worker; the UI only needed it for the texture.
fn keep_arc(_: Arc<Rgb>) {}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);

        if let Some(p) = ctx.input(|i| {
            i.raw
                .dropped_files
                .first()
                .map(|f| f.path().to_path_buf())
                .filter(|p| !p.as_os_str().is_empty())
        }) {
            self.open(p);
        }
        let (open, save) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::O),
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::S),
            )
        });
        if open {
            self.open_dialog();
        }
        if save {
            self.export_dialog();
        }

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").on_hover_text("Ctrl+O").clicked() {
                    self.open_dialog();
                }
                let can_save = self.doc.is_some() && !self.busy;
                if ui
                    .add_enabled(can_save, egui::Button::new("Export…"))
                    .on_hover_text("Ctrl+S, full resolution")
                    .clicked()
                {
                    self.export_dialog();
                }
                ui.separator();
                for (v, label) in [
                    (View::Split, "Split"),
                    (View::After, "After"),
                    (View::Before, "Before"),
                    (View::Mask, "Mask"),
                ] {
                    ui.selectable_value(&mut self.view, v, label);
                }
                ui.separator();
                if ui.button("Fit").clicked() {
                    self.zoom = 0.0;
                    self.pan = Vec2::ZERO;
                }
                if ui
                    .button("1:1")
                    .on_hover_text("1 preview pixel = 1 screen pixel")
                    .clicked()
                {
                    self.zoom = 1.0;
                    self.pan = Vec2::ZERO;
                }
                ui.weak("hold Space = before · wheel = zoom · drag = pan");
            });
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if self.busy {
                    ui.spinner();
                }
                ui.label(&self.status);
                if let Some(d) = &self.doc {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(format!(
                            "{} ({}x{})",
                            d.path.display(),
                            d.full_size.0,
                            d.full_size.1
                        ));
                    });
                }
            });
        });
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.image_view(ui));

        self.request_render();
    }
}

fn main() -> eframe::Result {
    let open = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Skin Retouch"),
        ..Default::default()
    };
    eframe::run_native(
        "Skin Retouch",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, open)))),
    )
}
