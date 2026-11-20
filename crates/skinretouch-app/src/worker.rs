//! Background thread: loads images, runs the networks and the retouch.
//!
//! The UI sends [`Cmd`]s and never blocks; render requests are coalesced so
//! that only the newest slider state is computed.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use skinretouch::{Analysis, Models, Plane, Rgb, Settings, analyze, io, ops, process};

pub enum Cmd {
    Load { path: PathBuf, preview_max: usize },
    Render { generation: u64, settings: Settings },
    Export { path: PathBuf, settings: Settings },
}

pub enum Msg {
    ModelsReady(String),
    Loaded {
        path: PathBuf,
        full_w: usize,
        full_h: usize,
        preview: Arc<Rgb>,
        faces: usize,
        secs: f32,
    },
    Rendered {
        generation: u64,
        image: Rgb,
        mask: Option<Plane>,
        secs: f32,
    },
    Exported {
        path: PathBuf,
        secs: f32,
    },
    Error(String),
}

struct Current {
    full: Arc<io::Image>,
    preview: Arc<Rgb>,
    analysis: Analysis,
}

pub fn spawn(rx: Receiver<Cmd>, tx: Sender<Msg>, repaint: impl Fn() + Send + 'static) {
    std::thread::spawn(move || {
        let send = |m: Msg| {
            let _ = tx.send(m);
            repaint();
        };
        let mut models = Models::find_default();
        send(Msg::ModelsReady(models.describe()));
        let mut cur: Option<Current> = None;

        while let Ok(first) = rx.recv() {
            // coalesce: keep the order of loads/exports, only the last render
            let mut queue = vec![first];
            loop {
                match rx.try_recv() {
                    Ok(c) => queue.push(c),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }
            let last_render = queue.iter().rposition(|c| matches!(c, Cmd::Render { .. }));
            for (i, cmd) in queue.into_iter().enumerate() {
                match cmd {
                    Cmd::Render { .. } if Some(i) != last_render => {}
                    Cmd::Render {
                        generation,
                        settings,
                    } => {
                        if let Some(c) = &cur {
                            let t = Instant::now();
                            let out = process(&c.preview, &c.analysis, &settings);
                            send(Msg::Rendered {
                                generation,
                                image: out.image,
                                mask: out.mask,
                                secs: t.elapsed().as_secs_f32(),
                            });
                        }
                    }
                    Cmd::Load { path, preview_max } => {
                        match load(&path, preview_max, &mut models) {
                            Ok((c, secs)) => {
                                send(Msg::Loaded {
                                    path,
                                    full_w: c.full.rgb.w(),
                                    full_h: c.full.rgb.h(),
                                    preview: c.preview.clone(),
                                    faces: c.analysis.faces.len(),
                                    secs,
                                });
                                cur = Some(c);
                            }
                            Err(e) => send(Msg::Error(format!("{e:#}"))),
                        }
                    }
                    Cmd::Export { path, settings } => {
                        let Some(c) = &cur else { continue };
                        let t = Instant::now();
                        let res = (|| -> anyhow::Result<()> {
                            let analysis = analyze(&c.full.rgb, &mut models)?;
                            let out = process(&c.full.rgb, &analysis, &settings);
                            let img = io::Image {
                                rgb: out.image,
                                high_bit_depth: c.full.high_bit_depth,
                                icc: c.full.icc.clone(),
                            };
                            io::save(&img, &path)
                        })();
                        match res {
                            Ok(()) => send(Msg::Exported {
                                path,
                                secs: t.elapsed().as_secs_f32(),
                            }),
                            Err(e) => send(Msg::Error(format!("{e:#}"))),
                        }
                    }
                }
            }
        }
    });
}

fn load(path: &PathBuf, preview_max: usize, models: &mut Models) -> anyhow::Result<(Current, f32)> {
    let t = Instant::now();
    let full = io::load(path)?;
    let (w, h) = (full.rgb.w(), full.rgb.h());
    let s = (preview_max as f32 / w.max(h) as f32).min(1.0);
    let preview = if s < 1.0 {
        let (pw, ph) = (
            ((w as f32 * s).round() as usize).max(1),
            ((h as f32 * s).round() as usize).max(1),
        );
        full.rgb.map(|p| ops::resize_area(p, pw, ph))
    } else {
        full.rgb.clone()
    };
    let analysis = analyze(&preview, models)?;
    let c = Current {
        full: Arc::new(full),
        preview: Arc::new(preview),
        analysis,
    };
    Ok((c, t.elapsed().as_secs_f32()))
}
