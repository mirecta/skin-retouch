#!/usr/bin/env python3
"""GIMP 3 plug-in: Filters > Enhance > Skin Retouch...

Install (symlink keeps it in sync with the repo):
    ln -s /home/miro/Prj/skinretouch/gimp/skinretouch ~/.config/GIMP/3.0/plug-ins/skinretouch
"""

import os
import sys

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
from gi.repository import Gegl, GLib, GObject, Gimp, GimpUi  # noqa: E402

# make the 'skinretouch' package from the repo importable
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

PROC = "plug-in-skin-retouch"
MODELS = os.path.join(_REPO, "..", "models")
YUNET = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
BISENET = os.path.join(MODELS, "bisenet_resnet18.onnx")

# (name, label, min, max, default)
SLIDERS = [
    ("fine", "Fine detail smoothing", 0, 100, 60),
    ("medium", "Medium detail smoothing", 0, 100, 40),
    ("large", "Large detail smoothing", 0, 100, 15),
    ("threshold", "Threshold", 0, 100, 20),
    ("uniformity", "Color uniformity", 0, 100, 20),
    ("edge-protect", "Edge protection", 0, 100, 50),
    ("healing", "Skin healing (blemishes)", 0, 100, 40),
    ("amount", "Amount", 0, 100, 100),
    ("latitude", "Skin mask latitude", 0, 100, 50),
    ("fuzziness", "Skin mask fuzziness", 0, 100, 50),
    ("feather", "Skin mask feather", 0, 100, 30),
    ("sharpness", "Sharpness", 0, 100, 0),
    ("softness", "Softness (glow)", 0, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
    ("tint", "Tint", -100, 100, 0),
    ("brightness", "Brightness", -100, 100, 0),
    ("contrast", "Contrast", -100, 100, 0),
]


def run(procedure, run_mode, image, drawables, config, data):
    if len(drawables) != 1:
        return procedure.new_return_values(
            Gimp.PDBStatusType.CALLING_ERROR,
            GLib.Error.new_literal(Gimp.PlugIn.error_quark(), "Select exactly one layer.", 0))
    drawable = drawables[0]

    if run_mode == Gimp.RunMode.INTERACTIVE:
        GimpUi.init(PROC)
        dialog = GimpUi.ProcedureDialog.new(procedure, config, "Skin Retouch")
        dialog.fill(None)
        if not dialog.run():
            dialog.destroy()
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)
        dialog.destroy()

    import numpy as np
    from skinretouch import (EnhanceParams, FaceDetector, FaceParser, MaskParams,
                             RetouchParams, process)

    v = {name: config.get_property(name) for name, *_ in SLIDERS}
    use_mask = config.get_property("use-skin-mask")
    use_ai = config.get_property("use-ai-masks")
    detector = parser = None
    if use_ai and os.path.exists(YUNET):
        detector = FaceDetector(YUNET)
        if os.path.exists(BISENET):
            parser = FaceParser(BISENET)

    ok, x, y, w, h = drawable.mask_intersect()
    if not ok or w == 0 or h == 0:
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

    Gimp.progress_init("Skin retouch…")
    has_alpha = drawable.has_alpha()
    fmt = "R'G'B'A float" if has_alpha else "R'G'B' float"
    nch = 4 if has_alpha else 3
    rect = Gegl.Rectangle.new(x, y, w, h)

    src = drawable.get_buffer()
    raw = src.get(rect, 1.0, fmt, Gegl.AbyssPolicy.CLAMP)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, nch)

    rp = RetouchParams(v["fine"], v["medium"], v["large"], v["threshold"], 0.0,
                       v["uniformity"], v["edge-protect"], v["amount"], v["healing"])
    mp = MaskParams(v["latitude"], v["fuzziness"], v["feather"])
    ep = EnhanceParams(v["sharpness"], v["softness"], v["warmth"], v["tint"],
                       v["brightness"], v["contrast"])
    out, _ = process(arr[..., :3], rp, mp, ep, detector, use_mask, False, parser)
    Gimp.progress_update(0.9)

    if has_alpha:
        out = np.dstack([out, arr[..., 3]])
    shadow = drawable.get_shadow_buffer()
    shadow.set(rect, fmt, np.ascontiguousarray(out, dtype=np.float32).tobytes())
    shadow.flush()
    drawable.merge_shadow(True)          # respects the selection and creates undo
    drawable.update(x, y, w, h)
    Gimp.displays_flush()
    Gimp.progress_end()
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)


class SkinRetouch(Gimp.PlugIn):
    def do_query_procedures(self):
        return [PROC]

    def do_set_i18n(self, name):
        return False

    def do_create_procedure(self, name):
        proc = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)
        proc.set_image_types("RGB*")
        proc.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        proc.set_menu_label("Skin Retouch…")
        proc.add_menu_path("<Image>/Filters/Enhance")
        proc.set_documentation("Portraiture-style skin smoothing",
                               "Multi-band edge-aware smoothing of skin texture with a skin-tone mask.",
                               name)
        proc.set_attribution("miro", "miro", "2026")
        for pname, label, lo, hi, default in SLIDERS:
            proc.add_double_argument(pname, label, label, lo, hi, default, GObject.ParamFlags.READWRITE)
        proc.add_boolean_argument("use-skin-mask", "Use skin-tone mask",
                                  "Only retouch skin-coloured areas", True, GObject.ParamFlags.READWRITE)
        proc.add_boolean_argument("use-ai-masks", "Use AI face masks",
                                  "Detect faces (YuNet) and parse skin/eyes/lips/hair (BiSeNet) "
                                  "if the models are in the repo's models/ folder",
                                  True, GObject.ParamFlags.READWRITE)
        return proc


Gimp.main(SkinRetouch.__gtype__, sys.argv)
