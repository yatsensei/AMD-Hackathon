"""Classifiers (spec 5.5). Every CNN exposes forward(x) -> logits and a `cam_layer` module
whose output is >= 16x16 for a 64x64 input (used by Grad-CAM, spec 8)."""
