#!/usr/bin/env python3
"""
image_similarity_analysis.py
─────────────────────────────────────────────────────────────────────────────
Forensic Image Similarity Analysis Suite
C. R. Johnson  |  Companion tool for Deepfake and Image Forgery Detection

Computes the following similarity / identity measures for every image pair
in a target directory and writes a structured plain-text report:

  Hashing
    • MD5  / SHA-256          cryptographic (chain-of-custody / fixity)
    • aHash                   average hash
    • dHash                   difference hash
    • pHash                   perceptual hash (DCT-based)  ← primary
    • wHash                   wavelet hash

  Pixel-level metrics (decoded raster)
    • PSNR                    Peak Signal-to-Noise Ratio (dB)
    • SSIM                    Structural Similarity Index (luminance)
    • MS-SSIM                 Multi-Scale SSIM

  Neural perceptual metric
    • LPIPS                   Learned Perceptual Image Patch Similarity
                              (AlexNet backbone; lower = more similar)

  Forensic classification (per your book's taxonomy)
    Pixel-Identical | Raster-Equivalent | Display-Equivalent |
    Content-Equivalent | Content-Divergent

─────────────────────────────────────────────────────────────────────────────
USAGE
    python image_similarity_analysis.py /path/to/image/directory

OUTPUT
    image_similarity_report.txt  (written to the same directory)

INSTALL DEPENDENCIES
    pip install imagehash Pillow scikit-image numpy torch torchvision lpips

NOTES
  • LPIPS requires PyTorch.  If unavailable the script degrades gracefully
    and marks LPIPS as "unavailable".
  • All images are decoded to RGB before comparison; format differences
    (JPEG vs PNG vs HEIC) do not affect the decoded-raster metrics.
  • Hamming distance thresholds follow common forensic practice:
      0        → identical hash
      1–10     → likely visually equivalent (display-equivalent zone)
      11–19    → probable visible differences (content-equivalent zone)
      ≥ 20     → content-divergent
  • SSIM is used as a gate at BOTH the Display-Equivalent AND Content-Equivalent
    boundaries. A low SSIM overrides a low Hamming distance:
      SSIM ≥ 0.95  required for Display-Equivalent
      SSIM ≥ 0.75  required for Content-Equivalent (below → Divergent)
  • PSNR ≥ 40 dB is treated as perceptually lossless for 8-bit imagery.
  • PSNR divide-by-zero on pixel-identical images is handled gracefully.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import hashlib
import itertools
import traceback
from datetime import datetime
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
try:
    from PIL import Image
    import numpy as np
    import imagehash
    from skimage.metrics import (
        peak_signal_noise_ratio as psnr,
        structural_similarity as ssim,
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"[FATAL] Missing core dependency: {e}")
    print("Run:  pip install imagehash Pillow scikit-image numpy")
    sys.exit(1)

try:
    import torch
    import lpips
    _lpips_fn = lpips.LPIPS(net="alex", verbose=False)
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

# ── configurable thresholds ──────────────────────────────────────────────────
PHASH_DISPLAY_EQUIV_THRESHOLD  = 10   # Hamming distance (0-64)
PHASH_CONTENT_EQUIV_THRESHOLD  = 19   # strictly < 20; == 20 → Divergent
SSIM_DISPLAY_EQUIV_THRESHOLD   = 0.95
SSIM_CONTENT_EQUIV_FLOOR       = 0.75 # below this → Divergent regardless of pHash
PSNR_LOSSLESS_THRESHOLD        = 40.0 # dB

# ── supported extensions ─────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".gif", ".webp", ".heic", ".heif",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def crypto_hash(filepath: Path) -> dict:
    """MD5 and SHA-256 of raw file bytes (chain-of-custody hashes)."""
    md5  = hashlib.md5()
    sha  = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def raster_hash(img_rgb: np.ndarray) -> str:
    """SHA-256 of the decoded RGB raster (format-agnostic identity check)."""
    return hashlib.sha256(img_rgb.tobytes()).hexdigest()


def perceptual_hashes(pil_img: Image.Image) -> dict:
    """All four imagehash algorithms."""
    return {
        "ahash": imagehash.average_hash(pil_img),
        "dhash": imagehash.dhash(pil_img),
        "phash": imagehash.phash(pil_img),
        "whash": imagehash.whash(pil_img),
    }


def hamming(h1, h2) -> int:
    return h1 - h2          # imagehash overloads subtraction → hamming dist


def load_image(filepath: Path) -> tuple:
    """
    Returns (PIL.Image in RGB, np.ndarray uint8 HxWx3).
    Raises on failure.
    """
    pil = Image.open(filepath).convert("RGB")
    arr = np.array(pil)
    return pil, arr


def resize_to_match(arr1: np.ndarray, arr2: np.ndarray) -> tuple:
    """
    Resize arr2 to arr1's spatial dimensions if they differ.
    Returns (arr1, arr2_resized, was_resized: bool).
    """
    if arr1.shape == arr2.shape:
        return arr1, arr2, False
    h, w = arr1.shape[:2]
    pil2 = Image.fromarray(arr2).resize((w, h), Image.LANCZOS)
    return arr1, np.array(pil2), True


def compute_psnr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    PSNR in dB. Returns float('inf') when arrays are pixel-identical,
    avoiding the divide-by-zero RuntimeWarning from scikit-image.
    """
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return psnr(arr1, arr2, data_range=255)


def compute_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Mean SSIM across luminance channel (converted to grayscale)."""
    from skimage.color import rgb2gray
    g1 = rgb2gray(arr1)
    g2 = rgb2gray(arr2)
    return ssim(g1, g2, data_range=1.0)


def compute_ms_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Multi-Scale SSIM via pytorch-msssim if available,
    otherwise degrades to single-scale SSIM with a note.
    """
    try:
        from pytorch_msssim import ms_ssim
        import torch
        t1 = torch.from_numpy(arr1).permute(2,0,1).unsqueeze(0).float() / 255.
        t2 = torch.from_numpy(arr2).permute(2,0,1).unsqueeze(0).float() / 255.
        return ms_ssim(t1, t2, data_range=1.0, size_average=True).item()
    except Exception:
        # Fall back to skimage single-scale with a flag
        return None   # caller checks for None → labels as "single-scale fallback"


def compute_lpips(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """LPIPS (AlexNet). Lower = more similar. Range ≈ 0–1."""
    if not LPIPS_AVAILABLE:
        return None
    import torch
    def to_tensor(arr):
        t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        return t
    with torch.no_grad():
        d = _lpips_fn(to_tensor(arr1), to_tensor(arr2))
    return float(d.squeeze())


# ─────────────────────────────────────────────────────────────────────────────
# Forensic classification (book taxonomy)
# ─────────────────────────────────────────────────────────────────────────────

def classify(
    file_hashes_match: bool,
    raster_hashes_match: bool,
    phash_dist: int,
    ssim_score: float,
) -> tuple:
    """
    Returns (classification_label, explanation) per the taxonomy in
    Deepfake and Image Forgery Detection (Johnson).

    Decision logic (in priority order):
      1. Pixel-Identical      — file SHA-256 hashes match
      2. Raster-Equivalent    — decoded raster hashes match; file hashes differ
      3. Display-Equivalent   — pHash ≤ DISPLAY threshold AND SSIM ≥ 0.95
      4. Content-Equivalent   — pHash < 20 AND SSIM ≥ 0.75
                                SSIM below 0.75 overrides a low pHash and
                                pushes the result to Content-Divergent. This
                                prevents localized but substantial alterations
                                from being masked by a globally stable pHash.
      5. Content-Divergent    — everything else
    """
    if file_hashes_match:
        return (
            "PIXEL-IDENTICAL",
            "File hashes match — bitwise duplicate. "
            "Container, metadata, and pixel data are all identical."
        )
    if raster_hashes_match:
        return (
            "RASTER-EQUIVALENT",
            "Decoded rasters are pixel-identical; file hashes differ. "
            "Likely a lossless container/metadata change (e.g., EXIF rewrite, "
            "lossless rotation, format conversion with no re-encode)."
        )
    if (phash_dist <= PHASH_DISPLAY_EQUIV_THRESHOLD
            and ssim_score >= SSIM_DISPLAY_EQUIV_THRESHOLD):
        return (
            "DISPLAY-EQUIVALENT",
            f"pHash Hamming={phash_dist} (≤{PHASH_DISPLAY_EQUIV_THRESHOLD}), "
            f"SSIM={ssim_score:.4f} (≥{SSIM_DISPLAY_EQUIV_THRESHOLD}). "
            "Perceptually indistinguishable under defined tolerances. "
            "Likely a lossy transcode, platform recompression, or minor resize."
        )
    # SSIM floor gate: a low SSIM overrides a low pHash distance.
    # This catches localized but visually significant alterations (splice,
    # clone-stamp, inpainting) where the global pHash is not disturbed enough
    # to exceed the Divergent threshold on its own.
    if ssim_score < SSIM_CONTENT_EQUIV_FLOOR:
        return (
            "CONTENT-DIVERGENT",
            f"SSIM={ssim_score:.4f} is below the Content-Equivalent floor "
            f"({SSIM_CONTENT_EQUIV_FLOOR}), indicating substantial pixel-level "
            f"differences despite pHash Hamming={phash_dist}. "
            "This pattern is consistent with a significant localized alteration "
            "(splice, inpainting, clone-stamp, or composite element) whose "
            "global hash signature was not large enough to exceed the pHash "
            "threshold alone. Recommend tile-level analysis."
        )
    if phash_dist <= PHASH_CONTENT_EQUIV_THRESHOLD:
        return (
            "CONTENT-EQUIVALENT",
            f"pHash Hamming={phash_dist} suggests same underlying scene "
            f"with detectable encoding differences (SSIM={ssim_score:.4f}). "
            "May be a crop, color-graded derivative, or aggressive recompression."
        )
    return (
        "CONTENT-DIVERGENT",
        f"pHash Hamming={phash_dist} — images do not share the same visual "
        f"content (SSIM={ssim_score:.4f}). "
        "Treat as separate images for lineage purposes."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-image section
# ─────────────────────────────────────────────────────────────────────────────

def analyze_single(filepath: Path) -> dict:
    """Compute all single-image metrics."""
    result = {
        "path":     filepath,
        "filename": filepath.name,
        "size_bytes": filepath.stat().st_size,
        "error":    None,
    }
    try:
        pil, arr = load_image(filepath)
        result["dimensions"]   = f"{pil.width} x {pil.height} px"
        result["mode"]         = pil.mode
        result["format"]       = pil.format or filepath.suffix.upper().lstrip(".")
        result["crypto"]       = crypto_hash(filepath)
        result["raster_sha256"]= raster_hash(arr)
        result["phashes"]      = perceptual_hashes(pil)
        result["arr"]          = arr
        result["pil"]          = pil
    except Exception as e:
        result["error"] = str(e)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_pair(r1: dict, r2: dict) -> dict:
    """All pairwise metrics between two analyzed images."""
    out = {
        "file_a": r1["filename"],
        "file_b": r2["filename"],
        "error":  None,
    }

    try:
        # ── hashing ──────────────────────────────────────────────────────────
        out["file_hashes_match"]   = r1["crypto"]["sha256"] == r2["crypto"]["sha256"]
        out["raster_hashes_match"] = r1["raster_sha256"]    == r2["raster_sha256"]

        # ── perceptual hamming distances ─────────────────────────────────────
        ph1, ph2 = r1["phashes"], r2["phashes"]
        out["ahash_dist"] = hamming(ph1["ahash"], ph2["ahash"])
        out["dhash_dist"] = hamming(ph1["dhash"], ph2["dhash"])
        out["phash_dist"] = hamming(ph1["phash"], ph2["phash"])
        out["whash_dist"] = hamming(ph1["whash"], ph2["whash"])

        # ── pixel metrics (resize if needed) ────────────────────────────────
        a1, a2, resized = resize_to_match(r1["arr"], r2["arr"])
        out["resized_for_comparison"] = resized

        out["psnr_db"]   = compute_psnr(a1, a2)
        out["ssim"]      = compute_ssim(a1, a2)

        ms = compute_ms_ssim(a1, a2)
        out["ms_ssim"]        = ms
        out["ms_ssim_fallback"] = (ms is None)
        if ms is None:
            out["ms_ssim"] = out["ssim"]   # use single-scale as fallback value

        out["lpips"]     = compute_lpips(a1, a2)

        # ── taxonomy classification ──────────────────────────────────────────
        label, explanation = classify(
            out["file_hashes_match"],
            out["raster_hashes_match"],
            out["phash_dist"],
            out["ssim"],
        )
        out["classification"] = label
        out["classification_note"] = explanation

    except Exception as e:
        out["error"] = traceback.format_exc()

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

SEP_MAJOR = "=" * 80
SEP_MINOR = "-" * 80
SEP_THIN  = "·" * 80

def fmt_bool(val: bool) -> str:
    return "YES" if val else "no"

def fmt_float(val, decimals=6) -> str:
    if val is None:
        return "unavailable"
    if val == float("inf"):
        return "∞ (identical)"
    return f"{val:.{decimals}f}"

def fmt_hamming(dist: int) -> str:
    if dist == 0:
        return f"{dist}  [identical]"
    if dist <= PHASH_DISPLAY_EQUIV_THRESHOLD:
        return f"{dist}  [display-equivalent zone]"
    if dist <= 20:
        return f"{dist}  [content-equivalent zone]"
    return f"{dist}  [content-divergent]"


def write_report(
    directory: Path,
    singles: list,
    pairs: list,
    output_path: Path,
) -> None:

    lines = []
    a = lines.append   # shorthand

    # ── header ───────────────────────────────────────────────────────────────
    a(SEP_MAJOR)
    a("FORENSIC IMAGE SIMILARITY ANALYSIS REPORT")
    a("Companion tool — Deepfake and Image Forgery Detection (C. R. Johnson)")
    a(SEP_MAJOR)
    a(f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"Directory   : {directory.resolve()}")
    a(f"Images found: {len(singles)}")
    a(f"Pairs compared: {len(pairs)}")
    a(f"LPIPS       : {'available (AlexNet)' if LPIPS_AVAILABLE else 'unavailable — install torch + lpips'}")
    a("")
    a("THRESHOLDS USED")
    a(f"  pHash Hamming ≤ {PHASH_DISPLAY_EQUIV_THRESHOLD}   → display-equivalent zone")
    a(f"  pHash Hamming ≤ {PHASH_CONTENT_EQUIV_THRESHOLD}   → content-equivalent zone (if SSIM floor met)")
    a(f"  SSIM          ≥ {SSIM_DISPLAY_EQUIV_THRESHOLD}   → display-equivalent zone")
    a(f"  SSIM          ≥ {SSIM_CONTENT_EQUIV_FLOOR}   → content-equivalent floor (below → divergent)")
    a(f"  PSNR          ≥ {PSNR_LOSSLESS_THRESHOLD} dB → perceptually lossless")
    a("")

    # ── taxonomy reference ────────────────────────────────────────────────────
    a(SEP_MINOR)
    a("FORENSIC CLASSIFICATION TAXONOMY  (Johnson, Deepfake and Image Forgery Detection)")
    a(SEP_MINOR)
    a("  PIXEL-IDENTICAL      File SHA-256 hashes match. Bitwise duplicate.")
    a("  RASTER-EQUIVALENT    Decoded rasters match; file hashes differ.")
    a("                       Container/metadata changed; pixels untouched.")
    a("  DISPLAY-EQUIVALENT   Perceptually indistinguishable within thresholds.")
    a("                       pHash Hamming ≤ 10 AND SSIM ≥ 0.95.")
    a("                       Likely platform transcode, minor resize, or")
    a("                       lossy recompression.")
    a("  CONTENT-EQUIVALENT   Same scene; encoding differences detectable.")
    a("                       pHash Hamming ≤ 19 AND SSIM ≥ 0.75.")
    a("                       May be crop, color-grade, or aggressive compression.")
    a("  CONTENT-DIVERGENT    Different visual content, OR SSIM < 0.75 regardless")
    a("                       of pHash distance. A low SSIM with globally stable")
    a("                       pHash is a strong indicator of localized alteration.")
    a("")
    a("  ⚠  SSIM FLOOR OVERRIDE")
    a("     If SSIM falls below 0.75, the result is forced to CONTENT-DIVERGENT")
    a("     even when pHash Hamming is low. This prevents localized but visually")
    a("     significant alterations (splice, inpainting, clone-stamp, composite)")
    a("     from hiding behind a globally stable perceptual hash.")
    a("     When this override fires, tile-level analysis is recommended.")
    a("")
    a("INTERPRETIVE NOTE")
    a("  Perceptual similarity ≠ forensic identity. A display-equivalent pair")
    a("  may have completely different PRNU signatures and compression lineages.")
    a("  Perceptual hashes are discovery/clustering tools; cryptographic hashes")
    a("  are chain-of-custody tools. They answer different questions.")
    a("")

    # ── per-image section ─────────────────────────────────────────────────────
    a(SEP_MAJOR)
    a("SECTION 1 — PER-IMAGE FINGERPRINTS")
    a(SEP_MAJOR)

    for r in singles:
        a("")
        a(SEP_MINOR)
        a(f"FILE: {r['filename']}")
        a(SEP_MINOR)
        if r.get("error"):
            a(f"  ERROR loading image: {r['error']}")
            a("")
            continue

        a(f"  Path          : {r['path']}")
        a(f"  Size          : {r['size_bytes']:,} bytes")
        a(f"  Dimensions    : {r['dimensions']}")
        a(f"  Color mode    : {r['mode']}")
        a(f"  Format        : {r['format']}")
        a("")
        a("  CRYPTOGRAPHIC HASHES  (raw file bytes)")
        a(f"    MD5         : {r['crypto']['md5']}")
        a(f"    SHA-256     : {r['crypto']['sha256']}")
        a("")
        a("  DECODED RASTER HASH   (format-agnostic identity)")
        a(f"    SHA-256     : {r['raster_sha256']}")
        a("")
        a("  PERCEPTUAL HASHES     (64-bit fingerprints)")
        ph = r["phashes"]
        a(f"    aHash       : {ph['ahash']}")
        a(f"    dHash       : {ph['dhash']}")
        a(f"    pHash       : {ph['phash']}  ← primary forensic hash")
        a(f"    wHash       : {ph['whash']}")
        a("")

    # ── pairwise section ──────────────────────────────────────────────────────
    a(SEP_MAJOR)
    a("SECTION 2 — PAIRWISE COMPARISONS")
    a(SEP_MAJOR)

    for c in pairs:
        a("")
        a(SEP_MINOR)
        a(f"COMPARISON: {c['file_a']}  ↔  {c['file_b']}")
        a(SEP_MINOR)

        if c.get("error"):
            a(f"  ERROR during comparison:\n{c['error']}")
            a("")
            continue

        # Classification — lead result
        a("")
        a(f"  ┌─ FORENSIC CLASSIFICATION ─────────────────────────────────┐")
        a(f"  │  {c['classification']:<56}│")
        a(f"  └───────────────────────────────────────────────────────────┘")
        a(f"  {c['classification_note']}")
        a("")

        # Hash comparison
        a("  HASH COMPARISON")
        a(f"    File SHA-256 match    : {fmt_bool(c['file_hashes_match'])}")
        a(f"    Raster SHA-256 match  : {fmt_bool(c['raster_hashes_match'])}")
        a("")

        # Perceptual hashing
        a("  PERCEPTUAL HASH HAMMING DISTANCES  (0 = identical, 64 = max)")
        a(f"    aHash   : {fmt_hamming(c['ahash_dist'])}")
        a(f"    dHash   : {fmt_hamming(c['dhash_dist'])}")
        a(f"    pHash   : {fmt_hamming(c['phash_dist'])}  ← primary")
        a(f"    wHash   : {fmt_hamming(c['whash_dist'])}")
        a("")

        # Pixel metrics
        if c["resized_for_comparison"]:
            a("  NOTE: Images differ in dimensions. B was resized to match A")
            a("        before pixel-level metric computation.")
            a("")

        psnr_note = ""
        if c["psnr_db"] == float("inf"):
            psnr_note = " (pixels are identical after decode)"
        elif c["psnr_db"] >= PSNR_LOSSLESS_THRESHOLD:
            psnr_note = " (≥ lossless threshold — perceptually indistinguishable)"

        ms_note = " [fallback: single-scale SSIM]" if c.get("ms_ssim_fallback") else ""

        a("  PIXEL-LEVEL SIMILARITY METRICS")
        a(f"    PSNR          : {fmt_float(c['psnr_db'], 2)} dB{psnr_note}")
        a(f"    SSIM          : {fmt_float(c['ssim'], 6)}  (1.0 = identical)")
        a(f"    MS-SSIM       : {fmt_float(c['ms_ssim'], 6)}{ms_note}")
        a(f"    LPIPS         : {fmt_float(c['lpips'], 6)}  (0.0 = identical) "
          f"{'[unavailable]' if c['lpips'] is None else ''}")
        a("")

        # Plain-language summary
        a("  PLAIN-LANGUAGE SUMMARY")
        if c["classification"] == "PIXEL-IDENTICAL":
            a("    These files are byte-for-byte identical. They are the same")
            a("    file. No further similarity analysis is meaningful.")
        elif c["classification"] == "RASTER-EQUIVALENT":
            a("    The decoded image pixels are identical, but the files differ")
            a("    at the container or metadata level. This is consistent with")
            a("    a lossless operation such as EXIF rewrite, orientation flip,")
            a("    or format conversion without re-encoding.")
        elif c["classification"] == "DISPLAY-EQUIVALENT":
            a("    These images are perceptually indistinguishable to a human")
            a("    observer under normal viewing conditions. The differences are")
            a("    consistent with lossy compression, platform transcoding, or")
            a("    minor resizing. They are NOT the same file and NOT the same")
            a("    decoded raster. Each has an independent forensic identity.")
        elif c["classification"] == "CONTENT-EQUIVALENT":
            a("    These images depict the same scene but show measurable")
            a("    encoding differences. Could indicate a crop, color grade,")
            a("    aggressive recompression, or format conversion that altered")
            a("    visual content. Further analysis of compression lineage,")
            a("    PRNU, and metadata is recommended before making lineage claims.")
        else:
            a("    These images are visually distinct. They do not share the")
            a("    same depicted content at a perceptual level.")
            if c["ssim"] < SSIM_CONTENT_EQUIV_FLOOR and c["phash_dist"] <= PHASH_CONTENT_EQUIV_THRESHOLD:
                a("")
                a("    ⚠  SSIM FLOOR OVERRIDE TRIGGERED: The pHash Hamming distance")
                a(f"    was {c['phash_dist']} (within the content-equivalent zone), but")
                a(f"    SSIM={c['ssim']:.4f} fell below the floor of {SSIM_CONTENT_EQUIV_FLOOR}.")
                a("    This pattern — low SSIM with a globally stable perceptual hash —")
                a("    is a strong indicator of a localized alteration such as a splice,")
                a("    inpainting, clone-stamp, or composite insertion. The overall image")
                a("    structure was preserved but a region was substantially changed.")
                a("    Tile-level (block) analysis is recommended to localize the edit.")
        a("")

    # ── summary matrix ────────────────────────────────────────────────────────
    if len(pairs) > 1:
        a(SEP_MAJOR)
        a("SECTION 3 — SUMMARY MATRIX")
        a(SEP_MAJOR)
        a("")

        # Build column headers from filenames (truncated)
        names = [r["filename"] for r in singles if not r.get("error")]
        col_w = 22
        header_row = " " * col_w + "".join(n[:col_w-2].ljust(col_w) for n in names)
        a(header_row)
        a(" " * col_w + (SEP_THIN[:col_w] * len(names))[:col_w * len(names)])

        # Build a lookup dict
        lookup = {}
        for c in pairs:
            lookup[(c["file_a"], c["file_b"])] = c.get("classification", "ERROR")
            lookup[(c["file_b"], c["file_a"])] = c.get("classification", "ERROR")

        for r in singles:
            if r.get("error"):
                continue
            row = r["filename"][:col_w-2].ljust(col_w)
            for c in singles:
                if c.get("error"):
                    row += "ERROR".ljust(col_w)
                elif r["filename"] == c["filename"]:
                    row += "(self)".ljust(col_w)
                else:
                    cls = lookup.get((r["filename"], c["filename"]), "—")
                    # Abbreviate for matrix
                    abbrev = {
                        "PIXEL-IDENTICAL":   "PIXEL-ID",
                        "RASTER-EQUIVALENT": "RASTER-EQ",
                        "DISPLAY-EQUIVALENT":"DISPLAY-EQ",
                        "CONTENT-EQUIVALENT":"CONTENT-EQ",
                        "CONTENT-DIVERGENT": "DIVERGENT",
                    }.get(cls, cls)
                    row += abbrev.ljust(col_w)
            a(row)

        a("")

    # ── footer ────────────────────────────────────────────────────────────────
    a(SEP_MAJOR)
    a("END OF REPORT")
    a(SEP_MAJOR)
    a("")
    a("METHODOLOGICAL DISCLOSURE")
    a("  This report was generated by image_similarity_analysis.py.")
    a("  All perceptual metrics are probabilistic. Classification labels")
    a("  follow the taxonomy in Johnson, Deepfake and Image Forgery Detection.")
    a("  Thresholds are configurable and should be validated against a")
    a("  ground-truth dataset for any specific evidentiary application.")
    a("  Perceptual similarity findings are leads, not conclusions.")
    a("  Chain-of-custody integrity must be established independently.")
    a("")
    a(f"  imagehash version : {imagehash.__version__}")
    try:
        import skimage
        a(f"  scikit-image      : {skimage.__version__}")
    except Exception:
        pass
    if LPIPS_AVAILABLE:
        a(f"  LPIPS / PyTorch   : available")
    else:
        a(f"  LPIPS / PyTorch   : NOT available")
    a("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python image_similarity_analysis.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1]).resolve()
    if not directory.is_dir():
        print(f"[FATAL] Not a directory: {directory}")
        sys.exit(1)

    # ── discover images ───────────────────────────────────────────────────────
    image_files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No supported image files found in {directory}")
        sys.exit(0)

    print(f"\nFound {len(image_files)} image(s) in {directory}\n")

    # ── per-image analysis ────────────────────────────────────────────────────
    print("Analyzing individual images...")
    singles = []
    for fp in image_files:
        print(f"  {fp.name}")
        singles.append(analyze_single(fp))

    # ── pairwise comparisons ──────────────────────────────────────────────────
    valid = [r for r in singles if not r.get("error")]
    pairs_input = list(itertools.combinations(valid, 2))

    print(f"\nComputing {len(pairs_input)} pairwise comparison(s)...")
    pairs = []
    for r1, r2 in pairs_input:
        print(f"  {r1['filename']}  ↔  {r2['filename']}")
        pairs.append(compare_pair(r1, r2))

    # ── write report ──────────────────────────────────────────────────────────
    output_path = directory / "image_similarity_report.txt"
    write_report(directory, singles, pairs, output_path)

    print(f"\nReport written to:\n  {output_path}\n")


if __name__ == "__main__":
    main()
