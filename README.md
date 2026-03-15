# Forensic-Image-Similarity-Analysis-Suite
Computes the similarity / identity measures for every image in a target directory and writes a structured plain-text report:.

Computes the following similarity / identity measures:

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
  • LPIPS                   Learned Perceptual Image Patch Similarity (AlexNet backbone; lower = more similar)

Forensic classification
  Pixel-Identical | Raster-Equivalent | Display-Equivalent | Content-Equivalent | Content-Divergent
  
USAGE
  python image_similarity_analysis.py /path/to/image/directory
  
OUTPUT
  image_similarity_report.txt  (written to the same directory)
  
INSTALL DEPENDENCIES
  pip install imagehash Pillow scikit-image numpy torch torchvision lpips
  
NOTES
  • LPIPS requires PyTorch.  If unavailable the script degrades gracefully and marks LPIPS as "unavailable".
  • All images are decoded to RGB before comparison; format differences (JPEG vs PNG vs HEIC) do not affect the decoded-raster metrics.
  • Hamming distance thresholds follow common forensic practice:
      0        → identical hash
      1–10     → likely visually equivalent (display-equivalent zone)
      11–19    → probable visible differences (content-equivalent zone)
      ≥ 20     → content-divergent
    • SSIM is used as a gate at BOTH the Display-Equivalent AND Content-Equivalent boundaries. A low SSIM overrides a low Hamming distance:
      SSIM ≥ 0.95  required for Display-Equivalent
      SSIM ≥ 0.75  required for Content-Equivalent (below → Divergent)
  • PSNR ≥ 40 dB is treated as perceptually lossless for 8-bit imagery.
  • PSNR divide-by-zero on pixel-identical images is handled gracefully.
