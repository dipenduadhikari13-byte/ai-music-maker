import streamlit as st
import subprocess
import sys
import io
import math
import tempfile
import os

# ──────────────────────────────────────────────
# Auto-install missing dependencies on startup
# ──────────────────────────────────────────────
_REQUIRED_PACKAGES = {
    "PIL":                "pillow",
    "pikepdf":            "pikepdf",
    "streamlit_cropper":  "streamlit-cropper",
    "pypdf":              "pypdf",
}
# NOTE: rembg is intentionally excluded from auto-install — it's very heavy
# (pulls onnxruntime + large ML models). Install manually if needed:
#   pip install rembg

def _ensure_packages():
    """Check for missing packages and install them once."""
    missing = []
    for import_name, pip_name in _REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        with st.spinner(f"Installing missing packages: {', '.join(missing)} …"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing
            )
        st.rerun()  # restart so the fresh imports take effect

_ensure_packages()

# ── Now safe to import everything ──
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from streamlit_cropper import st_cropper
import pikepdf

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

st.set_page_config(page_title="Image Resizer, Compressor & PDF Converter", page_icon="🖼️", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    .size-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 1.1em; margin: 4px 0;
    }
    .size-original { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b; }
    .size-result  { background: #00c85322; color: #00c853; border: 1px solid #00c853; }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #333;
    }
    .metric-label { color: #888; font-size: 0.85em; }
    .metric-value { color: #fff; font-size: 1.4em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Image Resizer, Compressor, Merger & PDF Tools")
st.caption("Resize, compress, merge images, convert to PDF, or merge PDFs — all in one place.")

# ──────────────────────────────────────────────
# Supported Formats
# ──────────────────────────────────────────────
SUPPORTED_INPUT = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif", "ico", "ppm", "pgm", "pbm", "pcx", "tga", "sgi", "eps", "dds"]
SUPPORTED_OUTPUT = ["JPG", "JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "ICO", "PPM"]

# Internal format name used by Pillow (JPG is saved as JPEG internally)
INTERNAL_FMT = {
    "JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP", "BMP": "BMP",
    "TIFF": "TIFF", "GIF": "GIF", "ICO": "ICO", "PPM": "PPM",
}

EXT_MAP = {
    "JPG": "jpg", "JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp",
    "TIFF": "tif", "GIF": "gif", "ICO": "ico", "PPM": "ppm",
}

MIME_MAP = {
    "JPG": "image/jpeg", "JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
    "BMP": "image/bmp", "TIFF": "image/tiff", "GIF": "image/gif",
    "ICO": "image/x-icon", "PPM": "image/x-portable-pixmap",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def prepare_for_format(img: Image.Image, fmt: str) -> Image.Image:
    """Ensure image mode is compatible with the output format."""
    fmt = INTERNAL_FMT.get(fmt, fmt)  # Normalize JPG → JPEG etc.
    # Convert CMYK to RGB first (common in print-ready images)
    if img.mode == "CMYK":
        img = img.convert("RGB")
    if fmt in ("JPEG", "BMP", "PPM", "ICO") and img.mode in ("RGBA", "P", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if "A" in img.mode:
            background.paste(img, mask=img.split()[-1])
        return background
    if fmt in ("JPEG", "BMP", "PPM") and img.mode not in ("RGB", "L"):
        return img.convert("RGB")
    return img


def get_image_bytes(img: Image.Image, fmt: str, quality: int, dpi: tuple | None = None) -> bytes:
    """Export a PIL Image to bytes with given format, quality, and optional DPI."""
    fmt = INTERNAL_FMT.get(fmt, fmt)  # Normalize JPG → JPEG etc.
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if dpi:
        save_kwargs["dpi"] = dpi
    img = prepare_for_format(img, fmt)
    if fmt == "PNG":
        save_kwargs["compress_level"] = max(0, min(9, 9 - int(quality / 11.2)))
    elif fmt == "WEBP":
        save_kwargs["quality"] = quality
        save_kwargs["method"] = 4
    elif fmt == "JPEG":
        save_kwargs["quality"] = quality
        # Baseline JPEG settings for maximum government/portal compatibility:
        # - subsampling=0 → 4:4:4 chroma (no colour info dropped, passes strict validators)
        # - progressive=False → force baseline (not progressive) JPEG
        # - optimize=False → standard Huffman tables (some strict validators reject optimized tables)
        save_kwargs["subsampling"] = 0
        save_kwargs["progressive"] = False
        save_kwargs["optimize"] = False
    elif fmt == "TIFF":
        save_kwargs["compression"] = "tiff_deflate"
    elif fmt == "GIF":
        pass  # GIF has no quality knob
    elif fmt == "ICO":
        # ICO needs specific sizes
        sizes = [(min(img.width, 256), min(img.height, 256))]
        save_kwargs["sizes"] = sizes
    else:
        save_kwargs["quality"] = quality
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def _binary_search_quality(img: Image.Image, target_bytes: int, fmt: str, dpi: tuple | None = None) -> tuple[bytes, int]:
    """Binary-search quality for a single image size. Returns (bytes, quality)."""
    lo, hi = 1, 100
    # Start with the lowest quality to get the smallest possible at this resolution
    smallest_data = get_image_bytes(img, fmt, lo, dpi)
    smallest_quality = lo

    # If even lowest quality exceeds target, return it (caller will downscale)
    if len(smallest_data) > target_bytes:
        return smallest_data, lo

    # Lowest quality fits — now binary search for the highest quality that still fits
    best_data = smallest_data
    best_quality = lo

    for _ in range(14):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        data = get_image_bytes(img, fmt, mid, dpi)
        if len(data) <= target_bytes:
            best_data = data
            best_quality = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best_data, best_quality


def compress_to_target(img: Image.Image, target_bytes: int, fmt: str, dpi: tuple | None = None) -> tuple[bytes, int]:
    """Compress image to target file size, downscaling if quality alone isn't enough."""
    real_fmt = INTERNAL_FMT.get(fmt, fmt)

    if real_fmt in ("PNG", "BMP", "GIF", "PPM", "ICO"):
        # Lossless formats: can only downscale dimensions to reduce size
        current = img.copy()
        for attempt in range(20):
            data = get_image_bytes(current, fmt, 95, dpi)
            if len(data) <= target_bytes:
                return data, 95
            # Shrink by 20% each pass
            new_w = max(1, int(current.width * 0.8))
            new_h = max(1, int(current.height * 0.8))
            if new_w == current.width and new_h == current.height:
                break  # can't shrink further
            current = img.resize((new_w, new_h), Image.LANCZOS)
        return get_image_bytes(current, fmt, 95, dpi), 95

    # Lossy formats (JPEG, WEBP, etc.): first try quality, then downscale + quality
    # Quick check: does quality=100 already fit?
    hi_data = get_image_bytes(img, fmt, 100, dpi)
    if len(hi_data) <= target_bytes:
        return hi_data, 100

    # Try quality search at current resolution
    data, quality = _binary_search_quality(img, target_bytes, fmt, dpi)
    if len(data) <= target_bytes:
        return data, quality

    # Quality alone isn't enough — progressively downscale
    current = img.copy()
    for attempt in range(20):
        # Shrink by 20% each pass
        new_w = max(1, int(current.width * 0.8))
        new_h = max(1, int(current.height * 0.8))
        if new_w == current.width and new_h == current.height:
            break  # can't shrink further
        current = img.resize((new_w, new_h), Image.LANCZOS)

        data, quality = _binary_search_quality(current, target_bytes, fmt, dpi)
        if len(data) <= target_bytes:
            return data, quality

    # Return the smallest we achieved (lowest quality at smallest dimensions)
    return data, quality


# ──────────────────────────────────────────────
# Background Color Presets
# ──────────────────────────────────────────────

BG_COLOR_PRESETS = {
    "No Change": None,
    "No Background (Transparent)": "transparent",
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Light Blue": (173, 216, 230),
    "Sky Blue": (135, 206, 235),
    "Light Gray": (211, 211, 211),
    "Light Green": (144, 238, 144),
    "Light Pink": (255, 182, 193),
    "Light Yellow": (255, 255, 224),
    "Lavender": (230, 230, 250),
    "Beige": (245, 245, 220),
    "Mint": (189, 252, 201),
    "Peach": (255, 218, 185),
    "Coral": (255, 127, 80),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Green": (0, 128, 0),
    "Custom Color": "custom",
    "Upload Background Image": "upload",
}


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from an image using rembg. Returns RGBA image."""
    if not REMBG_AVAILABLE:
        raise ImportError("rembg is not installed. Install it with: pip install rembg")
    buf_in = io.BytesIO()
    img.save(buf_in, format="PNG")
    buf_in.seek(0)
    buf_out = rembg_remove(buf_in.getvalue())
    return Image.open(io.BytesIO(buf_out)).convert("RGBA")


def apply_background(fg_img: Image.Image, bg_option, custom_color=None, bg_image=None) -> Image.Image:
    """Apply a background to a foreground RGBA image.

    Args:
        fg_img: Foreground RGBA image (after background removal).
        bg_option: Value from BG_COLOR_PRESETS.
        custom_color: RGB tuple when bg_option is 'custom'.
        bg_image: PIL Image to use as background when bg_option is 'upload'.
    Returns:
        Composited image (RGBA for transparent, RGB otherwise).
    """
    if bg_option == "transparent":
        return fg_img  # Keep RGBA with transparency

    if bg_option == "upload" and bg_image is not None:
        bg = bg_image.convert("RGBA").resize(fg_img.size, Image.LANCZOS)
        bg.paste(fg_img, (0, 0), fg_img)
        return bg.convert("RGB")

    # Solid color background
    color = custom_color if bg_option == "custom" and custom_color else bg_option
    if not isinstance(color, tuple) or len(color) < 3:
        color = (255, 255, 255)
    bg = Image.new("RGB", fg_img.size, color[:3])
    bg.paste(fg_img, (0, 0), fg_img)
    return bg


ASPECT_PRESETS = {
    "Free (custom)": None,
    "1:1 (Square)": (1, 1),
    "4:3 (Standard)": (4, 3),
    "3:4 (Portrait)": (3, 4),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Vertical / Reel)": (9, 16),
    "3:2 (Photo)": (3, 2),
    "2:3 (Photo Portrait)": (2, 3),
    "21:9 (Ultra-wide)": (21, 9),
    "5:4 (Large Format)": (5, 4),
    "4:5 (Instagram Portrait)": (4, 5),
    "2:1 (Panoramic)": (2, 1),
    "1:2 (Tall Banner)": (1, 2),
}

RESOLUTION_PRESETS = {
    "Custom": None,
    "HD (1280×720)": (1280, 720),
    "Full HD (1920×1080)": (1920, 1080),
    "2K (2560×1440)": (2560, 1440),
    "4K UHD (3840×2160)": (3840, 2160),
    "Instagram Post (1080×1080)": (1080, 1080),
    "Instagram Story (1080×1920)": (1080, 1920),
    "Facebook Cover (820×312)": (820, 312),
    "Twitter Header (1500×500)": (1500, 500),
    "YouTube Thumbnail (1280×720)": (1280, 720),
    "LinkedIn Banner (1584×396)": (1584, 396),
    "Passport Photo (600×600)": (600, 600),
    "A4 Print 300DPI (2480×3508)": (2480, 3508),
    "A4 Print 150DPI (1240×1754)": (1240, 1754),
    "A3 Print 300DPI (3508×4960)": (3508, 4960),
    "Letter Print 300DPI (2550×3300)": (2550, 3300),
    "Thumbnail (150×150)": (150, 150),
    "Icon (64×64)": (64, 64),
    "Favicon (32×32)": (32, 32),
}

DPI_PRESETS = {
    "Custom": None,
    "Screen (72 DPI)": 72,
    "Screen Retina (144 DPI)": 144,
    "Low Print (150 DPI)": 150,
    "Standard Print (300 DPI)": 300,
    "High Print (600 DPI)": 600,
    "Professional (1200 DPI)": 1200,
}

# ──────────────────────────────────────────────
# Image to PDF Helper
# ──────────────────────────────────────────────

def _jpeg_compress_image(img: Image.Image, quality: int = 85) -> Image.Image:
    """JPEG round-trip to compress an image in memory — dramatically reduces PDF size."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0, progressive=False, optimize=False)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() so buffer can be freed


def images_to_pdf(images: list[Image.Image], page_size: str = "A4", orientation: str = "Auto",
                  margin_mm: int = 10, fit_mode: str = "Fit to page", dpi: int = 300,
                  jpeg_quality: int = 85, title: str = "") -> bytes:
    """Convert PIL Images into a PDF, embedding JPEG data directly via pikepdf.

    This avoids Pillow's bitmap-canvas approach (which re-encodes images and
    inflates file size) and keeps output within ~1 % of the raw JPEG size.
    """
    PAGE_SIZES = {
        "A4": (210, 297), "A3": (297, 420), "A5": (148, 210),
        "Letter": (216, 279), "Legal": (216, 356),
        "Fit to Image": None,
    }
    page_mm_val = PAGE_SIZES.get(page_size)
    pdf = pikepdf.Pdf.new()

    for pil_img in images:
        pil_img = pil_img.convert("RGB")
        img_w, img_h = pil_img.size

        if page_mm_val is None:
            # --- Fit to Image: page = image size, no margins ---
            page_w_pt = img_w / max(dpi, 1) * 72
            page_h_pt = img_h / max(dpi, 1) * 72
            final_img = pil_img
            draw_w, draw_h = page_w_pt, page_h_pt
            x_off, y_off = 0.0, 0.0
        else:
            pw_mm, ph_mm = page_mm_val
            if orientation == "Landscape":
                pw_mm, ph_mm = ph_mm, pw_mm
            elif orientation == "Auto":
                if img_w > img_h:
                    pw_mm, ph_mm = max(pw_mm, ph_mm), min(pw_mm, ph_mm)
                else:
                    pw_mm, ph_mm = min(pw_mm, ph_mm), max(pw_mm, ph_mm)

            page_w_pt = pw_mm * 72 / 25.4
            page_h_pt = ph_mm * 72 / 25.4
            margin_pt = margin_mm * 72 / 25.4
            usable_w_pt = page_w_pt - 2 * margin_pt
            usable_h_pt = page_h_pt - 2 * margin_pt

            # Image physical size at target DPI (in PDF points)
            img_w_pt = img_w / max(dpi, 1) * 72
            img_h_pt = img_h / max(dpi, 1) * 72

            if fit_mode == "Fit to page":
                ratio = min(usable_w_pt / max(img_w_pt, 0.1),
                            usable_h_pt / max(img_h_pt, 0.1), 1.0)
                draw_w = img_w_pt * ratio
                draw_h = img_h_pt * ratio
                # Downscale pixel data when image is larger than needed
                needed_w = max(int(img_w * ratio), 1)
                needed_h = max(int(img_h * ratio), 1)
                if ratio < 1.0:
                    final_img = pil_img.resize((needed_w, needed_h), Image.LANCZOS)
                else:
                    final_img = pil_img
                x_off = margin_pt + (usable_w_pt - draw_w) / 2
                y_off = margin_pt + (usable_h_pt - draw_h) / 2

            elif fit_mode == "Fill page (crop)":
                scale_x = usable_w_pt / max(img_w_pt, 0.1)
                scale_y = usable_h_pt / max(img_h_pt, 0.1)
                ratio = min(max(scale_x, scale_y), 2.0)
                new_w = max(int(img_w * ratio), 1)
                new_h = max(int(img_h * ratio), 1)
                resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
                # Crop to usable-area pixel count (centered)
                crop_w = min(max(int(usable_w_pt * dpi / 72), 1), new_w)
                crop_h = min(max(int(usable_h_pt * dpi / 72), 1), new_h)
                left = (new_w - crop_w) // 2
                top = (new_h - crop_h) // 2
                final_img = resized.crop((left, top, left + crop_w, top + crop_h))
                draw_w = usable_w_pt
                draw_h = usable_h_pt
                x_off = margin_pt
                y_off = margin_pt

            else:  # Stretch
                target_w = max(int(usable_w_pt * dpi / 72), 1)
                target_h = max(int(usable_h_pt * dpi / 72), 1)
                final_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                draw_w = usable_w_pt
                draw_h = usable_h_pt
                x_off = margin_pt
                y_off = margin_pt

        # ── Single JPEG encoding — then embed directly into PDF ──
        jbuf = io.BytesIO()
        final_img.save(jbuf, format="JPEG", quality=jpeg_quality, subsampling=0, progressive=False, optimize=False)
        jpeg_data = jbuf.getvalue()

        image_obj = pikepdf.Stream(pdf, jpeg_data)
        image_obj["/Type"] = pikepdf.Name.XObject
        image_obj["/Subtype"] = pikepdf.Name.Image
        image_obj["/Width"] = final_img.width
        image_obj["/Height"] = final_img.height
        image_obj["/ColorSpace"] = pikepdf.Name.DeviceRGB
        image_obj["/BitsPerComponent"] = 8
        image_obj["/Filter"] = pikepdf.Name.DCTDecode
        image_ref = pdf.make_indirect(image_obj)

        # PDF content stream: position and scale the image
        content = f"q {draw_w:.4f} 0 0 {draw_h:.4f} {x_off:.4f} {y_off:.4f} cm /Im0 Do Q"
        content_stream = pikepdf.Stream(pdf, content.encode())

        page = pdf.add_blank_page(page_size=(page_w_pt, page_h_pt))
        page.obj["/Contents"] = pdf.make_indirect(content_stream)
        page.obj["/Resources"] = pikepdf.Dictionary({
            "/XObject": pikepdf.Dictionary({
                "/Im0": image_ref,
            })
        })

    out = io.BytesIO()
    pdf.save(out)
    return out.getvalue()


# ──────────────────────────────────────────────
# Merge Images Helper
# ──────────────────────────────────────────────

def merge_images(
    images: list[Image.Image],
    direction: str = "horizontal",
    alignment: str = "center",
    gap: int = 0,
    bg_color: tuple = (255, 255, 255),
    output_format: str = "JPEG",
    quality: int = 90,
) -> tuple[Image.Image, bytes]:
    """Merge 2–8 images into a single image.

    Args:
        direction: 'horizontal', 'vertical', or 'grid'
        alignment: 'top'/'left', 'center', 'bottom'/'right'
        gap: pixel gap between images
        bg_color: RGB background fill
        output_format: PIL format name
        quality: JPEG/WEBP quality
    Returns:
        (merged_pil_image, merged_bytes)
    """
    imgs = [im.convert("RGB") for im in images]
    n = len(imgs)

    if direction == "horizontal":
        total_w = sum(im.width for im in imgs) + gap * (n - 1)
        max_h = max(im.height for im in imgs)
        canvas = Image.new("RGB", (total_w, max_h), bg_color)
        x = 0
        for im in imgs:
            if alignment == "top":
                y = 0
            elif alignment == "bottom":
                y = max_h - im.height
            else:  # center
                y = (max_h - im.height) // 2
            canvas.paste(im, (x, y))
            x += im.width + gap

    elif direction == "vertical":
        max_w = max(im.width for im in imgs)
        total_h = sum(im.height for im in imgs) + gap * (n - 1)
        canvas = Image.new("RGB", (max_w, total_h), bg_color)
        y = 0
        for im in imgs:
            if alignment == "left":
                x = 0
            elif alignment == "right":
                x = max_w - im.width
            else:  # center
                x = (max_w - im.width) // 2
            canvas.paste(im, (x, y))
            y += im.height + gap

    else:  # grid
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        # Make all images the same size (largest dimensions)
        cell_w = max(im.width for im in imgs)
        cell_h = max(im.height for im in imgs)
        total_w = cols * cell_w + gap * (cols - 1)
        total_h = rows * cell_h + gap * (rows - 1)
        canvas = Image.new("RGB", (total_w, total_h), bg_color)
        for idx, im in enumerate(imgs):
            row, col = divmod(idx, cols)
            # Center image in its cell
            cx = col * (cell_w + gap) + (cell_w - im.width) // 2
            cy = row * (cell_h + gap) + (cell_h - im.height) // 2
            canvas.paste(im, (cx, cy))

    # Export
    buf = io.BytesIO()
    save_fmt = INTERNAL_FMT.get(output_format, output_format)
    save_img = prepare_for_format(canvas, save_fmt)
    if save_fmt == "JPEG":
        save_img.save(buf, format="JPEG", quality=quality, subsampling=0, progressive=False, optimize=False)
    elif save_fmt == "WEBP":
        save_img.save(buf, format="WEBP", quality=quality, method=4)
    elif save_fmt == "PNG":
        save_img.save(buf, format="PNG", compress_level=6)
    else:
        save_img.save(buf, format=save_fmt)
    return canvas, buf.getvalue()


# ──────────────────────────────────────────────
# Merge PDFs Helper
# ──────────────────────────────────────────────

def merge_pdfs(pdf_files: list[io.BytesIO]) -> bytes:
    """Merge 2–8 PDF files into a single PDF using pypdf."""
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    total_pages = 0
    for pdf_data in pdf_files:
        reader = PdfReader(pdf_data)
        for page in reader.pages:
            writer.add_page(page)
            total_pages += 1
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue(), total_pages


# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "🖼️ Image Resizer & Compressor",
    "📄 Image to PDF Converter",
    "🧩 Merge Images",
    "📚 Merge PDFs",
])

# ╔══════════════════════════════════════════════╗
# ║            TAB 1: IMAGE RESIZER              ║
# ╚══════════════════════════════════════════════╝
with main_tab1:

    uploaded = st.file_uploader("Upload an image", type=SUPPORTED_INPUT, key="img_upload")

    if uploaded:
        raw_bytes = uploaded.getvalue()
        original_size = len(raw_bytes)
        img = Image.open(io.BytesIO(raw_bytes))
        orig_w, orig_h = img.size
        _raw_dpi = img.info.get("dpi", (72, 72))
        orig_dpi = (float(_raw_dpi[0]), float(_raw_dpi[1]))
        orig_format = img.format or uploaded.name.rsplit(".", 1)[-1].upper()
        # Keep JPG as-is for display, map to JPEG only internally when saving
        if orig_format == "JPEG":
            orig_format = "JPG"  # Show user-friendly "JPG"
        orig_mode = img.mode

        # ── Show original info ──
        st.markdown("---")
        st.subheader("📊 Original Image Info")

        cols = st.columns(6)
        gcd_val = math.gcd(orig_w, orig_h)
        info_items = [
            ("File Size", format_size(original_size)),
            ("Dimensions", f"{orig_w} × {orig_h} px"),
            ("Aspect Ratio", f"{orig_w // gcd_val}:{orig_h // gcd_val}"),
            ("Format", orig_format),
            ("DPI", f"{orig_dpi[0]:.0f} × {orig_dpi[1]:.0f}"),
            ("Color Mode", orig_mode),
        ]
        for col, (label, value) in zip(cols, info_items):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.image(img, caption="Original", use_container_width=True)

        # ──────────────────────────────────────
        # Settings
        # ──────────────────────────────────────
        st.markdown("---")
        st.subheader("⚙️ Resize & Compress Settings")

        tab_size, tab_dim, tab_res, tab_dpi, tab_crop, tab_bg = st.tabs([
            "📦 Target File Size",
            "📐 Dimensions & Aspect Ratio",
            "🖥️ Resolution Presets",
            "🔍 DPI",
            "✂️ Crop",
            "🎨 Background",
        ])

        # --- Target file size ---
        with tab_size:
            size_unit = st.radio("Unit", ["KB", "MB"], horizontal=True, key="sz_unit")
            if size_unit == "MB":
                min_val = 0.01
                max_val = 100.0
                default_val = round(original_size / (1024 * 1024), 2)
                step = 0.01
            else:
                min_val = 1.0
                max_val = 10240.0
                default_val = round(original_size / 1024, 1)
                step = 5.0
            target_default = round(max(min_val, min(default_val, max_val)), 2)
            target_val = st.number_input(
                f"Desired file size ({size_unit})",
                min_value=min_val,
                max_value=max_val,
                value=target_default,
                step=step,
                key=f"target_size_{size_unit}",
            )
            enable_size_target = st.checkbox("Enable target file-size compression", value=False, key="en_sz")

        # --- Dimensions & aspect ratio ---
        with tab_dim:
            aspect_choice = st.selectbox("Aspect ratio preset", list(ASPECT_PRESETS.keys()), key="ar_preset")
            aspect = ASPECT_PRESETS[aspect_choice]

            if aspect:
                new_w = st.number_input("Width (px)", min_value=1, value=orig_w, step=1, key="dim_w_locked")
                new_h = int(new_w * aspect[1] / aspect[0])
                st.info(f"Height auto-calculated to **{new_h} px** for {aspect_choice} ratio.")
            else:
                c1, c2 = st.columns(2)
                new_w = c1.number_input("Width (px)", min_value=1, value=orig_w, step=1, key="dim_w")
                new_h = c2.number_input("Height (px)", min_value=1, value=orig_h, step=1, key="dim_h")

            lock_ratio = st.checkbox("Lock aspect ratio (scale proportionally)", value=False, key="lock_ar")
            if lock_ratio and aspect is None:
                ratio = orig_w / orig_h
                new_h = int(new_w / ratio)
                st.info(f"Height auto-set to **{new_h} px** to maintain original ratio.")

            enable_resize = st.checkbox("Enable dimension resize", value=False, key="en_dim")

            resample_options = {
                "Lanczos (best quality)": Image.LANCZOS,
                "Bicubic": Image.BICUBIC,
                "Bilinear": Image.BILINEAR,
                "Nearest (fastest)": Image.NEAREST,
            }
            resample_name = st.selectbox("Resampling filter", list(resample_options.keys()), key="resample")
            resample_filter = resample_options[resample_name]

        # --- Resolution Presets ---
        with tab_res:
            res_choice = st.selectbox("Resolution preset", list(RESOLUTION_PRESETS.keys()), key="res_preset")
            res_preset = RESOLUTION_PRESETS[res_choice]

            if res_preset:
                preset_w, preset_h = res_preset
                st.success(f"Selected: **{preset_w} × {preset_h} px**")
                enable_res_preset = st.checkbox("Apply this resolution preset", value=False, key="en_res")
            else:
                preset_w, preset_h = orig_w, orig_h
                enable_res_preset = False
                st.info("Select a preset above, or use the Dimensions tab for custom sizes.")

        # --- DPI ---
        with tab_dpi:
            dpi_preset_choice = st.selectbox("DPI preset", list(DPI_PRESETS.keys()), key="dpi_preset")
            dpi_preset_val = DPI_PRESETS[dpi_preset_choice]

            if dpi_preset_val:
                dpi_x = dpi_preset_val
                dpi_y = dpi_preset_val
                st.success(f"Selected: **{dpi_x} DPI**")
            else:
                c1, c2 = st.columns(2)
                dpi_x = c1.number_input("Horizontal DPI", min_value=1, max_value=2400, value=int(orig_dpi[0]), step=1, key="dpi_x")
                dpi_y = c2.number_input("Vertical DPI", min_value=1, max_value=2400, value=int(orig_dpi[1]), step=1, key="dpi_y")

            enable_dpi = st.checkbox("Change DPI metadata", value=False, key="en_dpi")
            st.caption("DPI change only updates metadata (for print). It does NOT resample pixels.")

        # --- Crop ---
        with tab_crop:
            enable_crop = st.checkbox("Enable crop & transform", value=False, key="en_crop")

            if enable_crop:
                # ── Gallery-style CSS for crop controls ──
                st.markdown("""
                <style>
                    .crop-toolbar {
                        display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px 0;
                        justify-content: center;
                    }
                    .crop-pill {
                        display: inline-block; padding: 6px 16px; border-radius: 20px;
                        font-size: 0.85em; font-weight: 600; cursor: pointer;
                        border: 1.5px solid #555; background: #1a1a2e; color: #ccc;
                        transition: all 0.15s ease;
                    }
                    .crop-pill.active { background: #ff4b4b; border-color: #ff4b4b; color: #fff; }
                    .crop-info-bar {
                        display: flex; justify-content: center; gap: 24px;
                        padding: 8px 16px; background: #111827; border-radius: 10px;
                        font-family: monospace; color: #94a3b8; font-size: 0.9em;
                        margin: 8px 0;
                    }
                    .crop-info-bar span { color: #60a5fa; font-weight: 700; }
                    .transform-btn-row {
                        display: flex; justify-content: center; gap: 8px; margin: 12px 0;
                    }
                </style>
                """, unsafe_allow_html=True)

                # ── 1) Rotate & Flip toolbar ──
                st.markdown("##### 🔄 Transform")
                tf_c1, tf_c2, tf_c3, tf_c4, tf_c5 = st.columns(5)
                with tf_c1:
                    rot_left = st.button("↺ 90°", key="crop_rot_l", help="Rotate left 90°", use_container_width=True)
                with tf_c2:
                    rot_right = st.button("↻ 90°", key="crop_rot_r", help="Rotate right 90°", use_container_width=True)
                with tf_c3:
                    flip_h = st.button("↔ Flip H", key="crop_flip_h_btn", help="Flip horizontally", use_container_width=True)
                with tf_c4:
                    flip_v = st.button("↕ Flip V", key="crop_flip_v_btn", help="Flip vertically", use_container_width=True)
                with tf_c5:
                    rot_custom = st.button("⟳ Custom", key="crop_rot_custom", help="Custom rotation angle", use_container_width=True)

                # Track transforms in session state
                if "crop_rotation" not in st.session_state:
                    st.session_state.crop_rotation = 0
                if "crop_flip_h_state" not in st.session_state:
                    st.session_state.crop_flip_h_state = False
                if "crop_flip_v_state" not in st.session_state:
                    st.session_state.crop_flip_v_state = False
                if "crop_custom_angle" not in st.session_state:
                    st.session_state.crop_custom_angle = 0

                if rot_left:
                    st.session_state.crop_rotation = (st.session_state.crop_rotation - 90) % 360
                    st.rerun()
                if rot_right:
                    st.session_state.crop_rotation = (st.session_state.crop_rotation + 90) % 360
                    st.rerun()
                if flip_h:
                    st.session_state.crop_flip_h_state = not st.session_state.crop_flip_h_state
                    st.rerun()
                if flip_v:
                    st.session_state.crop_flip_v_state = not st.session_state.crop_flip_v_state
                    st.rerun()

                if rot_custom:
                    st.session_state["_show_custom_rotation"] = True

                if st.session_state.get("_show_custom_rotation", False):
                    custom_angle = st.slider("Rotation angle (°)", -180, 180,
                                             st.session_state.crop_custom_angle, step=1, key="crop_angle_slider")
                    if custom_angle != st.session_state.crop_custom_angle:
                        st.session_state.crop_custom_angle = custom_angle

                # Show current transform status
                transforms = []
                if st.session_state.crop_rotation != 0:
                    transforms.append(f"Rotated {st.session_state.crop_rotation}°")
                if st.session_state.crop_custom_angle != 0:
                    transforms.append(f"Fine-rotated {st.session_state.crop_custom_angle}°")
                if st.session_state.crop_flip_h_state:
                    transforms.append("Flipped H")
                if st.session_state.crop_flip_v_state:
                    transforms.append("Flipped V")
                if transforms:
                    st.caption(f"Active transforms: {' · '.join(transforms)}")
                    if st.button("🔄 Reset transforms", key="crop_reset_tf"):
                        st.session_state.crop_rotation = 0
                        st.session_state.crop_flip_h_state = False
                        st.session_state.crop_flip_v_state = False
                        st.session_state.crop_custom_angle = 0
                        st.session_state["_show_custom_rotation"] = False
                        st.rerun()

                # Apply transforms to get the crop source image
                crop_source = img.copy()
                if st.session_state.crop_rotation != 0:
                    crop_source = crop_source.rotate(-st.session_state.crop_rotation, expand=True, resample=Image.BICUBIC)
                if st.session_state.crop_custom_angle != 0:
                    crop_source = crop_source.rotate(-st.session_state.crop_custom_angle, expand=True,
                                                     resample=Image.BICUBIC, fillcolor=(255, 255, 255) if crop_source.mode == "RGB" else None)
                if st.session_state.crop_flip_h_state:
                    crop_source = crop_source.transpose(Image.FLIP_LEFT_RIGHT)
                if st.session_state.crop_flip_v_state:
                    crop_source = crop_source.transpose(Image.FLIP_TOP_BOTTOM)

                st.markdown("---")

                # ── 2) Quick aspect-ratio pill toolbar (gallery style) ──
                st.markdown("##### ✂️ Crop")

                CROP_ASPECT_PRESETS = [
                    ("Free",    None),
                    ("1:1",     (1, 1)),
                    ("4:3",     (4, 3)),
                    ("3:4",     (3, 4)),
                    ("16:9",    (16, 9)),
                    ("9:16",    (9, 16)),
                    ("3:2",     (3, 2)),
                    ("2:3",     (2, 3)),
                    ("4:5",     (4, 5)),
                    ("5:4",     (5, 4)),
                    ("21:9",    (21, 9)),
                ]

                # Render aspect-ratio buttons as a horizontal row
                ar_cols = st.columns(len(CROP_ASPECT_PRESETS))
                if "crop_aspect_idx" not in st.session_state:
                    st.session_state.crop_aspect_idx = 0

                for idx, (label, _ratio) in enumerate(CROP_ASPECT_PRESETS):
                    with ar_cols[idx]:
                        btn_type = "primary" if st.session_state.crop_aspect_idx == idx else "secondary"
                        if st.button(label, key=f"ar_btn_{idx}", use_container_width=True, type=btn_type):
                            st.session_state.crop_aspect_idx = idx
                            st.rerun()

                crop_aspect = CROP_ASPECT_PRESETS[st.session_state.crop_aspect_idx][1]
                crop_aspect_label = CROP_ASPECT_PRESETS[st.session_state.crop_aspect_idx][0]

                # Social media presets
                with st.expander("📱 Social media presets", expanded=False):
                    SOCIAL_PRESETS = {
                        "Instagram Post (1:1)":      (1, 1),
                        "Instagram Story (9:16)":    (9, 16),
                        "Instagram Portrait (4:5)":  (4, 5),
                        "YouTube Thumbnail (16:9)":  (16, 9),
                        "Facebook Post (1.91:1)":    (191, 100),
                        "Twitter Post (16:9)":       (16, 9),
                        "TikTok (9:16)":             (9, 16),
                        "LinkedIn Banner (4:1)":     (4, 1),
                        "Pinterest Pin (2:3)":       (2, 3),
                    }
                    social_choice = st.selectbox("Quick preset", ["None"] + list(SOCIAL_PRESETS.keys()),
                                                 key="crop_social_preset")
                    if social_choice != "None":
                        crop_aspect = SOCIAL_PRESETS[social_choice]
                        crop_aspect_label = social_choice

                # ── 3) Settings row ──
                with st.expander("⚙️ Crop settings", expanded=False):
                    s_c1, s_c2 = st.columns(2)
                    with s_c1:
                        crop_box_color = st.color_picker("Box color", "#FF4B4B", key="crop_box_clr")
                        crop_stroke = st.slider("Box thickness", 1, 8, 3, key="crop_stroke_w")
                    with s_c2:
                        crop_realtime = st.checkbox("Real-time preview", value=True, key="crop_rt",
                                                    help="Update as you drag. Disable for large images.")

                # ── 4) Visual cropper — hero element ──
                crop_img_input = crop_source.copy()
                if crop_img_input.mode != "RGB":
                    crop_img_input = crop_img_input.convert("RGB")

                src_w, src_h = crop_img_input.size
                st.markdown(
                    f'<div class="crop-info-bar">'
                    f'Source: <span>{src_w} × {src_h}</span> px'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;Ratio: <span>{crop_aspect_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                cropped_result = st_cropper(
                    crop_img_input,
                    realtime_update=crop_realtime,
                    box_color=crop_box_color,
                    aspect_ratio=crop_aspect,
                    return_type="both",
                    stroke_width=crop_stroke,
                    key="visual_cropper",
                )

                # Parse result
                if isinstance(cropped_result, tuple) and len(cropped_result) == 2:
                    cropped_img, crop_box = cropped_result
                else:
                    cropped_img = cropped_result
                    crop_box = None

                # ── 5) Crop info & manual coordinate entry ──
                if cropped_img is not None:
                    cw, ch = cropped_img.size
                    box_left = int(crop_box.get("left", 0)) if crop_box else 0
                    box_top = int(crop_box.get("top", 0)) if crop_box else 0
                    box_w = int(crop_box.get("width", cw)) if crop_box else cw
                    box_h = int(crop_box.get("height", ch)) if crop_box else ch

                    st.markdown(
                        f'<div class="crop-info-bar">'
                        f'Crop: <span>{cw} × {ch}</span> px'
                        f'&nbsp;&nbsp;|&nbsp;&nbsp;Position: <span>({box_left}, {box_top})</span>'
                        f'&nbsp;&nbsp;|&nbsp;&nbsp;Size: <span>{box_w} × {box_h}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Manual coordinate entry (collapsible)
                    with st.expander("📏 Manual crop coordinates", expanded=False):
                        st.caption("Enter exact pixel coordinates for precise cropping.")
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        man_left = mc1.number_input("Left (X)", 0, src_w - 1, box_left, key="man_crop_l")
                        man_top = mc2.number_input("Top (Y)", 0, src_h - 1, box_top, key="man_crop_t")
                        man_right = mc3.number_input("Right (X)", 1, src_w, min(box_left + box_w, src_w), key="man_crop_r")
                        man_bottom = mc4.number_input("Bottom (Y)", 1, src_h, min(box_top + box_h, src_h), key="man_crop_b")

                        if st.button("Apply manual coordinates", key="apply_man_crop", type="primary"):
                            if man_right > man_left and man_bottom > man_top:
                                cropped_img = crop_img_input.crop((man_left, man_top, man_right, man_bottom))
                                cw, ch = cropped_img.size
                                st.success(f"Manual crop applied: **{cw} × {ch} px**")
                            else:
                                st.error("Invalid coordinates. Right must be > Left, Bottom must be > Top.")

                    # ── 6) Side-by-side preview ──
                    st.markdown("---")
                    prev_c1, prev_c2 = st.columns(2)
                    with prev_c1:
                        st.markdown("###### 📷 Original (transformed)")
                        st.image(crop_img_input, use_container_width=True)
                    with prev_c2:
                        st.markdown(f"###### ✂️ Cropped — {cw} × {ch} px")
                        st.image(cropped_img, use_container_width=True)
                else:
                    st.warning("Could not produce a crop. Try adjusting the crop box.")
                    cropped_img = None
            else:
                cropped_img = None
                st.info("☝ Enable **crop & transform** above to open the interactive crop tool "
                        "with aspect ratios, rotation, flip, and manual coordinates.")

        # --- Background ---
        with tab_bg:
            if not REMBG_AVAILABLE:
                st.warning("⚠️ `rembg` is not installed. Install it with `pip install rembg` to enable background features.")

            bg_choice = st.selectbox(
                "Background option",
                list(BG_COLOR_PRESETS.keys()),
                index=0,
                key="bg_choice",
                help="Remove the existing background and replace it with a solid color, transparency, or a custom image.",
            )
            bg_option = BG_COLOR_PRESETS[bg_choice]
            enable_bg = bg_option is not None

            custom_bg_color = None
            bg_upload_img = None

            if bg_option == "custom":
                picked = st.color_picker("Pick a custom background color", "#ADD8E6", key="bg_custom_color")
                custom_bg_color = tuple(int(picked.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                st.markdown(
                    f'<div style="width:60px;height:60px;border-radius:8px;'
                    f'background:rgb{custom_bg_color};border:1px solid #555;"></div>',
                    unsafe_allow_html=True,
                )
            elif bg_option == "upload":
                bg_file = st.file_uploader(
                    "Upload a background image",
                    type=SUPPORTED_INPUT,
                    key="bg_img_upload",
                )
                if bg_file:
                    bg_upload_img = Image.open(io.BytesIO(bg_file.getvalue()))
                    st.image(bg_upload_img, caption="Background preview", use_container_width=True)
                else:
                    st.info("Upload an image to use as the background.")
            elif bg_option == "transparent":
                st.info("Background will be removed. Output will be **PNG** with transparency.")
            elif isinstance(bg_option, tuple):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;">'
                    f'<div style="width:60px;height:60px;border-radius:8px;'
                    f'background:rgb{bg_option};border:1px solid #555;"></div>'
                    f'<span style="font-size:1.1em;">Preview: <b>{bg_choice}</b></span></div>',
                    unsafe_allow_html=True,
                )

            if enable_bg:
                st.caption("🔄 The background will be removed first, then the selected background will be applied.")

        # --- Output format & quality ---
        st.markdown("---")
        st.subheader("💾 Output Settings")

        c_fmt, c_qual = st.columns(2)
        with c_fmt:
            default_idx = SUPPORTED_OUTPUT.index(orig_format) if orig_format in SUPPORTED_OUTPUT else 0
            out_format = st.selectbox("Output format", SUPPORTED_OUTPUT, index=default_idx, key="out_fmt")
        with c_qual:
            if out_format in ("PNG", "BMP", "GIF", "PPM", "ICO"):
                st.info(f"{out_format} is lossless — quality slider doesn't apply.")
                base_quality = 95
            else:
                base_quality = st.slider("Base quality (ignored when target file-size is enabled)", 1, 100, 85, key="base_q")

        # ──────────────────────────────────────
        # Process
        # ──────────────────────────────────────
        if st.button("🚀 Process Image", type="primary", key="process_btn"):
            with st.spinner("Processing…"):
                result_img = img.copy()

                # 0) Background removal & replacement
                if enable_bg:
                    if not REMBG_AVAILABLE:
                        st.error("❌ `rembg` is not installed. Cannot process background.")
                        st.stop()
                    if bg_option == "upload" and bg_upload_img is None:
                        st.error("❌ Please upload a background image.")
                        st.stop()
                    with st.spinner("Removing background… (this may take a moment on first run)"):
                        fg_rgba = remove_background(result_img)
                        result_img = apply_background(fg_rgba, bg_option, custom_bg_color, bg_upload_img)
                    # Force PNG output for transparent background
                    if bg_option == "transparent":
                        out_format = "PNG"

                # 0.5) Freestyle crop
                if enable_crop and cropped_img is not None:
                    result_img = cropped_img.copy()
                    # Preserve mode from background step if needed
                    if enable_bg and result_img.mode != img.mode:
                        pass  # keep whatever mode the bg step produced

                # 1) Resolution preset takes priority over manual dimensions
                if enable_res_preset and res_preset:
                    result_img = result_img.resize((preset_w, preset_h), Image.LANCZOS)
                elif enable_resize and (new_w != orig_w or new_h != orig_h):
                    result_img = result_img.resize((int(new_w), int(new_h)), resample_filter)

                # 2) DPI
                custom_dpi = (dpi_x, dpi_y) if enable_dpi else None

                # 3) Compress to target size or use base quality
                if enable_size_target:
                    target_bytes = int(target_val * (1024 * 1024 if size_unit == "MB" else 1024))
                    result_bytes, used_quality = compress_to_target(result_img, target_bytes, out_format, custom_dpi)
                else:
                    result_bytes = get_image_bytes(result_img, out_format, base_quality, custom_dpi)
                    used_quality = base_quality

                result_size = len(result_bytes)
                result_pil = Image.open(io.BytesIO(result_bytes))
                res_w, res_h = result_pil.size
                _raw_res_dpi = result_pil.info.get("dpi", (72, 72))
                res_dpi = (float(_raw_res_dpi[0]), float(_raw_res_dpi[1]))
                reduction = ((original_size - result_size) / original_size) * 100 if original_size else 0

            # ── Results ──
            st.markdown("---")
            st.subheader("✅ Result")

            rc = st.columns(6)
            result_info = [
                ("Original Size", f'<span class="size-badge size-original">{format_size(original_size)}</span>'),
                ("New Size", f'<span class="size-badge size-result">{format_size(result_size)}</span>'),
                ("Reduction", f"{reduction:.1f} %"),
                ("Dimensions", f"{res_w} × {res_h} px"),
                ("DPI", f"{res_dpi[0]:.0f} × {res_dpi[1]:.0f}"),
                ("Quality Used", str(used_quality)),
            ]
            for col, (label, value) in zip(rc, result_info):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

            # Side-by-side preview
            lc, rc2 = st.columns(2)
            lc.image(img, caption="Original", use_container_width=True)
            rc2.image(result_bytes, caption="Processed", use_container_width=True)

            # Download
            ext = EXT_MAP.get(out_format, "img")
            mime = MIME_MAP.get(out_format, "application/octet-stream")
            dl_name = uploaded.name.rsplit(".", 1)[0] + f"_resized.{ext}"
            st.download_button(
                label=f"⬇️ Download ({format_size(result_size)})",
                data=result_bytes,
                file_name=dl_name,
                mime=mime,
                type="primary",
            )
    else:
        st.info("👆 Upload an image to get started.")
        st.markdown("**Supported formats:** JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, ICO, PPM, PGM, PBM, PCX, TGA, SGI, EPS, DDS")


# ╔══════════════════════════════════════════════╗
# ║         TAB 2: IMAGE TO PDF CONVERTER        ║
# ╚══════════════════════════════════════════════╝
with main_tab2:
    st.subheader("📄 Image to PDF Converter")
    st.caption("Upload one or more images and convert them into a single PDF document.")

    pdf_images = st.file_uploader(
        "Upload images (multiple allowed)",
        type=SUPPORTED_INPUT,
        accept_multiple_files=True,
        key="pdf_img_upload",
        help="Upload images in the order you want them in the PDF. Drag to reorder.",
    )

    if pdf_images:
        st.success(f"✅ {len(pdf_images)} image(s) uploaded")

        # Preview thumbnails
        preview_cols = st.columns(min(6, len(pdf_images)))
        pil_images: list[Image.Image] = []
        for idx, f in enumerate(pdf_images):
            pil_img = Image.open(io.BytesIO(f.getvalue()))
            pil_images.append(pil_img)
            with preview_cols[idx % len(preview_cols)]:
                st.image(pil_img, caption=f.name, use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ PDF Settings")

        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            page_size = st.selectbox("Page Size", ["A4", "A3", "A5", "Letter", "Legal", "Fit to Image"], key="pdf_page")
            orientation = st.selectbox("Orientation", ["Auto", "Portrait", "Landscape"], key="pdf_orient")

        with col_p2:
            fit_mode = st.selectbox("Image Fitting", ["Fit to page", "Fill page (crop)", "Stretch"], key="pdf_fit")
            margin_mm = st.slider("Margin (mm)", 0, 50, 10, key="pdf_margin")

        with col_p3:
            pdf_dpi = st.selectbox("PDF Resolution", [72, 150, 300, 600], index=1, key="pdf_dpi",
                                   help="Lower = smaller file. 150 DPI is good for screen, 300 for print.")
            pdf_quality = st.slider("JPEG Quality", 10, 100, 80, key="pdf_quality",
                                    help="Lower = smaller file. 70-85 is a good balance.")
            pdf_title = st.text_input("PDF Title (optional)", key="pdf_title")

        # Show estimated total original size
        total_original = sum(len(f.getvalue()) for f in pdf_images)
        st.caption(f"📊 Total original image size: **{format_size(total_original)}**")

        if st.button("📄 Generate PDF", type="primary", key="gen_pdf_btn"):
            with st.spinner("Creating PDF…"):
                pdf_bytes = images_to_pdf(
                    pil_images,
                    page_size=page_size,
                    orientation=orientation,
                    margin_mm=margin_mm,
                    fit_mode=fit_mode,
                    dpi=pdf_dpi,
                    jpeg_quality=pdf_quality,
                    title=pdf_title,
                )
                pdf_size = len(pdf_bytes)

            st.markdown("---")
            st.success(f"✅ PDF created — **{len(pil_images)} page(s)** — **{format_size(pdf_size)}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Pages", len(pil_images))
            c2.metric("File Size", format_size(pdf_size))
            c3.metric("Resolution", f"{pdf_dpi} DPI")

            dl_name = pdf_title.strip().replace(" ", "_") if pdf_title.strip() else "images_converted"
            st.download_button(
                label=f"⬇️ Download PDF ({format_size(pdf_size)})",
                data=pdf_bytes,
                file_name=f"{dl_name}.pdf",
                mime="application/pdf",
                type="primary",
                key="dl_pdf_btn",
            )
    else:
        st.info("👆 Upload one or more images to convert to PDF.")
        st.markdown("**Supported formats:** JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, ICO, PPM, PGM, PBM, PCX, TGA, SGI, EPS, DDS")


# ╔══════════════════════════════════════════════╗
# ║          TAB 3: MERGE IMAGES                ║
# ╚══════════════════════════════════════════════╝
with main_tab3:
    st.subheader("🧩 Merge Images")
    st.caption("Upload 2–8 images and combine them into a single image — horizontally, vertically, or as a grid.")

    merge_files = st.file_uploader(
        "Upload 2–8 images to merge",
        type=SUPPORTED_INPUT,
        accept_multiple_files=True,
        key="merge_img_upload",
    )

    if merge_files:
        if len(merge_files) < 2:
            st.warning("⚠️ Please upload at least 2 images to merge.")
        elif len(merge_files) > 8:
            st.warning("⚠️ Maximum 8 images. Using the first 8.")
            merge_files = merge_files[:8]

        if len(merge_files) >= 2:
            st.success(f"✅ {len(merge_files)} images ready to merge")

            # Preview
            preview_cols = st.columns(min(len(merge_files), 8))
            merge_pil: list[Image.Image] = []
            for idx, f in enumerate(merge_files):
                pil_img = Image.open(io.BytesIO(f.getvalue()))
                merge_pil.append(pil_img)
                with preview_cols[idx]:
                    st.image(pil_img, caption=f"{idx+1}. {f.name}", use_container_width=True)

            st.markdown("---")
            st.subheader("⚙️ Merge Settings")

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                merge_direction = st.selectbox(
                    "Layout",
                    ["Horizontal (side by side)", "Vertical (stacked)", "Grid (auto rows/cols)"],
                    key="merge_dir",
                )
                direction_map = {
                    "Horizontal (side by side)": "horizontal",
                    "Vertical (stacked)": "vertical",
                    "Grid (auto rows/cols)": "grid",
                }
                direction_val = direction_map[merge_direction]

                if direction_val == "horizontal":
                    alignment_options = ["Center", "Top", "Bottom"]
                elif direction_val == "vertical":
                    alignment_options = ["Center", "Left", "Right"]
                else:
                    alignment_options = ["Center"]
                merge_align = st.selectbox("Alignment", alignment_options, key="merge_align")

                merge_gap = st.slider("Gap between images (px)", 0, 100, 0, key="merge_gap")

            with col_m2:
                merge_bg = st.color_picker("Background color", "#FFFFFF", key="merge_bg")
                merge_bg_rgb = tuple(int(merge_bg.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

                merge_out_fmt = st.selectbox("Output format", ["JPG", "PDF"], key="merge_fmt")
                merge_quality = 90
                if merge_out_fmt == "JPG":
                    merge_quality = st.slider("Quality", 10, 100, 90, key="merge_q")

                # Optionally resize all images to same height/width
                merge_normalize = st.checkbox(
                    "Resize all images to same height (horizontal) / width (vertical)",
                    value=True, key="merge_norm",
                    help="Scales images so they line up evenly."
                )

            if st.button("🧩 Merge Images", type="primary", key="merge_btn"):
                with st.spinner("Merging images…"):
                    # Optionally normalize sizes
                    final_imgs = list(merge_pil)
                    if merge_normalize and direction_val == "horizontal":
                        target_h = min(im.height for im in final_imgs)
                        final_imgs = [
                            im.resize((int(im.width * target_h / im.height), target_h), Image.LANCZOS)
                            for im in final_imgs
                        ]
                    elif merge_normalize and direction_val == "vertical":
                        target_w = min(im.width for im in final_imgs)
                        final_imgs = [
                            im.resize((target_w, int(im.height * target_w / im.width)), Image.LANCZOS)
                            for im in final_imgs
                        ]
                    elif merge_normalize and direction_val == "grid":
                        target_w = min(im.width for im in final_imgs)
                        target_h = min(im.height for im in final_imgs)
                        final_imgs = [
                            im.resize((target_w, target_h), Image.LANCZOS)
                            for im in final_imgs
                        ]

                    merged_img, merged_bytes = merge_images(
                        final_imgs,
                        direction=direction_val,
                        alignment=merge_align.lower(),
                        gap=merge_gap,
                        bg_color=merge_bg_rgb,
                        output_format=merge_out_fmt,
                        quality=merge_quality,
                    )
                    merged_size = len(merged_bytes)

                st.markdown("---")
                st.success(f"✅ Merged! — **{merged_img.width} × {merged_img.height} px** — **{format_size(merged_size)}**")

                st.image(merged_img, caption="Merged Result", use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Dimensions", f"{merged_img.width} × {merged_img.height}")
                c2.metric("File Size", format_size(merged_size))
                c3.metric("Images Merged", len(final_imgs))

                if merge_out_fmt == "PDF":
                    # Convert merged image to a compact PDF
                    pdf_img = merged_img.convert("RGB")
                    pdf_img = _jpeg_compress_image(pdf_img, merge_quality)
                    pdf_buf = io.BytesIO()
                    pdf_img.save(pdf_buf, format="PDF", resolution=150)
                    pdf_data = pdf_buf.getvalue()
                    st.download_button(
                        label=f"⬇️ Download Merged as PDF ({format_size(len(pdf_data))})",
                        data=pdf_data,
                        file_name=f"merged_{len(final_imgs)}images.pdf",
                        mime="application/pdf",
                        type="primary",
                        key="dl_merge_btn",
                    )
                else:
                    st.download_button(
                        label=f"⬇️ Download Merged as JPG ({format_size(merged_size)})",
                        data=merged_bytes,
                        file_name=f"merged_{len(final_imgs)}images.jpg",
                        mime="image/jpeg",
                        type="primary",
                        key="dl_merge_btn",
                    )
    else:
        st.info("👆 Upload 2–8 images to merge them into one.")
        st.markdown("**Supported formats:** JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, ICO, PPM, PGM, PBM, PCX, TGA, SGI, EPS, DDS")


# ╔══════════════════════════════════════════════╗
# ║           TAB 4: MERGE PDFs                 ║
# ╚══════════════════════════════════════════════╝
with main_tab4:
    st.subheader("📚 Merge PDFs")
    st.caption("Upload 2–8 PDF files and combine them into a single PDF.")

    pdf_merge_files = st.file_uploader(
        "Upload 2–8 PDF files to merge",
        type=["pdf"],
        accept_multiple_files=True,
        key="merge_pdf_upload",
    )

    if pdf_merge_files:
        if len(pdf_merge_files) < 2:
            st.warning("⚠️ Please upload at least 2 PDFs to merge.")
        elif len(pdf_merge_files) > 8:
            st.warning("⚠️ Maximum 8 PDFs. Using the first 8.")
            pdf_merge_files = pdf_merge_files[:8]

        if len(pdf_merge_files) >= 2:
            st.success(f"✅ {len(pdf_merge_files)} PDFs ready to merge")

            # Show file info
            total_input_size = 0
            for idx, f in enumerate(pdf_merge_files):
                fsize = len(f.getvalue())
                total_input_size += fsize
                st.write(f"**{idx+1}.** {f.name} — {format_size(fsize)}")

            st.caption(f"📊 Total input size: **{format_size(total_input_size)}**")

            if st.button("📚 Merge PDFs", type="primary", key="merge_pdf_btn"):
                with st.spinner("Merging PDFs…"):
                    try:
                        pdf_buffers = [io.BytesIO(f.getvalue()) for f in pdf_merge_files]
                        merged_pdf_bytes, total_pages = merge_pdfs(pdf_buffers)
                        merged_pdf_size = len(merged_pdf_bytes)
                    except ImportError:
                        st.error("❌ `pypdf` is required for PDF merging. Install it: `pip install pypdf`")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Failed to merge PDFs: {e}")
                        st.stop()

                st.markdown("---")
                st.success(f"✅ Merged! — **{total_pages} pages** — **{format_size(merged_pdf_size)}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Files Merged", len(pdf_merge_files))
                c2.metric("Total Pages", total_pages)
                c3.metric("Output Size", format_size(merged_pdf_size))

                st.download_button(
                    label=f"⬇️ Download Merged PDF ({format_size(merged_pdf_size)})",
                    data=merged_pdf_bytes,
                    file_name="merged_document.pdf",
                    mime="application/pdf",
                    type="primary",
                    key="dl_merge_pdf_btn",
                )
    else:
        st.info("👆 Upload 2–8 PDF files to merge them into one full pdf.")
