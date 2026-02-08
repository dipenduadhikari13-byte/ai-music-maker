import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
import math
import tempfile
import os

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
        save_kwargs["optimize"] = True
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


def compress_to_target(img: Image.Image, target_bytes: int, fmt: str, dpi: tuple | None = None) -> tuple[bytes, int]:
    """Binary-search the quality parameter to hit the target file size."""
    real_fmt = INTERNAL_FMT.get(fmt, fmt)
    if real_fmt in ("PNG", "BMP", "GIF", "PPM", "ICO"):
        # These formats don't have a quality parameter to binary-search
        data = get_image_bytes(img, fmt, 95, dpi)
        return data, 95
    lo, hi = 1, 100
    best_data = get_image_bytes(img, fmt, hi, dpi)
    if len(best_data) <= target_bytes:
        return best_data, hi
    best_quality = hi

    for _ in range(14):
        mid = (lo + hi) // 2
        data = get_image_bytes(img, fmt, mid, dpi)
        if len(data) <= target_bytes:
            best_data = data
            best_quality = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best_data, best_quality


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
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() so buffer can be freed


def images_to_pdf(images: list[Image.Image], page_size: str = "A4", orientation: str = "Auto",
                  margin_mm: int = 10, fit_mode: str = "Fit to page", dpi: int = 300,
                  jpeg_quality: int = 85, title: str = "") -> bytes:
    """Convert one or more PIL Images into a single PDF with JPEG compression."""
    # Page sizes in mm
    PAGE_SIZES = {
        "A4": (210, 297), "A3": (297, 420), "A5": (148, 210),
        "Letter": (216, 279), "Legal": (216, 356),
        "Fit to Image": None,
    }
    page_mm = PAGE_SIZES.get(page_size)

    pdf_pages: list[Image.Image] = []

    for img in images:
        img = img.convert("RGB")

        if page_mm is None:
            # Fit to Image mode — compress and use as-is (no upscaling)
            compressed = _jpeg_compress_image(img, jpeg_quality)
            pdf_pages.append(compressed)
            continue

        pw_mm, ph_mm = page_mm
        if orientation == "Landscape":
            pw_mm, ph_mm = ph_mm, pw_mm
        elif orientation == "Auto":
            if img.width > img.height:
                pw_mm, ph_mm = max(pw_mm, ph_mm), min(pw_mm, ph_mm)
            else:
                pw_mm, ph_mm = min(pw_mm, ph_mm), max(pw_mm, ph_mm)

        # Convert mm to pixels at target DPI
        pw_px = int(pw_mm / 25.4 * dpi)
        ph_px = int(ph_mm / 25.4 * dpi)
        margin_px = int(margin_mm / 25.4 * dpi)

        usable_w = pw_px - 2 * margin_px
        usable_h = ph_px - 2 * margin_px

        if fit_mode == "Fit to page":
            # Never upscale beyond original — cap the ratio at 1.0
            ratio = min(usable_w / img.width, usable_h / img.height, 1.0)
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            resized = img.resize((new_w, new_h), Image.LANCZOS) if ratio < 1.0 else img
        elif fit_mode == "Fill page (crop)":
            ratio = max(usable_w / img.width, usable_h / img.height)
            # Allow upscale only when needed to fill, but cap at 2x
            ratio = min(ratio, 2.0)
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - usable_w) // 2
            top = (new_h - usable_h) // 2
            resized = resized.crop((left, top, left + usable_w, top + usable_h))
            new_w, new_h = usable_w, usable_h
        else:  # Stretch
            resized = img.resize((usable_w, usable_h), Image.LANCZOS)
            new_w, new_h = usable_w, usable_h

        # Build the page at the image's actual size (not the full DPI canvas)
        # This avoids huge white bitmaps. We use DPI metadata to tell the
        # PDF reader how large to display the page.
        page = Image.new("RGB", (pw_px, ph_px), (255, 255, 255))
        x_offset = margin_px + (usable_w - new_w) // 2
        y_offset = margin_px + (usable_h - new_h) // 2
        page.paste(resized, (x_offset, y_offset))

        # JPEG-compress the page to reduce embedded data size
        page = _jpeg_compress_image(page, jpeg_quality)
        pdf_pages.append(page)

    # Save as PDF
    buf = io.BytesIO()
    if len(pdf_pages) == 1:
        pdf_pages[0].save(buf, format="PDF", resolution=dpi, title=title or "Converted PDF")
    else:
        pdf_pages[0].save(
            buf, format="PDF", resolution=dpi, title=title or "Converted PDF",
            save_all=True, append_images=pdf_pages[1:]
        )
    return buf.getvalue()


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
    if save_fmt in ("JPEG", "WEBP"):
        save_img.save(buf, format=save_fmt, quality=quality, optimize=True)
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
        orig_dpi = img.info.get("dpi", (72, 72))
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

        tab_size, tab_dim, tab_res, tab_dpi = st.tabs([
            "📦 Target File Size",
            "📐 Dimensions & Aspect Ratio",
            "🖥️ Resolution Presets",
            "🔍 DPI",
        ])

        # --- Target file size ---
        with tab_size:
            size_unit = st.radio("Unit", ["KB", "MB"], horizontal=True, key="sz_unit")
            max_val = 100.0 if size_unit == "MB" else 10240.0
            default_val = round(original_size / (1024 * 1024), 2) if size_unit == "MB" else round(original_size / 1024, 1)
            target_val = st.number_input(
                f"Desired file size ({size_unit})",
                min_value=1.0,
                max_value=max_val,
                value=min(default_val, max_val),
                step=0.5 if size_unit == "MB" else 5.0,
                key="target_size",
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
                res_dpi = result_pil.info.get("dpi", (72, 72))
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
        st.info("👆 Upload 2–8 PDF files to merge them into one.")
