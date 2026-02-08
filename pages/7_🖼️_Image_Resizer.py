import streamlit as st
from PIL import Image
import io
import math

st.set_page_config(page_title="Image Resizer & Compressor", page_icon="🖼️", layout="wide")

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

st.title("🖼️ Image Resizer & Compressor")
st.caption("Upload an image → resize, compress, change DPI — download the result.")

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


def get_image_bytes(img: Image.Image, fmt: str, quality: int, dpi: tuple | None = None) -> bytes:
    """Export a PIL Image to bytes with given format, quality, and optional DPI."""
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt, "quality": quality}
    if dpi:
        save_kwargs["dpi"] = dpi
    if fmt == "PNG":
        save_kwargs.pop("quality", None)  # PNG uses compress_level, not quality
        # Map quality (1-100) → compress_level (9-0)  higher quality = less compression
        save_kwargs["compress_level"] = max(0, min(9, 9 - int(quality / 11.2)))
    if fmt == "WEBP":
        save_kwargs["method"] = 4  # balanced speed/size
    # Ensure RGB for JPEG
    if fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def compress_to_target(img: Image.Image, target_bytes: int, fmt: str, dpi: tuple | None = None) -> tuple[bytes, int]:
    """Binary-search the quality parameter to hit the target file size."""
    lo, hi = 1, 100
    best_data = get_image_bytes(img, fmt, hi, dpi)
    # If even max compression is already under target, return it
    if len(best_data) <= target_bytes:
        return best_data, hi
    best_quality = hi

    for _ in range(14):  # 14 iterations → precision of ~1 quality unit
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
}

# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if uploaded:
    raw_bytes = uploaded.getvalue()
    original_size = len(raw_bytes)
    img = Image.open(io.BytesIO(raw_bytes))
    orig_w, orig_h = img.size
    orig_dpi = img.info.get("dpi", (72, 72))
    orig_format = img.format or uploaded.name.rsplit(".", 1)[-1].upper()
    if orig_format == "JPG":
        orig_format = "JPEG"

    # ── Show original info ──
    st.markdown("---")
    st.subheader("📊 Original Image Info")

    cols = st.columns(5)
    info_items = [
        ("File Size", format_size(original_size)),
        ("Dimensions", f"{orig_w} × {orig_h} px"),
        ("Aspect Ratio", f"{orig_w / math.gcd(orig_w, orig_h):.0f}:{orig_h / math.gcd(orig_w, orig_h):.0f}"),
        ("Format", orig_format),
        ("DPI", f"{orig_dpi[0]:.0f} × {orig_dpi[1]:.0f}"),
    ]
    for col, (label, value) in zip(cols, info_items):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.image(img, caption="Original", use_container_width=True)

    # ──────────────────────────────────────────
    # Settings
    # ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Resize & Compress Settings")

    tab_size, tab_dim, tab_dpi = st.tabs(["📦 Target File Size", "📐 Dimensions & Aspect Ratio", "🔍 DPI"])

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
            # Lock ratio: let user pick width, auto-calc height
            new_w = st.number_input("Width (px)", min_value=1, value=orig_w, step=1, key="dim_w_locked")
            new_h = int(new_w * aspect[1] / aspect[0])
            st.info(f"Height auto-calculated to **{new_h} px** for {aspect_choice} ratio.")
        else:
            c1, c2 = st.columns(2)
            new_w = c1.number_input("Width (px)", min_value=1, value=orig_w, step=1, key="dim_w")
            new_h = c2.number_input("Height (px)", min_value=1, value=orig_h, step=1, key="dim_h")

        enable_resize = st.checkbox("Enable dimension resize", value=False, key="en_dim")

        resample_options = {"Lanczos (best quality)": Image.LANCZOS, "Bilinear": Image.BILINEAR, "Nearest (fastest)": Image.NEAREST}
        resample_name = st.selectbox("Resampling filter", list(resample_options.keys()), key="resample")
        resample_filter = resample_options[resample_name]

    # --- DPI ---
    with tab_dpi:
        c1, c2 = st.columns(2)
        dpi_x = c1.number_input("Horizontal DPI", min_value=1, max_value=2400, value=int(orig_dpi[0]), step=1, key="dpi_x")
        dpi_y = c2.number_input("Vertical DPI", min_value=1, max_value=2400, value=int(orig_dpi[1]), step=1, key="dpi_y")
        enable_dpi = st.checkbox("Change DPI metadata", value=False, key="en_dpi")
        st.caption("DPI change only updates metadata (for print). It does NOT resample pixels.")

    # --- Output format & quality ---
    st.markdown("---")
    out_fmt_options = ["JPEG", "PNG", "WEBP"]
    default_idx = out_fmt_options.index(orig_format) if orig_format in out_fmt_options else 0
    out_format = st.selectbox("Output format", out_fmt_options, index=default_idx, key="out_fmt")
    base_quality = st.slider("Base quality (ignored when target file-size is enabled)", 1, 100, 85, key="base_q")

    # ──────────────────────────────────────────
    # Process
    # ──────────────────────────────────────────
    if st.button("🚀 Process Image", type="primary"):
        with st.spinner("Processing…"):
            result_img = img.copy()

            # 1) Resize dimensions
            if enable_resize and (new_w != orig_w or new_h != orig_h):
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
        ext_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
        dl_name = uploaded.name.rsplit(".", 1)[0] + f"_resized.{ext_map[out_format]}"
        st.download_button(
            label=f"⬇️ Download ({format_size(result_size)})",
            data=result_bytes,
            file_name=dl_name,
            mime=f"image/{ext_map[out_format]}",
            type="primary",
        )
else:
    st.info("👆 Upload an image to get started.")
