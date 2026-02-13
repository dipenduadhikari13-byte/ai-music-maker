import streamlit as st
from pypdf import PdfReader, PdfWriter
from PIL import Image
import io
import subprocess
import shutil
import tempfile
import os
import math

st.set_page_config(page_title="PDF Resizer & Compressor", page_icon="📄", layout="wide")

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
    .preset-card {
        background: #1a1a2e; border-radius: 10px; padding: 14px;
        border: 1px solid #444; margin-bottom: 8px;
    }
    .preset-title { font-weight: 700; font-size: 1.05em; }
    .preset-desc  { color: #aaa; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF Resizer & Compressor")
st.caption("Upload a PDF → choose compression level & target size → download the smaller file.")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


GS_AVAILABLE = shutil.which("gs") is not None

# Ghostscript presets map to /screen, /ebook, /printer, /prepress
GS_PRESETS = {
    "Ultra High Compression": {
        "gs_setting": "/screen",
        "image_dpi": 72,
        "image_quality": 15,
        "description": "Smallest file. Images down-sampled to 72 DPI, heavy JPEG compression. Best for quick email / preview.",
        "badge_color": "#ff4b4b",
    },
    "High Compression": {
        "gs_setting": "/ebook",
        "image_dpi": 150,
        "image_quality": 35,
        "description": "Good balance. Images at 150 DPI with moderate JPEG. Ideal for on-screen reading.",
        "badge_color": "#ff9800",
    },
    "Normal Compression": {
        "gs_setting": "/printer",
        "image_dpi": 300,
        "image_quality": 60,
        "description": "Print-friendly. Images at 300 DPI with light compression. Suitable for most documents.",
        "badge_color": "#2196f3",
    },
    "Light Compression": {
        "gs_setting": "/prepress",
        "image_dpi": 300,
        "image_quality": 80,
        "description": "Minimal quality loss. Keeps images near original. Best for archival / high-quality prints.",
        "badge_color": "#4caf50",
    },
}


def compress_pdf_ghostscript(input_bytes: bytes, gs_setting: str, image_dpi: int) -> bytes:
    """Use Ghostscript for professional-grade PDF compression."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
        tmp_in.write(input_bytes)
        tmp_in.flush()
        cmd = [
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            "-dNOPAUSE", "-dQUIET", "-dBATCH",
            f"-dPDFSETTINGS={gs_setting}",
            f"-dDownsampleColorImages=true",
            f"-dColorImageResolution={image_dpi}",
            f"-dDownsampleGrayImages=true",
            f"-dGrayImageResolution={image_dpi}",
            f"-dDownsampleMonoImages=true",
            f"-dMonoImageResolution={image_dpi}",
            f"-sOutputFile={tmp_out.name}",
            tmp_in.name,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            with open(tmp_out.name, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_in.name)
            os.unlink(tmp_out.name)


def compress_pdf_pypdf(input_bytes: bytes, image_quality: int, image_dpi: int, remove_metadata: bool) -> bytes:
    """Pure-Python PDF compression using pypdf + Pillow image recompression."""
    reader = PdfReader(io.BytesIO(input_bytes))
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    # Compress content streams
    for page in writer.pages:
        page.compress_content_streams()

    # Recompress embedded images
    recompressed = 0
    for page in writer.pages:
        if "/Resources" not in page or "/XObject" not in page["/Resources"]:
            continue
        x_objects = page["/Resources"]["/XObject"].get_object()
        for obj_name in x_objects:
            x_obj = x_objects[obj_name].get_object()
            if x_obj.get("/Subtype") != "/Image":
                continue
            try:
                width = int(x_obj["/Width"])
                height = int(x_obj["/Height"])

                # Scale dimensions based on target DPI vs assumed 300 source
                scale = min(1.0, image_dpi / 300.0)
                new_w = max(1, int(width * scale))
                new_h = max(1, int(height * scale))

                # Determine the current filter to choose decoding strategy
                current_filter = x_obj.get("/Filter", "")
                if isinstance(current_filter, list):
                    current_filter = str(current_filter[-1]) if current_filter else ""
                else:
                    current_filter = str(current_filter)

                img = None
                original_stream_size = len(x_obj._data) if hasattr(x_obj, '_data') and x_obj._data else 0

                # Strategy 1: For JPEG images, open directly from encoded stream
                if "/DCTDecode" in current_filter and hasattr(x_obj, '_data') and x_obj._data:
                    try:
                        img = Image.open(io.BytesIO(x_obj._data))
                    except Exception:
                        pass

                # Strategy 2: Decode raw pixel data and reconstruct
                if img is None:
                    data = x_obj.get_data()
                    color_space = x_obj.get("/ColorSpace", "/DeviceRGB")
                    if isinstance(color_space, list):
                        color_space = str(color_space[0])
                    else:
                        color_space = str(color_space)

                    if "/DeviceRGB" in color_space:
                        mode = "RGB"
                    elif "/DeviceGray" in color_space:
                        mode = "L"
                    elif "/DeviceCMYK" in color_space:
                        mode = "CMYK"
                    else:
                        mode = "RGB"

                    expected_size = width * height * len(mode)
                    if len(data) >= expected_size:
                        try:
                            img = Image.frombytes(mode, (width, height), data)
                        except Exception:
                            pass

                    # Strategy 3: Let Pillow auto-detect from decoded bytes
                    if img is None:
                        try:
                            img = Image.open(io.BytesIO(data))
                        except Exception:
                            continue

                if img is None:
                    continue

                # Resize if needed
                if new_w != width or new_h != height:
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                # Convert to a JPEG-compatible mode
                if img.mode in ("CMYK", "P", "RGBA", "LA"):
                    img = img.convert("RGB")
                elif img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=image_quality, optimize=True)
                new_data = buf.getvalue()

                # Only replace if the new image data is actually smaller
                if original_stream_size > 0 and len(new_data) >= original_stream_size:
                    continue  # skip — re-encoding would increase size

                x_obj._data = new_data
                x_obj["/Filter"] = "/DCTDecode"
                x_obj["/Width"] = new_w
                x_obj["/Height"] = new_h
                x_obj["/ColorSpace"] = "/DeviceGray" if img.mode == "L" else "/DeviceRGB"
                x_obj["/BitsPerComponent"] = 8
                x_obj["/Length"] = len(new_data)
                recompressed += 1
            except Exception:
                continue  # skip problematic images gracefully

    # Remove metadata if requested
    if remove_metadata:
        writer.add_metadata({
            "/Producer": "",
            "/Creator": "",
            "/Author": "",
            "/Subject": "",
            "/Keywords": "",
        })

    # Remove duplicate objects
    try:
        writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    except Exception:
        pass

    out = io.BytesIO()
    writer.write(out)
    result = out.getvalue()

    # If compression actually increased the size, return the original
    if len(result) >= len(input_bytes):
        return input_bytes

    return result


def iterative_compress_to_target(
    input_bytes: bytes, target_bytes: int, preset_key: str, use_gs: bool, remove_meta: bool
) -> tuple[bytes, dict]:
    """Iteratively compress, lowering quality/DPI until target size is met (or floor reached)."""
    preset = GS_PRESETS[preset_key]
    dpi = preset["image_dpi"]
    quality = preset["image_quality"]
    best_result = input_bytes
    attempts = []

    for attempt in range(6):
        if use_gs and GS_AVAILABLE:
            result = compress_pdf_ghostscript(input_bytes, preset["gs_setting"], dpi)
        else:
            result = compress_pdf_pypdf(input_bytes, quality, dpi, remove_meta)

        attempts.append({"dpi": dpi, "quality": quality, "size": len(result)})

        if len(result) <= target_bytes:
            return result, {"attempts": attempts, "dpi": dpi, "quality": quality, "hit_target": True}

        if len(result) < len(best_result):
            best_result = result

        # Ratchet down
        dpi = max(36, int(dpi * 0.7))
        quality = max(5, int(quality * 0.65))

    return best_result, {"attempts": attempts, "dpi": dpi, "quality": quality, "hit_target": False}


# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    raw_bytes = uploaded.getvalue()
    original_size = len(raw_bytes)

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        num_pages = len(reader.pages)
        pdf_version = reader.pdf_header
        has_images = False
        image_count = 0
        for pg in reader.pages:
            if "/Resources" in pg and "/XObject" in pg["/Resources"]:
                try:
                    xobjs = pg["/Resources"]["/XObject"].get_object()
                    for name in xobjs:
                        if xobjs[name].get_object().get("/Subtype") == "/Image":
                            image_count += 1
                            has_images = True
                except Exception:
                    pass
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()

    # ── Original Info ──
    st.markdown("---")
    st.subheader("📊 Original PDF Info")

    cols = st.columns(5)
    info_items = [
        ("File Size", format_size(original_size)),
        ("Pages", str(num_pages)),
        ("Images Found", str(image_count)),
        ("PDF Version", pdf_version),
        ("Engine", "Ghostscript ✅" if GS_AVAILABLE else "pypdf (pure Python)"),
    ]
    for col, (label, value) in zip(cols, info_items):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    if not GS_AVAILABLE:
        st.info(
            "💡 **Ghostscript not detected.** Using pure-Python compression (pypdf + Pillow). "
            "For best results, install Ghostscript: `sudo apt install ghostscript`"
        )

    # ──────────────────────────────────────────
    # Settings
    # ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Compression Settings")

    tab_preset, tab_target = st.tabs(["🎚️ Compression Preset", "🎯 Target File Size"])

    # --- Preset selector ---
    with tab_preset:
        st.markdown("Choose a compression level:")
        preset_choice = st.radio(
            "Preset",
            list(GS_PRESETS.keys()),
            index=1,
            format_func=lambda x: x,
            key="pdf_preset",
            label_visibility="collapsed",
        )
        for name, p in GS_PRESETS.items():
            selected = "→ " if name == preset_choice else ""
            st.markdown(
                f'<div class="preset-card">'
                f'<span class="preset-title" style="color:{p["badge_color"]}">{selected}{name}</span> '
                f'&nbsp;·&nbsp; {p["image_dpi"]} DPI &nbsp;·&nbsp; Q{p["image_quality"]}<br>'
                f'<span class="preset-desc">{p["description"]}</span></div>',
                unsafe_allow_html=True,
            )

    # --- Target size ---
    with tab_target:
        size_unit = st.radio("Unit", ["KB", "MB"], horizontal=True, key="pdf_sz_unit")
        if size_unit == "MB":
            min_val = 0.01
            max_val = 200.0
            default_val = round(original_size / (1024 * 1024), 2)
            step = 0.01
        else:
            min_val = 1.0
            max_val = 51200.0
            default_val = round(original_size / 1024, 1)
            step = 10.0
        # Clamp default value to valid range
        target_default = round(max(min_val, min(default_val * 0.5, max_val)), 2)
        target_val = st.number_input(
            f"Desired file size ({size_unit})",
            min_value=min_val,
            max_value=max_val,
            value=target_default,
            step=step,
            key=f"pdf_target_size_{size_unit}",  # unit-specific key avoids stale value crash
        )
        enable_target = st.checkbox("Enable target file-size mode", value=False, key="pdf_en_target",
                                    help="Iteratively compresses until the file is at or below your target size.")
        st.caption("The compressor will progressively lower quality/DPI across multiple passes to reach your target.")

    # --- Extra options ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    remove_meta = c1.checkbox("Strip metadata (author, producer, etc.)", value=False, key="pdf_rm_meta")
    use_gs = c2.checkbox("Use Ghostscript engine (if available)", value=GS_AVAILABLE, disabled=not GS_AVAILABLE, key="pdf_use_gs")

    # ──────────────────────────────────────────
    # Process
    # ──────────────────────────────────────────
    if st.button("🚀 Compress PDF", type="primary"):
        with st.spinner("Compressing… this may take a moment for large PDFs."):
            if enable_target:
                target_bytes = int(target_val * (1024 * 1024 if size_unit == "MB" else 1024))
                result_bytes, info = iterative_compress_to_target(
                    raw_bytes, target_bytes, preset_choice, use_gs, remove_meta
                )
                if not info["hit_target"]:
                    st.warning(
                        f"⚠️ Could not reach the target of **{format_size(target_bytes)}**. "
                        f"Best achieved: **{format_size(len(result_bytes))}** after {len(info['attempts'])} passes."
                    )
            else:
                preset = GS_PRESETS[preset_choice]
                if use_gs and GS_AVAILABLE:
                    result_bytes = compress_pdf_ghostscript(raw_bytes, preset["gs_setting"], preset["image_dpi"])
                    # Ghostscript can also increase size for already-optimized PDFs
                    if len(result_bytes) >= original_size:
                        result_bytes = raw_bytes
                else:
                    result_bytes = compress_pdf_pypdf(raw_bytes, preset["image_quality"], preset["image_dpi"], remove_meta)
                info = {"hit_target": None}

            result_size = len(result_bytes)
            reduction = ((original_size - result_size) / original_size) * 100 if original_size else 0

        # ── Results ──
        st.markdown("---")
        st.subheader("✅ Compression Result")

        rc = st.columns(5)
        result_info = [
            ("Original", f'<span class="size-badge size-original">{format_size(original_size)}</span>'),
            ("Compressed", f'<span class="size-badge size-result">{format_size(result_size)}</span>'),
            ("Reduction", f"{reduction:.1f} %"),
            ("Preset", preset_choice.split()[0]),
            ("Pages", str(num_pages)),
        ]
        for col, (label, value) in zip(rc, result_info):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        if result_size >= original_size:
            st.info("ℹ️ The compressed file is not smaller — the PDF may already be well-optimized or contain mostly vector/text content.")

        # Attempt details (target mode)
        if enable_target and info.get("attempts"):
            with st.expander("🔍 Compression passes detail"):
                for i, a in enumerate(info["attempts"], 1):
                    st.write(f"Pass {i}: DPI={a['dpi']}, Quality={a['quality']} → {format_size(a['size'])}")

        # Download
        dl_name = uploaded.name.rsplit(".", 1)[0] + "_compressed.pdf"
        st.download_button(
            label=f"⬇️ Download Compressed PDF ({format_size(result_size)})",
            data=result_bytes,
            file_name=dl_name,
            mime="application/pdf",
            type="primary",
        )
else:
    st.info("👆 Upload a PDF to get started.")
