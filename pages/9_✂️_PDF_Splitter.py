import streamlit as st
from pypdf import PdfReader, PdfWriter
import io
import zipfile
import re
import math

st.set_page_config(page_title="PDF Splitter & Extractor", page_icon="✂️", layout="wide")

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #333;
    }
    .metric-label { color: #888; font-size: 0.85em; }
    .metric-value { color: #fff; font-size: 1.4em; font-weight: 700; }
    .mode-card {
        background: #1a1a2e; border-radius: 10px; padding: 16px;
        border: 1px solid #444; margin-bottom: 10px;
    }
    .mode-title { font-weight: 700; font-size: 1.1em; color: #4fc3f7; }
    .mode-desc  { color: #aaa; font-size: 0.88em; margin-top: 4px; }
    .page-chip {
        display: inline-block; padding: 4px 12px; margin: 3px;
        border-radius: 16px; font-size: 0.85em; font-weight: 600;
        background: #263238; color: #80cbc4; border: 1px solid #4db6ac;
    }
    .range-badge {
        display: inline-block; padding: 5px 14px; margin: 3px;
        border-radius: 18px; font-weight: 600; font-size: 0.92em;
        background: #1a237e22; color: #7986cb; border: 1px solid #5c6bc0;
    }
    .preview-header {
        background: #1b1b2f; border-radius: 10px; padding: 12px 18px;
        border: 1px solid #333; margin: 10px 0;
    }
    .size-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 1.05em; margin: 4px 0;
    }
    .size-original { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b; }
    .size-result   { background: #00c85322; color: #00c853; border: 1px solid #00c853; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("✂️ PDF Splitter & Extractor")
st.caption("Split, extract, and reorganize PDF pages — lossless, no quality loss, no file-size inflation.")


# ─── Helpers ────────────────────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Format byte count to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def parse_page_spec(spec: str, total_pages: int) -> list[int]:
    """
    Parse an iLovePDF-style page specification string.

    Supported formats:
        - Single page:         "5"
        - Range:               "3-7"
        - Open-ended range:    "5-"  (page 5 to last)
        - Open-started range:  "-5"  (page 1 to 5)
        - Last N with 'last':  "last3" (last 3 pages)
        - Even/Odd:            "odd", "even"
        - Comma-separated mix: "1, 3-5, 8, 10-"
        - Reverse range:       "7-3" → [7,6,5,4,3]
        - "all"                all pages

    Returns 0-indexed page numbers. Raises ValueError on bad input.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Page specification is empty.")

    # Handle special keywords
    lower = spec.lower().strip()
    if lower == "all":
        return list(range(total_pages))
    if lower == "odd":
        return [i for i in range(total_pages) if (i + 1) % 2 == 1]
    if lower == "even":
        return [i for i in range(total_pages) if (i + 1) % 2 == 0]

    # Handle "lastN" pattern
    last_match = re.match(r'^last\s*(\d+)$', lower)
    if last_match:
        n = int(last_match.group(1))
        if n < 1:
            raise ValueError("'last' count must be >= 1.")
        if n > total_pages:
            n = total_pages
        return list(range(total_pages - n, total_pages))

    pages: list[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]

    for part in parts:
        # Range: "a-b", "-b", "a-"
        if "-" in part:
            # Could be negative start like "-5" meaning 1-5
            range_match = re.match(r'^(\d*)\s*-\s*(\d*)$', part)
            if not range_match:
                raise ValueError(f"Invalid range: '{part}'. Use format like '3-7', '-5', or '5-'.")
            start_str, end_str = range_match.group(1), range_match.group(2)
            start = int(start_str) if start_str else 1
            end = int(end_str) if end_str else total_pages

            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be >= 1. Got range '{part}'.")
            if start > total_pages:
                raise ValueError(f"Start page {start} exceeds document length ({total_pages} pages).")
            if end > total_pages:
                raise ValueError(f"End page {end} exceeds document length ({total_pages} pages).")

            if start <= end:
                pages.extend(range(start - 1, end))  # convert to 0-indexed
            else:
                # Reverse range: 7-3 → [7,6,5,4,3]
                pages.extend(range(start - 1, end - 2, -1))
        else:
            # Single page number
            if not part.isdigit():
                raise ValueError(f"Invalid page number: '{part}'. Must be a positive integer.")
            pg = int(part)
            if pg < 1:
                raise ValueError("Page numbers must be >= 1.")
            if pg > total_pages:
                raise ValueError(f"Page {pg} exceeds document length ({total_pages} pages).")
            pages.append(pg - 1)

    if not pages:
        raise ValueError("No valid pages found in specification.")

    return pages


def build_pdf_from_pages(reader: PdfReader, page_indices: list[int]) -> bytes:
    """
    Build a new PDF containing only the specified pages (0-indexed).

    Uses lossless page cloning — no re-encoding of content streams or images.
    Preserves all page content, annotations, links, form fields, and media boxes.
    """
    writer = PdfWriter()

    for idx in page_indices:
        if 0 <= idx < len(reader.pages):
            writer.add_page(reader.pages[idx])

    # Copy document metadata losslessly
    if reader.metadata:
        try:
            meta = {}
            for key in ["/Title", "/Author", "/Subject", "/Keywords",
                        "/Creator", "/Producer", "/CreationDate", "/ModDate"]:
                val = reader.metadata.get(key)
                if val:
                    meta[key] = val
            if meta:
                writer.add_metadata(meta)
        except Exception:
            pass

    # Preserve the PDF version
    try:
        if hasattr(reader, 'pdf_header'):
            version = reader.pdf_header.replace('%PDF-', '')
            if version:
                writer._header = reader.pdf_header.encode() if isinstance(reader.pdf_header, str) else reader.pdf_header
    except Exception:
        pass

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def split_into_ranges(reader: PdfReader, ranges: list[list[int]]) -> list[tuple[str, bytes]]:
    """Split a PDF into multiple PDFs based on page ranges. Returns list of (label, bytes)."""
    results = []
    for i, page_indices in enumerate(ranges):
        if not page_indices:
            continue
        label = format_range_label(page_indices)
        pdf_bytes = build_pdf_from_pages(reader, page_indices)
        results.append((label, pdf_bytes))
    return results


def format_range_label(page_indices: list[int]) -> str:
    """Create a human-readable label from 0-indexed page indices."""
    if not page_indices:
        return "empty"
    # Convert to 1-indexed for display
    pages = [p + 1 for p in page_indices]
    if len(pages) == 1:
        return f"page_{pages[0]}"

    # Try to detect contiguous ranges
    ranges = []
    start = pages[0]
    prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = p
            prev = p
    ranges.append((start, prev))

    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")
    return "pages_" + "_".join(parts)


def parse_split_ranges(spec: str, total_pages: int) -> list[list[int]]:
    """
    Parse range specification where each semicolon-separated group becomes
    a separate PDF output.

    E.g. "1-3; 4-6; 7-10" → three PDFs
         "1,3,5; 2,4,6"    → two PDFs
    """
    groups = [g.strip() for g in spec.split(";") if g.strip()]
    if not groups:
        raise ValueError("No ranges specified. Separate groups with semicolons (;).")

    results = []
    for group in groups:
        pages = parse_page_spec(group, total_pages)
        results.append(pages)
    return results


def split_every_n_pages(total_pages: int, n: int) -> list[list[int]]:
    """Split into chunks of N pages each."""
    if n < 1:
        raise ValueError("Pages per split must be at least 1.")
    ranges = []
    for start in range(0, total_pages, n):
        end = min(start + n, total_pages)
        ranges.append(list(range(start, end)))
    return ranges


def split_into_n_equal_parts(total_pages: int, n: int) -> list[list[int]]:
    """Split into N roughly equal parts."""
    if n < 1:
        raise ValueError("Number of parts must be at least 1.")
    if n > total_pages:
        n = total_pages
    base_size = total_pages // n
    remainder = total_pages % n
    ranges = []
    start = 0
    for i in range(n):
        size = base_size + (1 if i < remainder else 0)
        ranges.append(list(range(start, start + size)))
        start += size
    return ranges


def create_zip_from_pdfs(pdf_list: list[tuple[str, bytes]], base_name: str) -> bytes:
    """Package multiple PDFs into a ZIP file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, (label, pdf_bytes) in enumerate(pdf_list):
            filename = f"{base_name}_{label}.pdf"
            zf.writestr(filename, pdf_bytes)
    return buf.getvalue()


def merge_pdfs(pdf_list: list[tuple[str, bytes]]) -> bytes:
    """Merge multiple PDFs into a single PDF."""
    writer = PdfWriter()
    for label, pdf_bytes in pdf_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def get_page_size_info(page) -> str:
    """Get page dimensions in a readable format."""
    try:
        box = page.mediabox
        w = float(box.width)
        h = float(box.height)
        # Convert points to inches (72 points per inch)
        w_in = w / 72
        h_in = h / 72
        # Convert to mm
        w_mm = w_in * 25.4
        h_mm = h_in * 25.4
        # Detect standard sizes
        size_name = detect_paper_size(w_mm, h_mm)
        return f"{w_mm:.0f}×{h_mm:.0f} mm ({size_name})" if size_name else f"{w_mm:.0f}×{h_mm:.0f} mm"
    except Exception:
        return "Unknown"


def detect_paper_size(w_mm: float, h_mm: float) -> str:
    """Detect standard paper size from dimensions in mm."""
    # Normalize: smaller dimension first
    dims = (min(w_mm, h_mm), max(w_mm, h_mm))
    standards = {
        "A4": (210, 297), "A3": (297, 420), "A5": (148, 210),
        "Letter": (216, 279), "Legal": (216, 356), "Tabloid": (279, 432),
        "A0": (841, 1189), "A1": (594, 841), "A2": (420, 594),
        "A6": (105, 148), "B5": (176, 250), "B4": (250, 353),
    }
    for name, (sw, sh) in standards.items():
        if abs(dims[0] - sw) < 5 and abs(dims[1] - sh) < 5:
            return name
    return ""


def remove_pages(reader: PdfReader, pages_to_remove: list[int]) -> bytes:
    """Remove specified pages (0-indexed) and return remaining PDF."""
    total = len(reader.pages)
    keep = [i for i in range(total) if i not in pages_to_remove]
    if not keep:
        raise ValueError("Cannot remove all pages — at least one page must remain.")
    return build_pdf_from_pages(reader, keep)


# ─── Main UI ────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_split_upload")

if uploaded:
    raw_bytes = uploaded.getvalue()
    original_size = len(raw_bytes)

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        total_pages = len(reader.pages)
        pdf_version = reader.pdf_header if hasattr(reader, 'pdf_header') else "Unknown"
    except Exception as e:
        st.error(f"❌ Could not read PDF: {e}")
        st.stop()

    if total_pages == 0:
        st.error("❌ This PDF has no pages.")
        st.stop()

    # ── Document Info ──
    st.markdown("---")
    st.subheader("📊 Document Info")

    # Gather metadata
    meta_title = ""
    meta_author = ""
    if reader.metadata:
        meta_title = reader.metadata.get("/Title", "") or ""
        meta_author = reader.metadata.get("/Author", "") or ""

    page_size_sample = get_page_size_info(reader.pages[0]) if total_pages > 0 else "N/A"

    info_cols = st.columns(5)
    info_items = [
        ("File Size", format_size(original_size)),
        ("Total Pages", str(total_pages)),
        ("Page Size", page_size_sample),
        ("PDF Version", str(pdf_version)),
        ("Title", meta_title[:30] + "…" if len(meta_title) > 30 else (meta_title or "—")),
    ]
    for col, (label, value) in zip(info_cols, info_items):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Split Mode Selection ──
    st.markdown("---")
    st.subheader("⚙️ Choose Operation")

    tab_extract, tab_range, tab_split_every, tab_split_equal, tab_remove, tab_every_page = st.tabs([
        "📄 Extract Pages",
        "📑 Split by Ranges",
        "📐 Split Every N Pages",
        "🔢 Split into N Parts",
        "🗑️ Remove Pages",
        "📃 Split Every Page",
    ])

    # Shared output mode selector
    output_results = None  # Will hold list of (label, bytes) tuples

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 1: Extract Pages
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_extract:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">📄 Extract Specific Pages</div>'
            '<div class="mode-desc">Pull out specific pages or ranges into a new PDF. '
            'Supports single pages, ranges, open-ended ranges, odd/even, and more.</div>'
            '</div>', unsafe_allow_html=True
        )

        st.markdown("**Syntax guide:**")
        syntax_cols = st.columns(4)
        with syntax_cols[0]:
            st.code("5", language=None)
            st.caption("Single page")
        with syntax_cols[1]:
            st.code("3-7", language=None)
            st.caption("Page range")
        with syntax_cols[2]:
            st.code("1, 3, 5-8, 10", language=None)
            st.caption("Mixed selection")
        with syntax_cols[3]:
            st.code("odd / even / last5", language=None)
            st.caption("Special keywords")

        extract_spec = st.text_input(
            "Enter pages to extract",
            value="",
            placeholder=f"e.g. 1-3, 5, 8-{total_pages}" if total_pages > 3 else "e.g. 1",
            key="extract_spec",
            help=(
                "Formats: single page (5), range (3-7), open range (5- or -5), "
                "comma-separated (1,3,5-8), keywords (odd, even, all, last3)"
            )
        )

        # Quick-select buttons
        st.markdown("**Quick select:**")
        qcols = st.columns(6)
        if qcols[0].button("All", key="q_all"):
            st.session_state["extract_spec"] = "all"
            st.rerun()
        if qcols[1].button("Odd pages", key="q_odd"):
            st.session_state["extract_spec"] = "odd"
            st.rerun()
        if qcols[2].button("Even pages", key="q_even"):
            st.session_state["extract_spec"] = "even"
            st.rerun()
        if qcols[3].button("First page", key="q_first"):
            st.session_state["extract_spec"] = "1"
            st.rerun()
        if qcols[4].button("Last page", key="q_last"):
            st.session_state["extract_spec"] = str(total_pages)
            st.rerun()
        if qcols[5].button("First half", key="q_half"):
            mid = math.ceil(total_pages / 2)
            st.session_state["extract_spec"] = f"1-{mid}"
            st.rerun()

        if extract_spec:
            try:
                parsed = parse_page_spec(extract_spec, total_pages)
                display_pages = [p + 1 for p in parsed]
                chips = " ".join(f'<span class="page-chip">{p}</span>' for p in display_pages)
                st.markdown(f"**Selected {len(parsed)} page(s):** {chips}", unsafe_allow_html=True)
            except ValueError as e:
                st.error(f"⚠️ {e}")

        dedup_pages = st.checkbox(
            "Remove duplicate page selections",
            value=True,
            key="dedup_extract",
            help="If the same page appears multiple times in your selection, keep only the first occurrence."
        )

        if st.button("✂️ Extract Pages", type="primary", key="btn_extract"):
            if not extract_spec:
                st.warning("Please enter a page specification.")
            else:
                try:
                    parsed = parse_page_spec(extract_spec, total_pages)
                    if dedup_pages:
                        seen = set()
                        deduped = []
                        for p in parsed:
                            if p not in seen:
                                seen.add(p)
                                deduped.append(p)
                        parsed = deduped
                    with st.spinner(f"Extracting {len(parsed)} page(s)…"):
                        pdf_bytes = build_pdf_from_pages(reader, parsed)
                    output_results = [("extracted", pdf_bytes)]
                    st.session_state["split_results"] = output_results
                    st.session_state["split_mode"] = "extract"
                except ValueError as e:
                    st.error(f"❌ {e}")

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 2: Split by Ranges
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_range:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">📑 Split by Custom Ranges</div>'
            '<div class="mode-desc">Define multiple ranges separated by semicolons. Each range '
            'becomes a separate PDF. Supports all page specification formats.</div>'
            '</div>', unsafe_allow_html=True
        )

        st.markdown("**Syntax:** Separate each output document with a semicolon `;`")
        st.code("1-3; 4-6; 7-10", language=None)
        st.caption("This creates 3 PDFs: pages 1–3, 4–6, and 7–10")

        range_spec = st.text_area(
            "Enter ranges (semicolon-separated)",
            value="",
            placeholder=f"e.g. 1-{total_pages // 2}; {total_pages // 2 + 1}-{total_pages}",
            key="range_spec",
            height=80,
            help="Each semicolon-separated group becomes a separate PDF. Within each group, use commas for individual pages or dashes for ranges."
        )

        if range_spec:
            try:
                parsed_ranges = parse_split_ranges(range_spec, total_pages)
                st.markdown(f"**Will create {len(parsed_ranges)} PDF(s):**")
                for i, pages in enumerate(parsed_ranges):
                    display = [p + 1 for p in pages]
                    label = format_range_label(pages)
                    st.markdown(
                        f'<span class="range-badge">PDF {i + 1}: {label.replace("pages_", "Pages ").replace("_", ", ")}'
                        f' ({len(pages)} page{"s" if len(pages) != 1 else ""})</span>',
                        unsafe_allow_html=True
                    )
            except ValueError as e:
                st.error(f"⚠️ {e}")

        if st.button("✂️ Split by Ranges", type="primary", key="btn_range"):
            if not range_spec:
                st.warning("Please enter at least one range.")
            else:
                try:
                    parsed_ranges = parse_split_ranges(range_spec, total_pages)
                    with st.spinner(f"Splitting into {len(parsed_ranges)} PDF(s)…"):
                        results = split_into_ranges(reader, parsed_ranges)
                    st.session_state["split_results"] = results
                    st.session_state["split_mode"] = "ranges"
                except ValueError as e:
                    st.error(f"❌ {e}")

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 3: Split Every N Pages
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_split_every:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">📐 Split Every N Pages</div>'
            '<div class="mode-desc">Automatically split the PDF into chunks of a fixed number of pages. '
            'The last chunk may have fewer pages.</div>'
            '</div>', unsafe_allow_html=True
        )

        n_pages = st.number_input(
            "Pages per split",
            min_value=1,
            max_value=total_pages,
            value=min(5, total_pages),
            step=1,
            key="split_every_n",
        )

        num_splits = math.ceil(total_pages / n_pages)
        st.markdown(f"Will create **{num_splits}** PDF(s)")

        # Preview
        preview_ranges = split_every_n_pages(total_pages, n_pages)
        preview_text = " | ".join(
            f"PDF {i + 1}: pages {pages[0] + 1}–{pages[-1] + 1}"
            for i, pages in enumerate(preview_ranges)
        )
        st.caption(f"Preview: {preview_text}")

        if st.button("✂️ Split Every N Pages", type="primary", key="btn_split_every"):
            with st.spinner(f"Splitting into {num_splits} PDF(s)…"):
                ranges = split_every_n_pages(total_pages, n_pages)
                results = split_into_ranges(reader, ranges)
            st.session_state["split_results"] = results
            st.session_state["split_mode"] = "every_n"

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 4: Split into N Equal Parts
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_split_equal:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">🔢 Split into N Equal Parts</div>'
            '<div class="mode-desc">Divide the entire document into a specific number of '
            'roughly equal parts.</div>'
            '</div>', unsafe_allow_html=True
        )

        n_parts = st.number_input(
            "Number of parts",
            min_value=1,
            max_value=total_pages,
            value=min(2, total_pages),
            step=1,
            key="split_n_parts",
        )

        preview_equal = split_into_n_equal_parts(total_pages, n_parts)
        pages_per = [len(r) for r in preview_equal]
        st.markdown(f"Will create **{len(preview_equal)}** PDF(s) with **{pages_per[0]}** page(s) each"
                     + (f" (last part: {pages_per[-1]})" if pages_per[-1] != pages_per[0] else ""))

        preview_text = " | ".join(
            f"Part {i + 1}: pages {pages[0] + 1}–{pages[-1] + 1}"
            for i, pages in enumerate(preview_equal)
        )
        st.caption(f"Preview: {preview_text}")

        if st.button("✂️ Split into Parts", type="primary", key="btn_split_equal"):
            with st.spinner(f"Splitting into {n_parts} part(s)…"):
                ranges = split_into_n_equal_parts(total_pages, n_parts)
                results = split_into_ranges(reader, ranges)
            st.session_state["split_results"] = results
            st.session_state["split_mode"] = "equal_parts"

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 5: Remove Pages
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_remove:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">🗑️ Remove Pages</div>'
            '<div class="mode-desc">Remove specific pages from the PDF and keep everything else. '
            'Opposite of extraction — specify what to REMOVE.</div>'
            '</div>', unsafe_allow_html=True
        )

        remove_spec = st.text_input(
            "Pages to remove",
            value="",
            placeholder=f"e.g. 1, 3, 5-8",
            key="remove_spec",
            help="Specify pages to REMOVE. The remaining pages will be saved."
        )

        if remove_spec:
            try:
                to_remove = parse_page_spec(remove_spec, total_pages)
                remaining = [i for i in range(total_pages) if i not in to_remove]
                st.markdown(
                    f"**Removing {len(to_remove)} page(s)**, keeping {len(remaining)} page(s)"
                )
                if remaining:
                    kept_display = [p + 1 for p in remaining]
                    chips = " ".join(f'<span class="page-chip">{p}</span>' for p in kept_display[:50])
                    if len(kept_display) > 50:
                        chips += f' <span class="page-chip">+{len(kept_display) - 50} more</span>'
                    st.markdown(f"**Pages kept:** {chips}", unsafe_allow_html=True)
            except ValueError as e:
                st.error(f"⚠️ {e}")

        if st.button("🗑️ Remove Pages", type="primary", key="btn_remove"):
            if not remove_spec:
                st.warning("Please specify pages to remove.")
            else:
                try:
                    to_remove = parse_page_spec(remove_spec, total_pages)
                    with st.spinner(f"Removing {len(to_remove)} page(s)…"):
                        pdf_bytes = remove_pages(reader, to_remove)
                    st.session_state["split_results"] = [("pages_removed", pdf_bytes)]
                    st.session_state["split_mode"] = "remove"
                except ValueError as e:
                    st.error(f"❌ {e}")

    # ──────────────────────────────────────────────────────────────────────────────
    # TAB 6: Split Every Page (burst)
    # ──────────────────────────────────────────────────────────────────────────────
    with tab_every_page:
        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">📃 Split Every Page (Burst Mode)</div>'
            '<div class="mode-desc">Explode the entire PDF into individual single-page PDFs. '
            'Each page becomes its own file.</div>'
            '</div>', unsafe_allow_html=True
        )

        st.markdown(f"This will create **{total_pages}** individual PDF files.")

        if total_pages > 200:
            st.warning("⚠️ This document has many pages. The resulting ZIP file may be large.")

        if st.button("✂️ Burst into Individual Pages", type="primary", key="btn_burst"):
            with st.spinner(f"Splitting {total_pages} pages…"):
                ranges = [[i] for i in range(total_pages)]
                results = split_into_ranges(reader, ranges)
            st.session_state["split_results"] = results
            st.session_state["split_mode"] = "burst"

    # ──────────────────────────────────────────────────────────────────────────────
    # Results & Download
    # ──────────────────────────────────────────────────────────────────────────────
    if "split_results" in st.session_state and st.session_state["split_results"]:
        results = st.session_state["split_results"]
        mode = st.session_state.get("split_mode", "")

        st.markdown("---")
        st.subheader("✅ Results")

        # Summary metrics
        total_output_size = sum(len(b) for _, b in results)
        res_cols = st.columns(4)
        res_items = [
            ("Original Size", f'<span class="size-badge size-original">{format_size(original_size)}</span>'),
            ("Output Size", f'<span class="size-badge size-result">{format_size(total_output_size)}</span>'),
            ("PDFs Created", str(len(results))),
            ("Total Pages", str(sum(
                len(PdfReader(io.BytesIO(b)).pages) for _, b in results
            ))),
        ]
        for col, (label, value) in zip(res_cols, res_items):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        # Size comparison
        if total_output_size > original_size:
            overhead = ((total_output_size - original_size) / original_size) * 100
            if overhead > 1:
                st.info(
                    f"ℹ️ Combined output is {overhead:.1f}% larger than the original. "
                    "This is normal when splitting — shared resources (fonts, images) "
                    "are duplicated across output files."
                )

        # ── Output Format Selection ──
        st.markdown("---")
        if len(results) == 1:
            # Single file — just download directly
            label, pdf_bytes = results[0]
            base_name = uploaded.name.rsplit(".", 1)[0]
            dl_name = f"{base_name}_{label}.pdf"

            st.download_button(
                label=f"⬇️ Download PDF ({format_size(len(pdf_bytes))})",
                data=pdf_bytes,
                file_name=dl_name,
                mime="application/pdf",
                type="primary",
                key="dl_single",
            )
        else:
            # Multiple files — let user choose format
            st.subheader("📦 Download Options")

            dl_format = st.radio(
                "How would you like to download?",
                [
                    "📁 Separate PDFs in a ZIP file",
                    "📄 Merge all into one PDF",
                    "⬇️ Download individually",
                ],
                index=0,
                key="dl_format",
                horizontal=True,
            )

            base_name = uploaded.name.rsplit(".", 1)[0]

            if dl_format == "📁 Separate PDFs in a ZIP file":
                zip_bytes = create_zip_from_pdfs(results, base_name)
                st.download_button(
                    label=f"⬇️ Download ZIP ({format_size(len(zip_bytes))})",
                    data=zip_bytes,
                    file_name=f"{base_name}_split.zip",
                    mime="application/zip",
                    type="primary",
                    key="dl_zip",
                )
            elif dl_format == "📄 Merge all into one PDF":
                merged_bytes = merge_pdfs(results)
                st.download_button(
                    label=f"⬇️ Download Merged PDF ({format_size(len(merged_bytes))})",
                    data=merged_bytes,
                    file_name=f"{base_name}_merged.pdf",
                    mime="application/pdf",
                    type="primary",
                    key="dl_merged",
                )
            else:
                # Individual downloads
                st.markdown("Download each PDF individually:")
                dl_cols_per_row = 3
                for row_start in range(0, len(results), dl_cols_per_row):
                    row_items = results[row_start:row_start + dl_cols_per_row]
                    cols = st.columns(dl_cols_per_row)
                    for col, (label, pdf_bytes) in zip(cols, row_items):
                        dl_name = f"{base_name}_{label}.pdf"
                        page_count = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
                        col.download_button(
                            label=f"⬇️ {label} ({page_count}p, {format_size(len(pdf_bytes))})",
                            data=pdf_bytes,
                            file_name=dl_name,
                            mime="application/pdf",
                            key=f"dl_individual_{label}",
                        )

        # ── Detailed Breakdown ──
        if len(results) > 1:
            with st.expander(f"📊 Detailed breakdown ({len(results)} files)", expanded=False):
                for i, (label, pdf_bytes) in enumerate(results):
                    sub_reader = PdfReader(io.BytesIO(pdf_bytes))
                    page_count = len(sub_reader.pages)
                    size = len(pdf_bytes)
                    st.markdown(
                        f"**{i + 1}.** `{label}.pdf` — "
                        f"{page_count} page{'s' if page_count != 1 else ''}, "
                        f"{format_size(size)}"
                    )

        # Reset button
        if st.button("🔄 Start Over", key="btn_reset"):
            if "split_results" in st.session_state:
                del st.session_state["split_results"]
            if "split_mode" in st.session_state:
                del st.session_state["split_mode"]
            st.rerun()

else:
    st.info("👆 Upload a PDF to get started.")

    # Feature showcase
    st.markdown("---")
    st.markdown("### ✨ Features")
    feat_cols = st.columns(3)
    with feat_cols[0]:
        st.markdown(
            "**📄 Extract Pages**\n\n"
            "Pull specific pages using flexible syntax: "
            "`1-3`, `5`, `odd`, `even`, `last5`, or any combination."
        )
    with feat_cols[1]:
        st.markdown(
            "**📑 Split by Ranges**\n\n"
            "Define custom ranges with `;` separators. "
            "Each range becomes a separate PDF file."
        )
    with feat_cols[2]:
        st.markdown(
            "**📐 Auto-Split**\n\n"
            "Split every N pages, into N equal parts, "
            "or burst into individual page files."
        )

    feat_cols2 = st.columns(3)
    with feat_cols2[0]:
        st.markdown(
            "**🗑️ Remove Pages**\n\n"
            "Specify pages to remove — everything else stays. "
            "Inverse of extraction."
        )
    with feat_cols2[1]:
        st.markdown(
            "**🔒 Lossless Quality**\n\n"
            "No re-encoding, no quality loss. Pages are cloned "
            "exactly as they are in the original."
        )
    with feat_cols2[2]:
        st.markdown(
            "**📦 Flexible Output**\n\n"
            "Download as separate PDFs in ZIP, merge into one file, "
            "or download individually."
        )
