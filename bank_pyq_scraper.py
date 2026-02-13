#!/usr/bin/env python3
"""
Bank PYQ Scraper – Download & organize previous-year question papers
for SBI PO, IBPS PO, and IBPS RRB PO from BankersAdda PDFs + web search.

Outputs:
    bank_pyqs/<Exam>/<Phase>/<Year>/Shift_<N>/<Subject>.txt
"""

import re, os, time, json, textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf

# ─── configuration ──────────────────────────────────────────────────
OUT_DIR = Path("bank_pyqs")
YEARS = range(2020, 2026)           # 2020–2025
SKIP_HINDI = True
SKIP_SOLUTIONS_ONLY = True

# BankersAdda pages that list PDF links per exam
BANKERSADDA_PAGES = {
    "SBI_PO": [
        "https://www.bankersadda.com/sbi-po-previous-years-papers-download-pdfs/",
    ],
    "IBPS_PO": [
        "https://www.bankersadda.com/ibps-po-previous-year-question-paper/",
    ],
    "IBPS_RRB_PO": [
        "https://www.bankersadda.com/ibps-rrb-previous-year-question-papers/",
        "https://www.bankersadda.com/ibps-rrb-po-previous-year-question-paper/",
    ],
}

# Subject keywords used to classify question content
SUBJECT_PATTERNS = {
    "English": [
        r"english\s+language", r"reading\s+comprehension", r"cloze\s+test",
        r"fill\s+in\s+the\s+blank", r"sentence\s+rearrangement",
        r"error\s+detection", r"para\s*jumbl",
    ],
    "Quant": [
        r"quantitative\s+aptitude", r"data\s+(interpretation|analysis)",
        r"number\s+series", r"simplification", r"profit\s+and\s+loss",
        r"average", r"percentage", r"ratio\s+and\s+proportion",
        r"quadratic\s+equation", r"data\s+sufficiency",
    ],
    "Reasoning": [
        r"reasoning\s+ability", r"logical\s+reasoning", r"syllogism",
        r"blood\s+relation", r"seating\s+arrangement", r"coding.?decoding",
        r"puzzle", r"direction\s+sense", r"inequality",
        r"input.?output", r"machine\s+input",
    ],
    "GS_and_CA": [
        r"general\s+(awareness|knowledge)", r"current\s+affairs",
        r"banking\s+awareness", r"financial\s+awareness",
        r"static\s+gk", r"who\s+is\s+the",
    ],
    "Computer": [
        r"computer\s+(aptitude|knowledge|awareness)",
        r"ms\s+office", r"operating\s+system", r"database",
    ],
}

# Prelims subject ordering (approximate Q number ranges per exam)
PRELIMS_SPLIT = {
    "SBI_PO":      [("English", 30), ("Quant", 35), ("Reasoning", 35)],
    "IBPS_PO":     [("English", 30), ("Quant", 35), ("Reasoning", 35)],
    "IBPS_RRB_PO": [("Reasoning", 40), ("Quant", 40)],
}

MAINS_SPLIT = {
    "SBI_PO":      [("Reasoning", 35), ("Quant", 35), ("English", 35), ("GS_and_CA", 40)],
    "IBPS_PO":     [("Reasoning", 35), ("Quant", 35), ("English", 35), ("GS_and_CA", 40)],
    "IBPS_RRB_PO": [("Reasoning", 40), ("Quant", 40), ("English", 40), ("GS_and_CA", 40), ("Computer", 40)],
}


# ─── HTTP session ──────────────────────────────────────────────────
_sess = requests.Session()
_sess.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


# ─── data class for parsed PDF info ───────────────────────────────
@dataclass
class PaperInfo:
    url: str
    exam: str                         # SBI_PO, IBPS_PO, IBPS_RRB_PO
    phase: str = "Unknown"            # Prelims, Mains
    year: Optional[int] = None
    shift: int = 1
    subject: Optional[str] = None     # English, Quant, Reasoning, GS_and_CA, Computer
    is_hindi: bool = False
    is_solution_only: bool = False
    link_text: str = ""
    filename: str = ""


# ─── filename parser ──────────────────────────────────────────────
def _parse_pdf_info(url: str, link_text: str, exam: str) -> PaperInfo:
    """Parse a PDF URL/filename into structured PaperInfo."""
    fname = url.split("/")[-1].lower()
    info = PaperInfo(url=url, exam=exam, filename=fname, link_text=link_text)

    # Phase
    if any(k in fname for k in ["pre-", "pre_", "prelim"]):
        info.phase = "Prelims"
    elif "mains" in fname or "mains" in link_text.lower():
        info.phase = "Mains"
    else:
        info.phase = "Prelims"  # default

    # Year – extract 4-digit year
    year_matches = re.findall(r'20(1[6-9]|2[0-9])', fname)
    if year_matches:
        years = sorted(set(int("20" + m) for m in year_matches))
        info.year = years[0]
        # "jan-2021" is 2020 exam cycle
        if info.year == 2021 and ("jan-2021" in fname or "jan_2021" in fname):
            info.year = 2020

    # Shift
    shift_m = re.search(r'(\d+)(?:st|nd|rd|th)[\-_\s]*shift', fname)
    if shift_m:
        info.shift = int(shift_m.group(1))
    else:
        mock_m = re.search(r'mock[\-_\s]*(\d+)', fname)
        if mock_m:
            info.shift = int(mock_m.group(1))

    # Hindi
    if "hindi" in fname:
        info.is_hindi = True

    # Solution only
    fname_low = fname.lower()
    lt_low = link_text.lower()
    if ("solution" in fname_low and "question" not in fname_low) or \
       ("solution" in lt_low and "question" not in lt_low and "download" not in lt_low):
        info.is_solution_only = True

    # Subject from filename
    subj_map = {
        "english": "English",
        "quant": "Quant",
        "quantitative": "Quant",
        "data-analysis": "Quant",
        "data_analysis": "Quant",
        "data-interpretation": "Quant",
        "reasoning": "Reasoning",
        "general-awareness": "GS_and_CA",
        "general_awareness": "GS_and_CA",
        "ga_mcq": "GS_and_CA",
        "current-affairs": "GS_and_CA",
        "computer": "Computer",
    }
    for key, subj in subj_map.items():
        if key in fname_low:
            info.subject = subj
            break

    # Filter out Clerk papers for IBPS RRB
    if exam == "IBPS_RRB_PO" and "clerk" in fname_low:
        info.exam = "SKIP_CLERK"

    return info


# ─── collect all PDF links from BankersAdda ──────────────────────
def collect_bankersadda_pdfs() -> list[PaperInfo]:
    """Scrape BankersAdda pages and return list of PaperInfo objects."""
    all_papers = []
    seen_urls = set()

    for exam, pages in BANKERSADDA_PAGES.items():
        for page_url in pages:
            print(f"  Fetching {exam}: {page_url[:60]}...")
            try:
                r = _sess.get(page_url, timeout=25)
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code}, skipping")
                    continue
            except Exception as e:
                print(f"    Error: {e}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" not in href.lower():
                    continue
                if href in seen_urls:
                    continue
                seen_urls.add(href)

                link_text = a.get_text(" ", strip=True)
                info = _parse_pdf_info(href, link_text, exam)
                all_papers.append(info)

            time.sleep(1)

    return all_papers


# ─── search for additional PDFs via ddgs ─────────────────────────
def search_additional_pdfs(existing: list[PaperInfo]) -> list[PaperInfo]:
    """Use ddgs to find PDFs missing from BankersAdda catalog."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("  ddgs not installed, skipping search")
        return []

    existing_keys = set()
    for p in existing:
        if p.year and p.exam != "SKIP_CLERK":
            existing_keys.add((p.exam, p.phase, p.year))

    needed = []
    for exam in ["SBI_PO", "IBPS_PO", "IBPS_RRB_PO"]:
        for phase in ["Prelims", "Mains"]:
            for year in YEARS:
                if (exam, phase, year) not in existing_keys:
                    needed.append((exam, phase, year))

    if not needed:
        print("  No gaps found, skipping search")
        return []

    ddg = DDGS()
    extra = []
    for exam, phase, year in needed[:20]:
        exam_name = exam.replace("_", " ")
        q = f"{exam_name} {phase} {year} memory based question paper PDF site:bankersadda.com OR site:prepp.in"
        print(f"  Searching: {exam_name} {phase} {year}...")
        try:
            results = list(ddg.text(q, max_results=5))
            for r in results:
                href = r["href"]
                if ".pdf" in href.lower():
                    info = _parse_pdf_info(href, r.get("title", ""), exam)
                    info.phase = phase
                    if not info.year:
                        info.year = year
                    extra.append(info)
            time.sleep(2)
        except Exception as e:
            print(f"    Search error: {e}")

    return extra


# ─── search for prepp.in PDFs ────────────────────────────────────
def collect_prepp_pdfs() -> list[PaperInfo]:
    """Scrape prepp.in for additional question paper PDFs."""
    all_papers = []
    exam_map = {
        "SBI_PO": "sbi-po-exam",
        "IBPS_PO": "ibps-po-exam",
        "IBPS_RRB_PO": "ibps-rrb-po-exam",
    }

    for exam, slug in exam_map.items():
        for year in YEARS:
            url = f"https://prepp.in/{slug}/question-paper-{year}"
            print(f"  Checking prepp.in: {exam} {year}...")
            try:
                r = _sess.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if ".pdf" not in href.lower():
                        continue
                    link_text = a.get_text(" ", strip=True)
                    info = _parse_pdf_info(href, link_text, exam)
                    if not info.year:
                        info.year = year
                    all_papers.append(info)
                time.sleep(1)
            except Exception:
                pass

    return all_papers


# ─── PDF download & text extraction ──────────────────────────────
def download_and_extract(url: str) -> Optional[str]:
    """Download a PDF and extract text via pymupdf."""
    try:
        r = _sess.get(url, timeout=45)
        if r.status_code != 200 or len(r.content) < 500:
            return None
        doc = fitz.open(stream=r.content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip() if len(text.strip()) > 100 else None
    except Exception as e:
        print(f"    PDF error: {e}")
        return None


# ─── subject detection from text ─────────────────────────────────
def detect_subject(text: str) -> Optional[str]:
    """Try to determine the primary subject of a paper from its content."""
    text_lower = text[:5000].lower()
    scores = {}
    for subj, patterns in SUBJECT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[subj] = score
    if scores:
        return max(scores, key=scores.get)
    return None


# ─── split composite paper by subject ────────────────────────────
def split_by_subject(text: str, exam: str, phase: str) -> dict[str, str]:
    """Split a composite paper into per-subject text."""
    # Strategy 1: Look for explicit section headers
    section_markers = [
        (r"(?:section|part)\s*[-:]\s*(english|quantitative|reasoning|general|computer)",
         {"english": "English", "quantitative": "Quant", "reasoning": "Reasoning",
          "general": "GS_and_CA", "computer": "Computer"}),
        (r"(english\s+language|quantitative\s+aptitude|reasoning\s+ability|"
         r"general\s+(?:awareness|knowledge)|computer\s+(?:aptitude|knowledge))",
         {"english language": "English", "quantitative aptitude": "Quant",
          "reasoning ability": "Reasoning", "general awareness": "GS_and_CA",
          "general knowledge": "GS_and_CA", "computer aptitude": "Computer",
          "computer knowledge": "Computer"}),
    ]

    for pattern, mapping in section_markers:
        matches = [(m.start(), m.group(1).lower()) for m in re.finditer(pattern, text, re.I)]
        if len(matches) >= 2:
            result = {}
            for i, (pos, key) in enumerate(matches):
                subj = None
                for k, v in mapping.items():
                    if k in key:
                        subj = v
                        break
                if not subj:
                    continue
                end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
                section_text = text[pos:end].strip()
                if len(section_text) > 50:
                    result[subj] = section_text
            if len(result) >= 2:
                return result

    # Strategy 2: Split by question numbers using known exam structure
    questions = list(re.finditer(r'\bQ\.?\s*(\d+)[\.\)\s]', text))
    if not questions:
        questions = list(re.finditer(r'\bQuestion\s*(\d+)', text))

    split_config = PRELIMS_SPLIT.get(exam) if phase == "Prelims" else MAINS_SPLIT.get(exam)
    if not split_config or not questions:
        return {}

    q_positions = {}
    for m in questions:
        qnum = int(m.group(1))
        if qnum not in q_positions:
            q_positions[qnum] = m.start()

    if len(q_positions) < 10:
        return {}

    result = {}
    cum = 0
    for i, (subj, count) in enumerate(split_config):
        start_q = cum + 1
        end_q = cum + count
        cum = end_q

        start_pos = None
        end_pos = None
        for qn in range(start_q, end_q + 5):
            if qn in q_positions:
                if start_pos is None:
                    start_pos = q_positions[qn]
                break

        if i + 1 < len(split_config):
            next_start_q = cum + 1
            for qn in range(next_start_q, next_start_q + 5):
                if qn in q_positions:
                    end_pos = q_positions[qn]
                    break

        if start_pos is not None:
            section = text[start_pos:end_pos].strip() if end_pos else text[start_pos:].strip()
            # Remove answer keys at the end
            ans_start = re.search(r'\n\s*S\.?\s*1\.?\s*Ans', section)
            if ans_start:
                section = section[:ans_start.start()]
            if len(section) > 100:
                result[subj] = section

    return result


# ─── clean extracted text ────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'(?m)^.*www\.bankersadda\.com.*$', '', text)
    text = re.sub(r'(?m)^.*adda247\.com.*$', '', text)
    text = re.sub(r'(?m)^.*Download from.*$', '', text)
    text = re.sub(r'(?m)^.*Join Telegram.*$', '', text)
    return text.strip()


# ─── save to disk ────────────────────────────────────────────────
def save_paper(text: str, exam: str, phase: str, year: int,
               shift: int, subject: str, source_url: str) -> Path:
    """Save paper text to organized directory."""
    out_dir = OUT_DIR / exam / phase / str(year) / f"Shift_{shift}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{subject}.txt"
    out_path = out_dir / fname

    # Don't overwrite if existing file is larger
    if out_path.exists():
        existing_size = out_path.stat().st_size
        if existing_size >= len(text):
            return out_path

    header = (
        f"{'='*70}\n"
        f"Exam: {exam.replace('_', ' ')}\n"
        f"Phase: {phase}\n"
        f"Year: {year}\n"
        f"Shift: {shift}\n"
        f"Subject: {subject}\n"
        f"Source: {source_url}\n"
        f"{'='*70}\n\n"
    )

    out_path.write_text(header + text, encoding="utf-8")
    return out_path


# ─── also try scraping HTML question pages ───────────────────────
def scrape_html_questions(exam: str, phase: str, year: int) -> dict[str, str]:
    """Search for HTML question pages and extract per-subject text."""
    try:
        from ddgs import DDGS
    except ImportError:
        return {}

    exam_name = exam.replace("_", " ")
    subjects_found = {}

    subject_queries = {
        "English": f"{exam_name} {phase} {year} English Language questions asked memory based",
        "Quant": f"{exam_name} {phase} {year} Quantitative Aptitude questions asked memory based",
        "Reasoning": f"{exam_name} {phase} {year} Reasoning Ability questions asked memory based",
    }
    if phase == "Mains":
        subject_queries["GS_and_CA"] = f"{exam_name} {phase} {year} General Awareness questions asked"
        if exam == "IBPS_RRB_PO":
            subject_queries["Computer"] = f"{exam_name} {phase} {year} Computer Aptitude questions"

    ddg = DDGS()
    for subj, query in subject_queries.items():
        if subj in subjects_found:
            continue
        try:
            results = list(ddg.text(query, max_results=3))
            for r in results:
                href = r["href"]
                if ".pdf" in href.lower():
                    continue
                text = _scrape_page_for_questions(href)
                if text and len(text) > 200:
                    subjects_found[subj] = text
                    break
            time.sleep(1.5)
        except Exception:
            pass

    return subjects_found


def _scrape_page_for_questions(url: str) -> Optional[str]:
    """Scrape a web page for question content."""
    try:
        r = _sess.get(url, timeout=20)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        content = None
        for sel in ["article", ".entry-content", ".post-content", ".blog-content",
                     "main", "#content", ".main-content"]:
            content = soup.select_one(sel)
            if content:
                break
        if not content:
            content = soup.body
        if not content:
            return None

        text = content.get_text("\n", strip=True)
        q_count = len(re.findall(r'Q\.?\s*\d+|Question\s*\d+|\d+[\.\)]\s*(?:What|Which|How|Find|If|The|A\s)', text))
        if q_count < 3:
            return None
        return clean_text(text)
    except Exception:
        return None


# ─── main pipeline ───────────────────────────────────────────────
def run():
    """Main execution pipeline."""
    print("=" * 60)
    print("  Bank PYQ Scraper – PDF + Web Hybrid")
    print("=" * 60)

    # Step 1: Collect all PDF links from BankersAdda
    print("\n[1] Collecting PDF links from BankersAdda...")
    papers = collect_bankersadda_pdfs()
    print(f"    Found {len(papers)} total PDF links")

    # Step 1b: Collect from prepp.in
    print("\n[1b] Collecting PDF links from Prepp.in...")
    prepp = collect_prepp_pdfs()
    papers.extend(prepp)
    print(f"    Found {len(prepp)} PDFs from Prepp.in")

    # Step 2: Search for additional PDFs
    print("\n[2] Searching for additional PDFs via web search...")
    extra = search_additional_pdfs(papers)
    papers.extend(extra)
    print(f"    Added {len(extra)} extra PDFs, total: {len(papers)}")

    # Step 3: Filter
    filtered = []
    for p in papers:
        if p.exam == "SKIP_CLERK":
            continue
        if SKIP_HINDI and p.is_hindi:
            continue
        if SKIP_SOLUTIONS_ONLY and p.is_solution_only:
            continue
        if p.year and p.year not in YEARS:
            continue
        if not p.year:
            continue
        filtered.append(p)

    # Deduplicate by URL
    seen = set()
    unique = []
    for p in filtered:
        if p.url not in seen:
            seen.add(p.url)
            unique.append(p)

    print(f"\n[3] After filtering: {len(unique)} PDFs to download")

    # Show summary
    exam_counts = {}
    for p in unique:
        key = f"{p.exam}/{p.phase}/{p.year}"
        exam_counts[key] = exam_counts.get(key, 0) + 1
    for k, v in sorted(exam_counts.items()):
        print(f"    {k}: {v} PDFs")

    # Step 4: Download and process each PDF
    print(f"\n[4] Downloading and processing PDFs...")
    saved_count = 0
    saved_files = []
    errors = []

    for i, paper in enumerate(unique, 1):
        label = f"{paper.exam}/{paper.phase}/{paper.year}/Shift_{paper.shift}"
        if paper.subject:
            label += f"/{paper.subject}"
        print(f"\n  [{i}/{len(unique)}] {label}")
        print(f"    URL: ...{paper.filename[:60]}")

        text = download_and_extract(paper.url)
        if not text:
            errors.append(f"  FAIL: {label} – no text extracted")
            print("    ✗ Could not extract text")
            continue

        text = clean_text(text)
        print(f"    ✓ Extracted {len(text)} chars")

        if paper.subject:
            # Subject-specific PDF – save directly
            path = save_paper(text, paper.exam, paper.phase, paper.year,
                              paper.shift, paper.subject, paper.url)
            saved_count += 1
            saved_files.append(str(path))
            print(f"    → Saved: {path}")
        else:
            # Composite paper – try to split by subject
            subjects = split_by_subject(text, paper.exam, paper.phase)
            if subjects:
                for subj, subj_text in subjects.items():
                    path = save_paper(subj_text, paper.exam, paper.phase,
                                      paper.year, paper.shift, subj, paper.url)
                    saved_count += 1
                    saved_files.append(str(path))
                    print(f"    → Saved: {path}")
            else:
                # Could not split – save as Full_Paper
                path = save_paper(text, paper.exam, paper.phase, paper.year,
                                  paper.shift, "Full_Paper", paper.url)
                saved_count += 1
                saved_files.append(str(path))
                print(f"    → Saved (full): {path}")

        time.sleep(0.5)

    # Step 5: Fill gaps with HTML scraping
    print(f"\n[5] Checking for gaps and filling via HTML scraping...")
    gap_count = 0
    for exam in ["SBI_PO", "IBPS_PO", "IBPS_RRB_PO"]:
        for phase in ["Prelims", "Mains"]:
            for year in YEARS:
                base_dir = OUT_DIR / exam / phase / str(year)
                if not base_dir.exists():
                    print(f"  Gap: {exam}/{phase}/{year} – searching web...")
                    subjects = scrape_html_questions(exam, phase, year)
                    for subj, text in subjects.items():
                        path = save_paper(text, exam, phase, year, 1, subj, "web_search")
                        saved_files.append(str(path))
                        gap_count += 1
                        print(f"    → Filled: {path}")
                    time.sleep(2)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total files saved: {saved_count + gap_count}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    {e}")

    print(f"\n  Directory structure:")
    for exam in ["SBI_PO", "IBPS_PO", "IBPS_RRB_PO"]:
        exam_dir = OUT_DIR / exam
        if not exam_dir.exists():
            continue
        for phase in sorted(exam_dir.iterdir()):
            if not phase.is_dir():
                continue
            for year in sorted(phase.iterdir()):
                if not year.is_dir():
                    continue
                for shift in sorted(year.iterdir()):
                    if not shift.is_dir():
                        continue
                    files = list(shift.glob("*.txt"))
                    subjects = [f.stem for f in files]
                    print(f"    {exam}/{phase.name}/{year.name}/{shift.name}: {', '.join(subjects)}")

    return saved_count + gap_count


if __name__ == "__main__":
    total = run()
    print(f"\nDone! {total} files saved to {OUT_DIR}/")
