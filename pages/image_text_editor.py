"""
Advanced Image Text Editor
- Detects text in images using OCR
- Analyzes font style, color, size, and effects
- Replaces text seamlessly with new text in matching style
- Makes edits look completely original
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import io
import os
import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import colorsys

# Page config
st.set_page_config(
    page_title="Image Text Editor",
    page_icon="✏️",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        font-weight: bold; 
    }
    .text-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .text-preview {
        font-family: monospace;
        background: #2d2d3d;
        padding: 10px;
        border-radius: 5px;
        color: #fff;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #17a2b8;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #0c5460;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #856404;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-label { color: #888; font-size: 0.85em; }
    .metric-value { color: #fff; font-size: 1.2em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("✏️ Image Text Editor")
st.caption("Detect, analyze, and replace text in images while preserving the original style")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextRegion:
    """Represents a detected text region in the image."""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    angle: float = 0.0
    font_size: int = 20
    font_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Optional[Tuple[int, int, int]] = None
    font_weight: str = "normal"  # normal, bold
    font_style: str = "normal"  # normal, italic
    has_shadow: bool = False
    shadow_color: Optional[Tuple[int, int, int]] = None
    has_outline: bool = False
    outline_color: Optional[Tuple[int, int, int]] = None
    line_height: float = 1.2
    letter_spacing: float = 0.0
    

@dataclass
class FontMatch:
    """Represents a matched font for replacement."""
    font_path: str
    font_name: str
    similarity_score: float


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    if img.mode == 'RGBA':
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format."""
    if len(img.shape) == 2:
        return Image.fromarray(img)
    if img.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_dominant_color(img: Image.Image, region: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Get the dominant color in a region."""
    cropped = img.crop(region)
    pixels = list(cropped.getdata())
    
    if not pixels:
        return (0, 0, 0)
    
    # Use k-means to find dominant color
    pixels_np = np.array(pixels)
    if len(pixels_np.shape) == 1:
        return tuple(pixels_np[:3])
    
    # Handle RGBA
    if pixels_np.shape[1] == 4:
        pixels_np = pixels_np[:, :3]
    
    # Simple averaging for speed
    avg_color = np.mean(pixels_np, axis=0).astype(int)
    return tuple(avg_color)


def get_text_color(img: Image.Image, region: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Analyze region to determine text color and background color.
    Returns (text_color, background_color)
    """
    x1, y1, x2, y2 = region
    cropped = img.crop(region).convert('RGB')
    
    # Convert to numpy
    pixels = np.array(cropped)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu's thresholding to separate text from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Determine if text is dark or light
    text_mask = binary < 128  # Assuming dark text
    bg_mask = ~text_mask
    
    # Check which is more likely text (text usually covers less area)
    if np.sum(text_mask) > np.sum(bg_mask):
        text_mask, bg_mask = bg_mask, text_mask
    
    # Get colors
    if np.any(text_mask):
        text_pixels = pixels[text_mask]
        text_color = tuple(np.median(text_pixels, axis=0).astype(int))
    else:
        text_color = (0, 0, 0)
    
    if np.any(bg_mask):
        bg_pixels = pixels[bg_mask]
        bg_color = tuple(np.median(bg_pixels, axis=0).astype(int))
    else:
        bg_color = (255, 255, 255)
    
    return text_color, bg_color


def estimate_font_size(bbox: Tuple[int, int, int, int], text: str) -> int:
    """Estimate font size from bounding box and text length."""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    
    # Approximate: font size is roughly the height of the bbox
    # Adjust based on text length
    estimated_size = int(height * 0.85)
    
    return max(8, min(200, estimated_size))


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate color distance between two RGB colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def detect_text_effects(img: Image.Image, region: Tuple[int, int, int, int], 
                       text_color: Tuple[int, int, int]) -> Dict:
    """Detect if text has shadow, outline, or other effects."""
    effects = {
        'has_shadow': False,
        'shadow_color': None,
        'shadow_offset': (0, 0),
        'has_outline': False,
        'outline_color': None,
        'outline_width': 0
    }
    
    x1, y1, x2, y2 = region
    cropped = img.crop(region).convert('RGB')
    cv_img = pil_to_cv(cropped)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find outlines
    edges = cv2.Canny(gray, 50, 150)
    
    # If there are significant edges, might have outline
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio > 0.1:
        effects['has_outline'] = True
        
        # Try to detect outline color by looking at edge pixels
        edge_mask = edges > 0
        if np.any(edge_mask):
            edge_pixels = cv_img[edge_mask]
            if len(edge_pixels) > 0:
                outline_color = tuple(np.median(edge_pixels, axis=0).astype(int))
                # Convert BGR to RGB
                effects['outline_color'] = (outline_color[2], outline_color[1], outline_color[0])
    
    # Shadow detection: look for darker version of text shifted
    # This is simplified - a full implementation would be more complex
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if np.std(blurred) > 30:  # Some variation suggests possible shadow
        # Check for asymmetric darkness (shadow typically on one side)
        h, w = gray.shape
        left_half = np.mean(gray[:, :w//2])
        right_half = np.mean(gray[:, w//2:])
        top_half = np.mean(gray[:h//2, :])
        bottom_half = np.mean(gray[h//2:, :])
        
        if abs(left_half - right_half) > 20 or abs(top_half - bottom_half) > 20:
            effects['has_shadow'] = True
            # Estimate shadow color as darker version of background
            effects['shadow_color'] = tuple(max(0, c - 50) for c in text_color)
    
    return effects


# ══════════════════════════════════════════════════════════════════════════════
# OCR AND TEXT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TextDetector:
    """Detect and extract text from images."""
    
    def __init__(self):
        self.ocr_engine = None
        self.engine_type = None
    
    @st.cache_resource
    def load_easyocr(_self):
        """Load EasyOCR reader."""
        try:
            import easyocr
            return easyocr.Reader(['en'], gpu=False)
        except ImportError:
            return None
    
    @st.cache_resource  
    def load_pytesseract(_self):
        """Check if pytesseract is available."""
        try:
            import pytesseract
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception:
            return None
    
    def detect_text_easyocr(self, img: Image.Image) -> List[TextRegion]:
        """Detect text using EasyOCR."""
        reader = self.load_easyocr()
        if reader is None:
            return []
        
        # Convert to numpy array
        img_np = np.array(img.convert('RGB'))
        
        # Detect text
        results = reader.readtext(img_np)
        
        text_regions = []
        for result in results:
            bbox_points, text, confidence = result
            
            if confidence < 0.3 or not text.strip():
                continue
            
            # Convert polygon to rectangle
            points = np.array(bbox_points)
            x1, y1 = points.min(axis=0).astype(int)
            x2, y2 = points.max(axis=0).astype(int)
            
            # Calculate angle
            angle = 0.0
            if len(points) >= 2:
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
                angle = math.degrees(math.atan2(dy, dx))
            
            # Analyze text properties
            text_color, bg_color = get_text_color(img, (x1, y1, x2, y2))
            font_size = estimate_font_size((x1, y1, x2, y2), text)
            effects = detect_text_effects(img, (x1, y1, x2, y2), text_color)
            
            region = TextRegion(
                text=text,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                angle=angle,
                font_size=font_size,
                font_color=text_color,
                bg_color=bg_color,
                has_shadow=effects['has_shadow'],
                shadow_color=effects['shadow_color'],
                has_outline=effects['has_outline'],
                outline_color=effects['outline_color']
            )
            text_regions.append(region)
        
        return text_regions
    
    def detect_text_pytesseract(self, img: Image.Image) -> List[TextRegion]:
        """Detect text using Pytesseract."""
        pytesseract = self.load_pytesseract()
        if pytesseract is None:
            return []
        
        # Get detailed data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        text_regions = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if conf < 30 or not text:
                continue
            
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            bbox = (x, y, x + w, y + h)
            
            # Analyze text properties
            text_color, bg_color = get_text_color(img, bbox)
            font_size = estimate_font_size(bbox, text)
            effects = detect_text_effects(img, bbox, text_color)
            
            region = TextRegion(
                text=text,
                bbox=bbox,
                confidence=conf / 100,
                font_size=font_size,
                font_color=text_color,
                bg_color=bg_color,
                has_shadow=effects['has_shadow'],
                shadow_color=effects['shadow_color'],
                has_outline=effects['has_outline'],
                outline_color=effects['outline_color']
            )
            text_regions.append(region)
        
        return text_regions
    
    def detect_text_opencv(self, img: Image.Image) -> List[TextRegion]:
        """Detect text using OpenCV EAST or MSER."""
        cv_img = pil_to_cv(img.convert('RGB'))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        mser.setMinArea(100)
        mser.setMaxArea(10000)
        
        regions, _ = mser.detectRegions(gray)
        
        # Group nearby regions into text blocks
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        
        text_regions = []
        processed_areas = []
        
        for hull in hulls:
            x, y, w, h = cv2.boundingRect(hull)
            
            # Filter by aspect ratio (text is usually wider than tall or square)
            if w < 10 or h < 10:
                continue
            
            aspect = w / h
            if aspect < 0.1 or aspect > 15:
                continue
            
            bbox = (x, y, x + w, y + h)
            
            # Check if this overlaps with existing regions
            overlap = False
            for px, py, pw, ph in processed_areas:
                if (x < px + pw and x + w > px and y < py + ph and y + h > py):
                    overlap = True
                    break
            
            if overlap:
                continue
            
            processed_areas.append((x, y, w, h))
            
            # Analyze text properties
            text_color, bg_color = get_text_color(img, bbox)
            font_size = estimate_font_size(bbox, "X" * max(1, w // 10))
            
            region = TextRegion(
                text="[Detected Region]",  # No OCR, just detection
                bbox=bbox,
                confidence=0.5,
                font_size=font_size,
                font_color=text_color,
                bg_color=bg_color
            )
            text_regions.append(region)
        
        return text_regions
    
    def detect(self, img: Image.Image, method: str = "auto") -> List[TextRegion]:
        """Detect text using the specified or best available method."""
        if method == "auto":
            # Try EasyOCR first, then Pytesseract, then OpenCV
            regions = self.detect_text_easyocr(img)
            if regions:
                return regions
            
            regions = self.detect_text_pytesseract(img)
            if regions:
                return regions
            
            return self.detect_text_opencv(img)
        
        elif method == "easyocr":
            return self.detect_text_easyocr(img)
        elif method == "pytesseract":
            return self.detect_text_pytesseract(img)
        elif method == "opencv":
            return self.detect_text_opencv(img)
        
        return []


# ══════════════════════════════════════════════════════════════════════════════
# FONT MATCHING
# ══════════════════════════════════════════════════════════════════════════════

class FontMatcher:
    """Match and find similar fonts."""
    
    # Common fonts that are likely available
    COMMON_FONTS = {
        "sans-serif": [
            "Arial", "Helvetica", "DejaVuSans", "FreeSans", 
            "Liberation Sans", "Roboto", "Open Sans"
        ],
        "serif": [
            "Times New Roman", "Georgia", "DejaVuSerif", 
            "FreeSerif", "Liberation Serif", "Merriweather"
        ],
        "monospace": [
            "Courier New", "Consolas", "DejaVuSansMono", 
            "FreeMono", "Liberation Mono", "Roboto Mono"
        ],
        "display": [
            "Impact", "Arial Black", "Cooper Black", "Bebas"
        ],
        "handwriting": [
            "Comic Sans MS", "Brush Script", "Pacifico"
        ]
    }
    
    # Font paths on different systems
    FONT_PATHS = [
        "/usr/share/fonts/",
        "/usr/local/share/fonts/",
        "C:/Windows/Fonts/",
        "/System/Library/Fonts/",
        "~/.fonts/",
        "~/.local/share/fonts/",
    ]
    
    def __init__(self):
        self.available_fonts = self._scan_fonts()
    
    def _scan_fonts(self) -> Dict[str, str]:
        """Scan system for available fonts."""
        fonts = {}
        
        # Add bundled fonts
        bundled_fonts_dir = Path(__file__).parent / "fonts" if "__file__" in dir() else Path("fonts")
        if bundled_fonts_dir.exists():
            for font_file in bundled_fonts_dir.glob("*.ttf"):
                fonts[font_file.stem.lower()] = str(font_file)
            for font_file in bundled_fonts_dir.glob("*.otf"):
                fonts[font_file.stem.lower()] = str(font_file)
        
        # Scan system fonts
        for font_path in self.FONT_PATHS:
            path = Path(os.path.expanduser(font_path))
            if path.exists():
                for ext in ["*.ttf", "*.TTF", "*.otf", "*.OTF"]:
                    for font_file in path.rglob(ext):
                        name = font_file.stem.lower().replace("-", " ").replace("_", " ")
                        fonts[name] = str(font_file)
        
        # Add default PIL fonts
        try:
            default_font = ImageFont.load_default()
            if hasattr(default_font, 'path') and isinstance(default_font.path, str):
                fonts["default"] = default_font.path
        except Exception:
            pass
        
        return fonts
    
    def find_font(self, font_name: str) -> Optional[str]:
        """Find a font by name."""
        name_lower = font_name.lower()
        
        # Exact match
        if name_lower in self.available_fonts:
            return self.available_fonts[name_lower]
        
        # Partial match
        for name, path in self.available_fonts.items():
            if name_lower in name or name in name_lower:
                return path
        
        # Try common variations
        variations = [
            name_lower.replace(" ", ""),
            name_lower.replace(" ", "-"),
            name_lower.replace(" ", "_"),
            name_lower + " regular",
            name_lower + "-regular",
        ]
        
        for var in variations:
            if var in self.available_fonts:
                return self.available_fonts[var]
        
        return None
    
    def get_default_font(self, category: str = "sans-serif") -> str:
        """Get a default font from a category."""
        for font_name in self.COMMON_FONTS.get(category, self.COMMON_FONTS["sans-serif"]):
            path = self.find_font(font_name)
            if path:
                return path
        
        # Return first available font
        if self.available_fonts:
            return list(self.available_fonts.values())[0]
        
        return None
    
    def analyze_font_style(self, img: Image.Image, region: TextRegion) -> Dict:
        """Analyze the font style in a text region."""
        x1, y1, x2, y2 = region.bbox
        # Ensure valid bounding box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return {
                "is_bold": False, "is_italic": False, "is_serif": False,
                "is_monospace": False, "weight": "normal", "style": "normal"
            }
        cropped = img.crop((x1, y1, x2, y2)).convert('L')
        
        # Analyze characteristics
        analysis = {
            "is_bold": False,
            "is_italic": False,
            "is_serif": False,
            "is_monospace": False,
            "weight": "normal",
            "style": "normal"
        }
        
        # Convert to binary for analysis
        cv_crop = np.array(cropped)
        _, binary = cv2.threshold(cv_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate stroke width (indicates boldness)
        # Use distance transform
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if np.max(dist) > 0:
            avg_stroke = np.mean(dist[dist > 0])
            # Bold text typically has thicker strokes relative to height
            height = y2 - y1
            if avg_stroke / height > 0.08:
                analysis["is_bold"] = True
                analysis["weight"] = "bold"
        
        # Detect italic by checking for slant
        # Find contours and check angle
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            if len(all_points) >= 5:
                try:
                    ellipse = cv2.fitEllipse(all_points)
                    angle = ellipse[2]
                    if 70 < abs(angle) < 110:
                        pass  # Upright
                    elif abs(angle) < 20 or abs(angle) > 160:
                        analysis["is_italic"] = True
                        analysis["style"] = "italic"
                except Exception:
                    pass
        
        # Detect serif by looking for small details at stroke ends
        # This is simplified - real detection would be more sophisticated
        edges = cv2.Canny(binary, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.15:
            analysis["is_serif"] = True
        
        return analysis
    
    def get_matching_font(self, region: TextRegion, style_analysis: Dict) -> Tuple[str, int]:
        """Get a matching font for the detected text region."""
        # Determine font category
        if style_analysis.get("is_monospace"):
            category = "monospace"
        elif style_analysis.get("is_serif"):
            category = "serif"
        else:
            category = "sans-serif"
        
        font_path = self.get_default_font(category)
        
        return font_path, region.font_size


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REMOVAL (INPAINTING)
# ══════════════════════════════════════════════════════════════════════════════

class TextRemover:
    """Remove text from images using inpainting."""
    
    @staticmethod
    def create_text_mask(img: Image.Image, region: TextRegion, 
                        expand: int = 5) -> Image.Image:
        """Create a mask for the text region."""
        x1, y1, x2, y2 = region.bbox
        
        # Expand the region slightly
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(img.width, x2 + expand)
        y2 = min(img.height, y2 + expand)
        
        # Create mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        if abs(region.angle) > 5:
            # Rotated rectangle
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            angle_rad = math.radians(region.angle)
            
            corners = [
                (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
            ]
            rotated = []
            for x, y in corners:
                rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + cx
                ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + cy
                rotated.append((rx, ry))
            
            draw.polygon(rotated, fill=255)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=255)
        
        return mask
    
    @staticmethod
    def inpaint_region(img: Image.Image, mask: Image.Image, 
                      method: str = "telea") -> Image.Image:
        """Inpaint the masked region."""
        cv_img = pil_to_cv(img.convert('RGB'))
        cv_mask = np.array(mask)
        
        if method == "telea":
            inpainted = cv2.inpaint(cv_img, cv_mask, 7, cv2.INPAINT_TELEA)
        elif method == "ns":
            inpainted = cv2.inpaint(cv_img, cv_mask, 7, cv2.INPAINT_NS)
        else:
            # Simple fill with background color
            bg_color = get_dominant_color(img, (0, 0, img.width, img.height))
            cv_img[cv_mask > 0] = bg_color[::-1]  # RGB to BGR
            inpainted = cv_img
        
        return cv_to_pil(inpainted)
    
    @staticmethod
    def remove_text(img: Image.Image, region: TextRegion, 
                   method: str = "telea") -> Image.Image:
        """Remove text from the specified region."""
        mask = TextRemover.create_text_mask(img, region)
        return TextRemover.inpaint_region(img, mask, method)
    
    @staticmethod
    def smart_fill(img: Image.Image, region: TextRegion) -> Image.Image:
        """Smart fill using surrounding context."""
        x1, y1, x2, y2 = region.bbox
        
        # Expand region to get context
        expand = 20
        cx1 = max(0, x1 - expand)
        cy1 = max(0, y1 - expand)
        cx2 = min(img.width, x2 + expand)
        cy2 = min(img.height, y2 + expand)
        
        # Get the surrounding context
        context = img.crop((cx1, cy1, cx2, cy2))
        cv_context = pil_to_cv(context.convert('RGB'))
        
        # Create mask for the text area within context
        mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
        mask[y1-cy1:y2-cy1, x1-cx1:x2-cx1] = 255
        
        # Inpaint
        inpainted = cv2.inpaint(cv_context, mask, 7, cv2.INPAINT_TELEA)
        
        # Put back into original image
        result = img.copy()
        result.paste(cv_to_pil(inpainted), (cx1, cy1))
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
# TEXT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

class TextRenderer:
    """Render text with matching style."""
    
    def __init__(self, font_matcher: FontMatcher):
        self.font_matcher = font_matcher
    
    def load_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a font at the specified size."""
        try:
            if font_path and isinstance(font_path, str) and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            pass
        
        # Try common fonts
        common_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        
        for font in common_fonts:
            try:
                if os.path.exists(font):
                    return ImageFont.truetype(font, size)
            except Exception:
                continue
        
        # Last resort: default font
        try:
            return ImageFont.load_default()
        except Exception:
            return None
    
    def render_text(self, img: Image.Image, region: TextRegion, 
                   new_text: str, font_path: str = None) -> Image.Image:
        """Render new text in place of the original."""
        result = img.copy()
        draw = ImageDraw.Draw(result)
        
        x1, y1, x2, y2 = region.bbox
        
        # Load font
        if font_path is None:
            style_analysis = self.font_matcher.analyze_font_style(img, region)
            font_path, _ = self.font_matcher.get_matching_font(region, style_analysis)
        
        # Adjust font size to fit the region
        target_width = x2 - x1
        target_height = y2 - y1
        
        font_size = region.font_size
        font = self.load_font(font_path, font_size)
        
        if font is None:
            # Fallback to basic drawing
            draw.text((x1, y1), new_text, fill=region.font_color)
            return result
        
        # Adjust font size to fit
        for _ in range(20):
            bbox = font.getbbox(new_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            width_ratio = target_width / max(text_width, 1)
            height_ratio = target_height / max(text_height, 1)
            
            if 0.9 <= min(width_ratio, height_ratio) <= 1.1:
                break
            
            if width_ratio < 1 or height_ratio < 1:
                font_size = int(font_size * min(width_ratio, height_ratio) * 0.95)
            else:
                font_size = int(font_size * min(width_ratio, height_ratio) * 0.95)
            
            font_size = max(8, min(200, font_size))
            font = self.load_font(font_path, font_size)
        
        # Calculate position
        bbox = font.getbbox(new_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text in the region
        text_x = x1 + (target_width - text_width) / 2
        text_y = y1 + (target_height - text_height) / 2
        
        # Handle rotation
        if abs(region.angle) > 5:
            # Create text on transparent layer
            text_layer = Image.new('RGBA', result.size, (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            
            # Draw shadow first if present
            if region.has_shadow and region.shadow_color:
                shadow_offset = (2, 2)
                text_draw.text(
                    (text_x + shadow_offset[0], text_y + shadow_offset[1]),
                    new_text,
                    font=font,
                    fill=region.shadow_color + (200,)
                )
            
            # Draw outline if present
            if region.has_outline and region.outline_color:
                for ox, oy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
                    text_draw.text(
                        (text_x + ox, text_y + oy),
                        new_text,
                        font=font,
                        fill=region.outline_color + (255,)
                    )
            
            # Draw main text
            text_draw.text(
                (text_x, text_y),
                new_text,
                font=font,
                fill=region.font_color + (255,)
            )
            
            # Rotate around center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            text_layer = text_layer.rotate(-region.angle, center=(cx, cy), resample=Image.BICUBIC)
            
            # Composite
            result = Image.alpha_composite(result.convert('RGBA'), text_layer)
            result = result.convert('RGB')
        
        else:
            # Draw shadow first if present
            if region.has_shadow and region.shadow_color:
                shadow_offset = (2, 2)
                draw.text(
                    (text_x + shadow_offset[0], text_y + shadow_offset[1]),
                    new_text,
                    font=font,
                    fill=region.shadow_color
                )
            
            # Draw outline if present
            if region.has_outline and region.outline_color:
                for ox, oy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    draw.text(
                        (text_x + ox, text_y + oy),
                        new_text,
                        font=font,
                        fill=region.outline_color
                    )
            
            # Draw main text
            draw.text(
                (text_x, text_y),
                new_text,
                font=font,
                fill=region.font_color
            )
        
        return result
    
    def render_with_effects(self, img: Image.Image, region: TextRegion,
                           new_text: str, font_path: str = None,
                           custom_color: Tuple[int, int, int] = None,
                           custom_size: int = None,
                           add_shadow: bool = None,
                           shadow_color: Tuple[int, int, int] = None,
                           add_outline: bool = None,
                           outline_color: Tuple[int, int, int] = None) -> Image.Image:
        """Render text with customizable effects."""
        
        # Create a modified region with custom parameters
        modified_region = TextRegion(
            text=region.text,
            bbox=region.bbox,
            confidence=region.confidence,
            angle=region.angle,
            font_size=custom_size if custom_size else region.font_size,
            font_color=custom_color if custom_color else region.font_color,
            bg_color=region.bg_color,
            has_shadow=add_shadow if add_shadow is not None else region.has_shadow,
            shadow_color=shadow_color if shadow_color else region.shadow_color,
            has_outline=add_outline if add_outline is not None else region.has_outline,
            outline_color=outline_color if outline_color else region.outline_color
        )
        
        return self.render_text(img, modified_region, new_text, font_path)


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE TEXT EDITOR
# ══════════════════════════════════════════════════════════════════════════════

class ImageTextEditor:
    """Complete image text editing system."""
    
    def __init__(self):
        self.detector = TextDetector()
        self.font_matcher = FontMatcher()
        self.renderer = TextRenderer(self.font_matcher)
        self.remover = TextRemover()
    
    def detect_text(self, img: Image.Image, method: str = "auto") -> List[TextRegion]:
        """Detect all text in the image."""
        return self.detector.detect(img, method)
    
    def replace_text(self, img: Image.Image, region: TextRegion, 
                    new_text: str, font_path: str = None,
                    inpaint_method: str = "telea") -> Image.Image:
        """Replace text in the specified region."""
        # First remove the original text
        cleaned = self.remover.remove_text(img, region, inpaint_method)
        
        # Then render the new text
        result = self.renderer.render_text(cleaned, region, new_text, font_path)
        
        return result
    
    def replace_text_advanced(self, img: Image.Image, region: TextRegion,
                             new_text: str, font_path: str = None,
                             custom_color: Tuple[int, int, int] = None,
                             custom_size: int = None,
                             add_shadow: bool = None,
                             shadow_color: Tuple[int, int, int] = None,
                             add_outline: bool = None,
                             outline_color: Tuple[int, int, int] = None,
                             inpaint_method: str = "telea") -> Image.Image:
        """Replace text with advanced customization."""
        # Remove original text
        cleaned = self.remover.remove_text(img, region, inpaint_method)
        
        # Render with custom effects
        result = self.renderer.render_with_effects(
            cleaned, region, new_text, font_path,
            custom_color, custom_size,
            add_shadow, shadow_color,
            add_outline, outline_color
        )
        
        return result
    
    def batch_replace(self, img: Image.Image, 
                     replacements: Dict[int, str]) -> Image.Image:
        """Replace multiple text regions."""
        regions = self.detect_text(img)
        result = img.copy()
        
        for idx, new_text in replacements.items():
            if 0 <= idx < len(regions):
                result = self.replace_text(result, regions[idx], new_text)
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# Initialize editor
@st.cache_resource
def get_editor():
    return ImageTextEditor()

editor = get_editor()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    ocr_method = st.selectbox(
        "OCR Method",
        ["auto", "easyocr", "pytesseract", "opencv"],
        help="Auto tries EasyOCR first, then Pytesseract"
    )
    
    inpaint_method = st.selectbox(
        "Inpainting Method",
        ["telea", "ns", "simple"],
        help="Method to remove original text"
    )
    
    st.markdown("---")
    
    st.header("📖 How to Use")
    st.markdown("""
    1. **Upload** an image with text
    2. **Detect** text regions
    3. **Select** a text region to edit
    4. **Enter** new text
    5. **Customize** style if needed
    6. **Apply** and download
    """)
    
    st.markdown("---")
    
    st.header("📦 Required Packages")
    st.code("""
pip install streamlit
pip install Pillow
pip install numpy
pip install opencv-python
pip install easyocr
# or
pip install pytesseract
    """)

# Main content
uploaded = st.file_uploader("Upload an image with text", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    # Load image
    original = Image.open(uploaded).convert('RGB')
    
    # Store in session state
    if 'current_image' not in st.session_state:
        st.session_state.current_image = original.copy()
    if 'text_regions' not in st.session_state:
        st.session_state.text_regions = []
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    
    # Display original and current
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    
    with col2:
        st.subheader("Edited")
        st.image(st.session_state.current_image, use_container_width=True)
    
    # Reset button
    if st.button("🔄 Reset to Original"):
        st.session_state.current_image = original.copy()
        st.session_state.text_regions = []
        st.session_state.selected_region = None
        st.rerun()
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: DETECT TEXT
    # ═══════════════════════════════════════════════════════════════════════
    
    st.subheader("Step 1: Detect Text")
    
    if st.button("🔍 Detect Text in Image", type="primary"):
        with st.spinner("Detecting text..."):
            st.session_state.text_regions = editor.detect_text(
                st.session_state.current_image, 
                ocr_method
            )
        
        if st.session_state.text_regions:
            st.success(f"✅ Found {len(st.session_state.text_regions)} text region(s)")
        else:
            st.warning("No text detected. Try a different OCR method or image.")
    
    # Show detected regions
    if st.session_state.text_regions:
        st.markdown("### Detected Text Regions")
        
        # Create visualization
        viz_img = st.session_state.current_image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        for idx, region in enumerate(st.session_state.text_regions):
            x1, y1, x2, y2 = region.bbox
            color = (255, 0, 0) if idx == st.session_state.selected_region else (0, 255, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 15), f"#{idx+1}", fill=color)
        
        st.image(viz_img, caption="Detected text regions (green = detected, red = selected)", 
                use_container_width=True)
        
        # List regions
        for idx, region in enumerate(st.session_state.text_regions):
            with st.expander(f"Region #{idx+1}: \"{region.text[:50]}...\"" if len(region.text) > 50 else f"Region #{idx+1}: \"{region.text}\""):
                col_a, col_b, col_c = st.columns(3)
                
                col_a.markdown(f"**Text:** {region.text}")
                col_a.markdown(f"**Confidence:** {region.confidence:.1%}")
                
                col_b.markdown(f"**Font Size:** ~{region.font_size}px")
                col_b.markdown(f"**Angle:** {region.angle:.1f}°")
                
                # Show colors as swatches
                text_color_hex = '#{:02x}{:02x}{:02x}'.format(*region.font_color)
                bg_color_hex = '#{:02x}{:02x}{:02x}'.format(*region.bg_color) if region.bg_color else '#FFFFFF'
                
                col_c.markdown(f"**Text Color:** <span style='background:{text_color_hex};padding:2px 10px;border-radius:3px;'>{text_color_hex}</span>", unsafe_allow_html=True)
                col_c.markdown(f"**BG Color:** <span style='background:{bg_color_hex};padding:2px 10px;border-radius:3px;color:{'#000' if sum(region.bg_color or (255,255,255)) > 380 else '#fff'}'>{bg_color_hex}</span>", unsafe_allow_html=True)
                
                if region.has_shadow:
                    st.info("💫 Shadow detected")
                if region.has_outline:
                    st.info("🔲 Outline detected")
                
                if st.button(f"Select Region #{idx+1}", key=f"select_{idx}"):
                    st.session_state.selected_region = idx
                    st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: EDIT SELECTED TEXT
    # ═══════════════════════════════════════════════════════════════════════
    
    if st.session_state.selected_region is not None and st.session_state.text_regions:
        st.markdown("---")
        st.subheader("Step 2: Edit Selected Text")
        
        region = st.session_state.text_regions[st.session_state.selected_region]
        
        st.markdown(f"**Currently editing:** Region #{st.session_state.selected_region + 1}")
        st.markdown(f"**Original text:** `{region.text}`")
        
        # New text input
        new_text = st.text_input(
            "Enter new text",
            value=region.text,
            key="new_text_input"
        )
        
        # Advanced options
        with st.expander("🎨 Style Customization"):
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                use_custom_color = st.checkbox("Custom text color")
                if use_custom_color:
                    custom_color_hex = st.color_picker(
                        "Text Color",
                        value='#{:02x}{:02x}{:02x}'.format(*region.font_color)
                    )
                    custom_color = tuple(int(custom_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                else:
                    custom_color = None
                
                use_custom_size = st.checkbox("Custom font size")
                if use_custom_size:
                    custom_size = st.slider("Font Size", 8, 200, region.font_size)
                else:
                    custom_size = None
            
            with col_s2:
                add_shadow = st.checkbox("Add shadow", value=region.has_shadow)
                if add_shadow:
                    shadow_color_hex = st.color_picker(
                        "Shadow Color",
                        value='#{:02x}{:02x}{:02x}'.format(*(region.shadow_color or (100, 100, 100)))
                    )
                    shadow_color = tuple(int(shadow_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                else:
                    shadow_color = None
                
                add_outline = st.checkbox("Add outline", value=region.has_outline)
                if add_outline:
                    outline_color_hex = st.color_picker(
                        "Outline Color",
                        value='#{:02x}{:02x}{:02x}'.format(*(region.outline_color or (0, 0, 0)))
                    )
                    outline_color = tuple(int(outline_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                else:
                    outline_color = None
        
        # Apply button
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✏️ Preview Change", type="secondary"):
                with st.spinner("Generating preview..."):
                    preview = editor.replace_text_advanced(
                        st.session_state.current_image,
                        region,
                        new_text,
                        custom_color=custom_color,
                        custom_size=custom_size,
                        add_shadow=add_shadow,
                        shadow_color=shadow_color,
                        add_outline=add_outline,
                        outline_color=outline_color,
                        inpaint_method=inpaint_method
                    )
                    st.image(preview, caption="Preview", use_container_width=True)
        
        with col_btn2:
            if st.button("✅ Apply Change", type="primary"):
                with st.spinner("Applying changes..."):
                    st.session_state.current_image = editor.replace_text_advanced(
                        st.session_state.current_image,
                        region,
                        new_text,
                        custom_color=custom_color,
                        custom_size=custom_size,
                        add_shadow=add_shadow,
                        shadow_color=shadow_color,
                        add_outline=add_outline,
                        outline_color=outline_color,
                        inpaint_method=inpaint_method
                    )
                    
                    # Update region text
                    st.session_state.text_regions[st.session_state.selected_region] = TextRegion(
                        text=new_text,
                        bbox=region.bbox,
                        confidence=region.confidence,
                        angle=region.angle,
                        font_size=custom_size or region.font_size,
                        font_color=custom_color or region.font_color,
                        bg_color=region.bg_color,
                        has_shadow=add_shadow,
                        shadow_color=shadow_color,
                        has_outline=add_outline,
                        outline_color=outline_color
                    )
                    
                    st.success("✅ Text replaced successfully!")
                    st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: DOWNLOAD
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("Step 3: Download")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        export_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"])
    
    with col_dl2:
        if export_format == "JPEG":
            quality = st.slider("Quality", 50, 100, 95)
        else:
            quality = 95
    
    # Prepare download
    buf = io.BytesIO()
    
    if export_format == "PNG":
        st.session_state.current_image.save(buf, format="PNG", optimize=True)
        mime = "image/png"
        ext = "png"
    elif export_format == "JPEG":
        st.session_state.current_image.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
        ext = "jpg"
    else:
        st.session_state.current_image.save(buf, format="WEBP", quality=quality)
        mime = "image/webp"
        ext = "webp"
    
    st.download_button(
        f"⬇️ Download Edited Image ({export_format})",
        buf.getvalue(),
        f"edited_text.{ext}",
        mime,
        type="primary",
        use_container_width=True
    )

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to the Image Text Editor!</h3>
        <p>This tool allows you to detect text in images and replace it with new text 
        while preserving the original style, font, and effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Detect
        - Automatic text detection
        - Multiple OCR engines
        - Accurate bounding boxes
        - Rotation support
        """)
    
    with col2:
        st.markdown("""
        ### 🎨 Analyze
        - Font color detection
        - Background extraction
        - Shadow detection
        - Outline detection
        - Font size estimation
        """)
    
    with col3:
        st.markdown("""
        ### ✏️ Replace
        - Seamless text removal
        - Style-matching rendering
        - Custom color/size
        - Effect preservation
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Tips for Best Results
    
    1. **Use high-quality images** - Higher resolution = better detection
    2. **Clear, readable text** - Works best with printed/digital text
    3. **Good contrast** - Text should stand out from background
    4. **Simple backgrounds** - Solid colors work better than complex patterns
    5. **Standard fonts** - Common fonts are matched more accurately
    
    ### ⚠️ Limitations
    
    - Handwritten text may not be detected accurately
    - Complex backgrounds may affect inpainting quality
    - Artistic/decorative fonts may not be perfectly matched
    - Very small text may be hard to detect
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Image Text Editor • Detect • Analyze • Replace</p>
    <p style="font-size: 0.8em;">Powered by OpenCV, EasyOCR/Pytesseract, and PIL</p>
</div>
""", unsafe_allow_html=True)