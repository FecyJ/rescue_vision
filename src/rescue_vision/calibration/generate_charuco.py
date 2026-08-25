#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


# ============================================================
# Configuration
# ============================================================

# A4 landscape
PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0

# ChArUco board
SQUARES_X = 7
SQUARES_Y = 5

SQUARE_LENGTH_MM = 35.0
MARKER_LENGTH_MM = 25.0

ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100

# ChArUco board origin relative to paper bottom-left
BOARD_LEFT_MM = 15.0
BOARD_BOTTOM_MM = 15.0

# Raster output resolution
DPI = 600

# Output files
PNG_PATH = Path("charuco_a4_7x5_35mm_25mm_600dpi.png")
PDF_PATH = Path("charuco_a4_7x5_35mm_25mm_print.pdf")

# Use current OpenCV pattern unless you explicitly need legacy
USE_LEGACY_PATTERN = False

# ============================================================
# Reference marks configuration
# ============================================================

# Light gray so that these guides do not interfere with detection.
GUIDE_COLOR = 180   # 0=black, 255=white
GUIDE_THICKNESS_MM = 0.3

# Paper bottom-left alignment L mark
DRAW_PAPER_ALIGNMENT_L = True
PAPER_ALIGNMENT_L_LEN_MM = 12.0
PAPER_ALIGNMENT_OFFSET_MM = 10.0  # offset from paper edges

# Board-origin alignment L mark
DRAW_BOARD_ORIGIN_L = True
BOARD_ORIGIN_L_LEN_MM = 8.0
BOARD_ORIGIN_L_GAP_MM = 2.0  # leave a small gap from board edges

# 100 mm verification line
DRAW_VERIFY_LINE = True
VERIFY_LINE_LENGTH_MM = 100.0
VERIFY_LINE_X_MM = 175.0
VERIFY_LINE_Y_MM = 12.0

# Optional text labels
DRAW_LABELS = True


# ============================================================
# Utilities
# ============================================================

MM_PER_INCH = 25.4


def mm_to_px(mm_value: float) -> int:
    return round(mm_value / MM_PER_INCH * DPI)


def bottom_left_mm_to_image_xy(x_mm: float, y_mm: float) -> tuple[int, int]:
    """
    Convert bottom-left-origin physical mm coordinates
    to top-left-origin image pixel coordinates.
    """
    x_px = mm_to_px(x_mm)
    y_px = mm_to_px(PAGE_HEIGHT_MM - y_mm)
    return x_px, y_px


def draw_line_mm(
    img: np.ndarray,
    x1_mm: float,
    y1_mm: float,
    x2_mm: float,
    y2_mm: float,
    color: int,
    thickness_mm: float,
) -> None:
    p1 = bottom_left_mm_to_image_xy(x1_mm, y1_mm)
    p2 = bottom_left_mm_to_image_xy(x2_mm, y2_mm)
    thickness_px = max(1, mm_to_px(thickness_mm))
    cv2.line(img, p1, p2, color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


def draw_text_mm(
    img: np.ndarray,
    text: str,
    x_mm: float,
    y_mm: float,
    scale: float = 0.5,
    color: int = 120,
    thickness: int = 1,
) -> None:
    x_px, y_px = bottom_left_mm_to_image_xy(x_mm, y_mm)
    cv2.putText(
        img,
        text,
        (x_px, y_px),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# Validate geometry
# ============================================================

board_width_mm = SQUARES_X * SQUARE_LENGTH_MM
board_height_mm = SQUARES_Y * SQUARE_LENGTH_MM

board_right_mm = BOARD_LEFT_MM + board_width_mm
board_top_mm = BOARD_BOTTOM_MM + board_height_mm

if board_right_mm > PAGE_WIDTH_MM:
    raise ValueError(
        f"Board exceeds page width: {board_right_mm:.2f} > "
        f"{PAGE_WIDTH_MM:.2f} mm"
    )

if board_top_mm > PAGE_HEIGHT_MM:
    raise ValueError(
        f"Board exceeds page height: {board_top_mm:.2f} > "
        f"{PAGE_HEIGHT_MM:.2f} mm"
    )


# ============================================================
# Create ChArUco board
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH_MM,
    MARKER_LENGTH_MM,
    dictionary,
)

if hasattr(board, "setLegacyPattern"):
    board.setLegacyPattern(USE_LEGACY_PATTERN)

board_width_px = mm_to_px(board_width_mm)
board_height_px = mm_to_px(board_height_mm)

board_img = board.generateImage(
    (board_width_px, board_height_px),
    marginSize=0,
    borderBits=1,
)


# ============================================================
# Create A4 raster page
# ============================================================

page_width_px = mm_to_px(PAGE_WIDTH_MM)
page_height_px = mm_to_px(PAGE_HEIGHT_MM)

page = np.full(
    (page_height_px, page_width_px),
    255,
    dtype=np.uint8,
)

left_px = mm_to_px(BOARD_LEFT_MM)

# Image origin is top-left, physical paper origin is bottom-left.
top_mm = PAGE_HEIGHT_MM - BOARD_BOTTOM_MM - board_height_mm
top_px = mm_to_px(top_mm)

x0 = left_px
y0 = top_px
x1 = x0 + board_width_px
y1 = y0 + board_height_px

page[y0:y1, x0:x1] = board_img


# ============================================================
# Draw reference guides
# ============================================================

# 1) Paper bottom-left L mark
if DRAW_PAPER_ALIGNMENT_L:
    ox = PAPER_ALIGNMENT_OFFSET_MM
    oy = PAPER_ALIGNMENT_OFFSET_MM
    L = PAPER_ALIGNMENT_L_LEN_MM

    # Horizontal leg
    draw_line_mm(
        page, ox, oy, ox + L, oy,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )
    # Vertical leg
    draw_line_mm(
        page, ox, oy, ox, oy + L,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )

    if DRAW_LABELS:
        draw_text_mm(page, "Paper O", ox + 2, oy + L + 2, scale=0.4)

# 2) Board-origin L mark (around the board lower-left corner, outside board)
if DRAW_BOARD_ORIGIN_L:
    bx = BOARD_LEFT_MM
    by = BOARD_BOTTOM_MM
    gap = BOARD_ORIGIN_L_GAP_MM
    L = BOARD_ORIGIN_L_LEN_MM

    # Horizontal leg: left of board, aligned with board bottom
    draw_line_mm(
        page, bx - L - gap, by, bx - gap, by,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )
    # Vertical leg: below board, aligned with board left
    draw_line_mm(
        page, bx, by - L - gap, bx, by - gap,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )

    if DRAW_LABELS:
        draw_text_mm(page, "Board O", bx - L - 1, by - 3, scale=0.4)

# 3) 100 mm verification line
if DRAW_VERIFY_LINE:
    x0v = VERIFY_LINE_X_MM
    y0v = VERIFY_LINE_Y_MM
    x1v = x0v + VERIFY_LINE_LENGTH_MM

    if x1v > PAGE_WIDTH_MM - 5:
        raise ValueError("Verification line exceeds page width.")

    draw_line_mm(
        page, x0v, y0v, x1v, y0v,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )

    # End ticks
    tick_half = 1.8
    draw_line_mm(
        page, x0v, y0v - tick_half, x0v, y0v + tick_half,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )
    draw_line_mm(
        page, x1v, y0v - tick_half, x1v, y0v + tick_half,
        GUIDE_COLOR, GUIDE_THICKNESS_MM
    )

    if DRAW_LABELS:
        draw_text_mm(
            page,
            f"{VERIFY_LINE_LENGTH_MM:.0f} mm",
            x0v + 35,
            y0v + 3,
            scale=0.45,
        )


# ============================================================
# Save PNG
# ============================================================

ok = cv2.imwrite(
    str(PNG_PATH),
    page,
    [cv2.IMWRITE_PNG_COMPRESSION, 3],
)

if not ok:
    raise RuntimeError(f"Failed to save PNG: {PNG_PATH}")


# ============================================================
# Create exact-size A4 PDF
# ============================================================

page_width_pt = PAGE_WIDTH_MM * mm
page_height_pt = PAGE_HEIGHT_MM * mm

png_image = Image.open(PNG_PATH)

pdf = canvas.Canvas(
    str(PDF_PATH),
    pagesize=(page_width_pt, page_height_pt),
)

pdf.drawInlineImage(
    png_image,
    0,
    0,
    width=page_width_pt,
    height=page_height_pt,
)

pdf.showPage()
pdf.save()


# ============================================================
# Print summary
# ============================================================

print("Generated successfully.")
print()
print("Files:")
print(f"  PNG: {PNG_PATH.resolve()}")
print(f"  PDF: {PDF_PATH.resolve()}")
print()
print("Paper:")
print(f"  size          = {PAGE_WIDTH_MM:.1f} x {PAGE_HEIGHT_MM:.1f} mm")
print("  orientation   = landscape")
print(f"  raster DPI    = {DPI}")
print(f"  raster size   = {page_width_px} x {page_height_px} px")
print()
print("ChArUco:")
print(f"  squares       = {SQUARES_X} x {SQUARES_Y}")
print(f"  square length = {SQUARE_LENGTH_MM:.3f} mm")
print(f"  marker length = {MARKER_LENGTH_MM:.3f} mm")
print(f"  board size    = {board_width_mm:.3f} x {board_height_mm:.3f} mm")
print(f"  legacyPattern = {USE_LEGACY_PATTERN}")
print()
print("Board origin relative to paper bottom-left:")
print(f"  x = {BOARD_LEFT_MM:.3f} mm")
print(f"  y = {BOARD_BOTTOM_MM:.3f} mm")
print()
print("Printing:")
print("  1. Print the PDF, not the PNG.")
print("  2. Use A4 paper, landscape.")
print("  3. Select 100% / Actual Size.")
print("  4. Disable Fit to Page / Scale to Fit.")
print("  5. Disable borderless enlargement.")
print()
print("Verification after printing:")
print(f"  5 squares should measure {5 * SQUARE_LENGTH_MM:.2f} mm")
print(f"  4 squares should measure {4 * SQUARE_LENGTH_MM:.2f} mm")
print(f"  whole board should measure {board_width_mm:.2f} x {board_height_mm:.2f} mm")
print(f"  board origin offset should be {BOARD_LEFT_MM:.2f} mm (x), {BOARD_BOTTOM_MM:.2f} mm (y)")
print(f"  verification line should be {VERIFY_LINE_LENGTH_MM:.2f} mm")