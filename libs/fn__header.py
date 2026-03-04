"""
libs/fn__header.py
==================
Page header: logo rendering and full page-header bootstrap.
Called once at the top of every page.

Number range: 201–202  (UI series)

Function index
--------------
  f201__render_logo          – render logo (PNG/SVG) with sizing and margin control
  f202__create_page_header   – set_page_config → CSS → logo / title row
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import streamlit as st

try:
    from libs.fn__ui import f203__inject_global_css
except ImportError:
    from fn__ui import f203__inject_global_css  # fallback when run from root

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False




# ─────────────────────────────────────────────────────────────────────────────
# f201
# ─────────────────────────────────────────────────────────────────────────────
def f201__render_logo(
    logo_path: str = "./img/logo.png",
    *,
    width: str = "100%",
    height: str = "auto",
    max_width: str | None = None,
    max_height: str | None = None,
    margin: str = "0",
    padding: str = "0",
) -> None:
    """
    Render the app logo from a local file.
    Supports PNG, JPG, and SVG. Silently skips if the file is missing
    or (for raster formats) Pillow is unavailable.

    Sizing and spacing (CSS values, e.g. "100%", "120px", "10px 5px"):
      width, height      – image dimensions
      max_width, max_height – optional constraints
      margin, padding    – full control over logo container spacing
    """
    path = Path(logo_path)
    if not path.exists():
        return

    # Build image style from sizing params
    img_style_parts = [
        f"width:{width}",
        f"height:{height}",
        "display:block",
    ]
    if max_width:
        img_style_parts.append(f"max-width:{max_width}")
    if max_height:
        img_style_parts.append(f"max-height:{max_height}")
    img_style = ";".join(img_style_parts)

    # Container style: margin and padding
    container_style = f"margin:{margin};padding:{padding}"

    suffix = path.suffix.lower()

    # SVG: embed as data URI (no PIL needed)
    if suffix == ".svg":
        try:
            svg_bytes = path.read_bytes()
            svg_b64 = base64.b64encode(svg_bytes).decode()
            st.markdown(
                f'<div style="{container_style}">'
                f'<img src="data:image/svg+xml;base64,{svg_b64}" style="{img_style}"/>'
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        return

    # Raster (PNG, JPG, etc.): use PIL
    if not _PIL_AVAILABLE:
        return
    try:
        image = Image.open(path)
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f'<div style="{container_style}">'
            f'<img src="data:image/png;base64,{img_b64}" style="{img_style}"/>'
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass







# ─────────────────────────────────────────────────────────────────────────────
# f202
# ─────────────────────────────────────────────────────────────────────────────
def f202__create_page_header(
    subtitle: str = "EPW Data Analysis & PV generation estimates"
        " [by <a href='https://absrd.xyz/' target='_blank'>AB.S.RD</a>]",
    caption: str = "",
    logo_path: str = "./img/logo_camus_02.png",
    *,
    logo_width: str = "100%",
    logo_height: str = "auto",
    logo_max_width: str | None = None,
    logo_max_height: str | None = None,
    logo_margin: str = "0",
    logo_padding: str = "0",
) -> None:
    """
    Full page bootstrap: set_page_config → inject CSS → render logo + title.

    This must be the **first** Streamlit call on each page.

    Logo sizing and spacing (CSS values, e.g. "120px", "10px 5px"):
      logo_width, logo_height       – image dimensions
      logo_max_width, logo_max_height – optional constraints
      logo_margin, logo_padding      – full control over logo container
    """
    # 1. Page config
    st.set_page_config(
        page_title="Climate Analysis App",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Global CSS (fonts, layout, component styles)
    f203__inject_global_css()

    # 3. Logo + title row
    col_logo, col_title, spacing, col_logo_2 = st.columns([110, 500, 900, 100], gap="small")

    # col_title, col_logo, spacing, col_logo_2 = st.columns([500, 200, 900, 100], gap="small")


    with col_title:
        st.markdown(
            f"""
            <div style="margin:50px 0px 10px 10px;padding:0;">
                <h2 class="custom-title">Climate Data Analysis App</h2>
                <p class="custom-caption">
                    {subtitle} {caption}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_logo:
        f201__render_logo(
            logo_path="./img/EETRA_logo_rect.png",
            width="100%",
            height="200px",  
            max_width=logo_max_width,
            max_height=logo_max_height,
            margin="36px 0px 0px 0px",
            padding=logo_padding,
        )

    with col_logo_2:
        f201__render_logo(
            logo_path="./img/logo_camus_02.png",
            width="100%",
            height="60px",
            max_width=logo_max_width,
            max_height=logo_max_height,
            margin="60px 0px 0px 0px",
            padding="0",
        )

    # 4. Bottom green accent line
    st.markdown(
        """
        <div style="margin: 0px 0; padding: 0;">
            <hr style="width: 30%; height: 1px;
                background-color: #2a7d2e; border: none; margin: 5px 0px 40px 0px; padding: 0;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Backward-compatible aliases ───────────────────────────────────────────────
create_page_header = f202__create_page_header
