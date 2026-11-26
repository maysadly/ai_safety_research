"""Export poster/poster.html to an exact-size PDF (A0 portrait by default) and a PNG preview.

Usage: python poster/export.py [--width-mm 841 --height-mm 1189]
Outputs: poster/poster.pdf, poster/poster_preview.png, results/figures/fig1_method.{png,svg}
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MM_TO_PX = 96 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=841)
    ap.add_argument("--height-mm", type=float, default=1189)
    a = ap.parse_args()
    w_px, h_px = round(a.width_mm * MM_TO_PX), round(a.height_mm * MM_TO_PX)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": w_px, "height": h_px}, device_scale_factor=1)
        page.goto((HERE / "poster.html").as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("document.fonts.ready")
        overflow = page.evaluate("""() => {
            const poster = document.querySelector('.poster');
            const limit = poster.getBoundingClientRect().bottom - parseFloat(getComputedStyle(poster).paddingBottom);
            const out = [];
            document.querySelectorAll('.col, .results, .masthead, .hero').forEach(el => {
              let bottom = 0;
              for (const c of el.children) bottom = Math.max(bottom, c.getBoundingClientRect().bottom);
              const name = el.className;
              if (bottom > limit + 1) out.push(name + ' overflows by ' + Math.round(bottom - limit) + 'px');
              else out.push(name + ' free ' + Math.round(limit - bottom) + 'px');
            });
            document.querySelectorAll('p, li, h1, h2').forEach(el => {
              if (el.scrollWidth > el.clientWidth + 2) out.push('wide: ' + el.textContent.slice(0, 40));
            });
            return out;
        }""")
        print("overflow:", overflow or "none")
        page.pdf(path=str(HERE / "poster.pdf"), width=f"{a.width_mm}mm", height=f"{a.height_mm}mm",
                 print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        page.screenshot(path=str(HERE / "poster_preview.png"), full_page=False, scale="css")
        fig1 = page.query_selector("#fig1")
        if fig1:
            fig1.screenshot(path=str(ROOT / "results/figures/fig1_method.png"))
            svg = page.evaluate("() => document.querySelector('#fig1 svg')?.outerHTML")
            if svg:
                (ROOT / "results/figures/fig1_method.svg").write_text(svg)
        b.close()
    print("wrote poster/poster.pdf, poster/poster_preview.png")


if __name__ == "__main__":
    main()
