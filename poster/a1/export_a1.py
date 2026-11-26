"""Export poster/a1/poster_a1.html to an exact A1 landscape PDF (841 x 594 mm) and a PNG preview,
and report how much vertical space each column has left (negative = overflow).

Usage: python poster/a1/export_a1.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 3179, 2245  # A1 landscape at 96 px/in


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.goto((HERE / "poster_a1.html").as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("document.fonts.ready")
        rep = page.evaluate("""() => {
            const poster = document.querySelector('.poster');
            const limit = poster.getBoundingClientRect().bottom - parseFloat(getComputedStyle(poster).paddingBottom);
            const out = [];
            document.querySelectorAll('.col').forEach((el, i) => {
              let bottom = 0;
              for (const c of el.children) bottom = Math.max(bottom, c.getBoundingClientRect().bottom);
              out.push('col' + (i + 1) + ': ' + Math.round(limit - bottom) + 'px free');
            });
            document.querySelectorAll('p, li, h1, h2, td, th').forEach(el => {
              if (el.scrollWidth > el.clientWidth + 2) out.push('wide: ' + el.textContent.slice(0, 40));
            });
            return out;
        }""")
        print(rep)
        page.pdf(path=str(HERE / "poster_A1_landscape.pdf"), width="841mm", height="594mm", print_background=True,
                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        page.screenshot(path=str(HERE / "poster_A1_landscape_preview.png"))
        fig1 = page.query_selector("figure svg")
        if fig1:
            (ROOT / "results/figures/a1_fig1_architecture.svg").write_text(fig1.evaluate("e => e.outerHTML"))
            page.query_selector("figure").screenshot(path=str(ROOT / "results/figures/a1_fig1_architecture.png"))
        b.close()
    print("wrote poster/a1/poster_A1_landscape.pdf and preview")


if __name__ == "__main__":
    main()
