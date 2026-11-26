# Phase 4 report: poster

**Status:** done. `poster/poster.pdf` is a single A0 portrait page (841 × 1189 mm, verified) with vector text and embedded fonts. Preview: `poster/poster_preview.png`.

## Build

```
python scripts/07_figures.py --poster    # poster-sized figures -> poster/figures/*.svg
python poster/build_poster.py            # fills poster/poster_template.html from results/*.json -> poster/poster.html
python poster/export.py                  # Playwright/Chromium -> poster.pdf (exact size) + preview PNG + Fig. 1 export
```

- **Numbers.** Every number on the poster is read from `results/phase3_metrics*.json`, `phase2_summary_*.json`, `phase1_vectors.json` and `judge_validation.json`. None is typed by hand.
- **Figures.** They are embedded as SVG.
- **Fig. 1.** The method diagram is an inline SVG in the poster. It is also exported to `results/figures/fig1_method.{svg,png}`.
- **Fit check.** `export.py` measures each column against the page; the final build has 3 px and 33 px of spare height.

## Design

- **Typeface:** Seravek, used throughout, figures included.
- **Body text** is 32 px = 24 pt at A0, the minimum in TASK.md. Takeaways are 34 px, headings 50 px, the title 118 px.
- **Smaller text:**

  | Element | Size |
  |---|---|
  | Figure notes | 27 px (20 pt) |
  | References | 22 px (16.5 pt) |

- **Data colours** are the same validated, colour-blind-safe palette as in the report figures. Every series also has a text label, so colour is never the only cue.
- **Layout:**
  - The full-width band under the title is Fig. 4, the clearest picture of the main result: emotion projections are identical at the end of the prompt and diverge only once the answer starts.
  - Left column: motivation, hypotheses with verdicts, method (Fig. 1) and references.
  - Right two columns: the short answer, Fig. 2 (headline), Fig. 3, Fig. 5, conclusions and limitations.
- **Ethics:** no harmful text anywhere. Only category-level wording, labels and aggregates appear. A note on the poster says so.

## Defaults used (please confirm or correct)

TASK.md asks the user for these. The run was autonomous, so I used defaults:

| Item | Used | Where to change |
|---|---|---|
| Size and orientation | A0 portrait | `python poster/export.py --width-mm … --height-mm …`; the layout is tuned for A0 portrait |
| Course name / code | "AI Safety course project, 2026" | byline in `poster/poster_template.html` |
| QR code | https://github.com/maysadly/ai_safety_research | `poster/qr_repo.svg` (made locally with `qrcode`) |

## References

All 9 references were verified against arXiv metadata, venue proceedings and Crossref (`poster/references.md`). Two were added to the TASK.md list because they are used: HarmBench [8] and PAIR [9]. The PAIR entry cites its SaTML 2025 version.
