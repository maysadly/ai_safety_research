"""Build poster v2 (A1 portrait, all sections required by the course's Part C) from the results JSON files,
then export it to an exact-size PDF and a PNG preview.

Usage: python poster/make_a1_figures.py && python poster/build_poster_v2.py
Outputs: poster/poster_v2.html, poster/poster_v2_A1.pdf, poster/poster_v2_preview.png
"""
import json
from pathlib import Path
from string import Template

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "results"
W_MM, H_MM, W_PX, H_PX = 594, 841, 2245, 3179

m_main = json.load(open(R / "phase3_metrics.json"))
m_pair = json.load(open(R / "phase3_metrics_pair.json"))
s_main = json.load(open(R / "phase2_summary_main.json"))
s_pair = json.load(open(R / "phase2_summary_pair.json"))
p1 = json.load(open(R / "phase1_vectors.json"))
jv = json.load(open(R / "judge_validation.json"))


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


def a2(x):
    return f"{x:.2f}"


def cell(e):
    return f"{a2(e['test_auc'])} [{a2(e['ci95'][0])}, {a2(e['ci95'][1])}]"


mon = m_pair["monitors"]
k0, k5 = mon["0"], mon["5"]
delta = {k: mon[k]["delta_refusal+emotion_minus_refusal"] for k in mon}
wr = m_main["wrappers"]
w_emo = wr["emotional_pressure"]["prompt_projection_shift_vs_neutral"]
win = m_pair["h1"]["within_prompt"]["tokens_1-5"]
loto = p1["heldout"]["loto_acc_denoised"]
best_l = max(range(7, 28), key=lambda l: loto[l])
fa0, fa5 = k0["false_alarms"], k5["false_alarms"]
ks = [int(k) for k in mon]
names = ("emotion", "refusal", "refusal+emotion")
k_high = max(mon[str(k)][n]["test_auc"] for k in ks if k > 0 for n in names)
rnd_meds = [mon[str(k)]["random"]["median"] for k in ks if k > 0]


def rnd_cell(e):
    r = e["random"]
    return f"{a2(r['median'])} [{a2(r['p2.5_p97.5'][0])}, {a2(r['p2.5_p97.5'][1])}]"


def sgn(x):
    """Signed number; 3 decimals when it would otherwise round to a signed zero."""
    return f"{x:+.3f}" if abs(x) < 0.005 else f"{x:+.2f}"


def dcell(d):
    return f"{sgn(d['point'])} [{sgn(d['ci95'][0])}, {sgn(d['ci95'][1])}]"


refs = [
    'N. Sofroniew et al., "Emotion concepts and their function in a large language model," <i>Transformer Circuits Thread</i>, Anthropic, 2026.',
    'A. Arditi et al., "Refusal in language models is mediated by a single direction," in <i>Proc. NeurIPS</i>, vol. 37, 2024.',
    'P. Chao et al., "JailbreakBench: An open robustness benchmark for jailbreaking large language models," in <i>Proc. NeurIPS Datasets and Benchmarks Track</i>, 2024.',
    'A. Korznikov, A. Galichin, A. Dontsov, O. Y. Rogov, I. Oseledets, and E. Tutubalina, "The rogue scalpel: Activation steering compromises LLM safety," arXiv:2509.22067, 2025.',
    'Y. Wang et al., "Read-best is not steer-best: A probing&#8211;steering layer dissociation in omni-modal large language models," arXiv:2609.22135, 2026.',
    'J. Xu et al., "Qwen2.5-Omni technical report," arXiv:2503.20215, 2025.',
    'A. Goldenberg and J. J. Gross, "Do large language models have emotions?" arXiv:2606.14742, 2026.',
    'M. Mazeika et al., "HarmBench: A standardized evaluation framework for automated red teaming and robust refusal," in <i>Proc. ICML</i>, PMLR vol. 235, 2024.',
    'P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, "Jailbreaking black box large language models in twenty queries," in <i>Proc. IEEE SaTML</i>, 2025.',
]

FIG1 = """
<svg viewBox="0 0 688 658" xmlns="http://www.w3.org/2000/svg" font-family="Seravek, 'Avenir Next', 'Gill Sans', sans-serif" role="img" aria-labelledby="f1t">
  <title id="f1t">System architecture: request, model activations, projection on fixed directions, logistic monitor, judge labels</title>
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#16202c"/></marker></defs>
  <g fill="#16202c">
    <circle cx="24" cy="47" r="21" fill="#16202c"/><text x="24" y="55" fill="#fff" text-anchor="middle" font-size="23" font-weight="600">1</text>
    <rect x="56" y="4" width="628" height="86" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="76" y="39" font-size="25" font-weight="600">Request from JailbreakBench</text>
    <text x="76" y="72" font-size="21" fill="#4d5868">harmful or benign; fixed wrapper or PAIR prompt</text>
    <line x1="370" y1="92" x2="370" y2="118" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>
    <circle cx="24" cy="165" r="21" fill="#16202c"/><text x="24" y="173" fill="#fff" text-anchor="middle" font-size="23" font-weight="600">2</text>
    <rect x="56" y="122" width="628" height="86" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="76" y="157" font-size="25" font-weight="600">Qwen2.5-Omni-7B answers (text only)</text>
    <text x="76" y="190" font-size="21" fill="#4d5868">read activations: 28 layers &#215; tokens 0&#8211;20</text>
    <line x1="370" y1="210" x2="370" y2="236" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>
    <circle cx="24" cy="283" r="21" fill="#16202c"/><text x="24" y="291" fill="#fff" text-anchor="middle" font-size="23" font-weight="600">3</text>
    <rect x="56" y="240" width="628" height="176" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="76" y="275" font-size="25" font-weight="600">Project onto fixed directions</text>
    <rect x="76" y="292" width="186" height="66" rx="8" fill="#fde9df"/><rect x="76" y="292" width="9" height="66" fill="#f47230"/>
    <text x="96" y="320" font-size="20">5 emotion</text><text x="96" y="346" font-size="20">vectors [1]</text>
    <rect x="276" y="292" width="186" height="66" rx="8" fill="#d9f4ee"/><rect x="276" y="292" width="9" height="66" fill="#00b894"/>
    <text x="296" y="320" font-size="20">refusal</text><text x="296" y="346" font-size="20">direction [2]</text>
    <rect x="476" y="292" width="190" height="66" rx="8" fill="#efefef"/><rect x="476" y="292" width="9" height="66" fill="#9a9a9a"/>
    <text x="496" y="320" font-size="20">200 random</text><text x="496" y="346" font-size="20">directions</text>
    <text x="76" y="394" font-size="20" fill="#4d5868">one number per direction, per layer and token</text>
    <line x1="370" y1="418" x2="370" y2="444" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>
    <circle cx="24" cy="491" r="21" fill="#16202c"/><text x="24" y="499" fill="#fff" text-anchor="middle" font-size="23" font-weight="600">4</text>
    <rect x="56" y="448" width="628" height="86" rx="12" fill="#fff" stroke="#16202c" stroke-width="3.5"/>
    <text x="76" y="483" font-size="25" font-weight="600">Logistic monitor: P(will comply)</text>
    <text x="76" y="516" font-size="21" fill="#4d5868">alarm above a threshold &#8594; pause and escalate</text>
    <circle cx="24" cy="611" r="21" fill="#16202c"/><text x="24" y="619" fill="#fff" text-anchor="middle" font-size="23" font-weight="600">5</text>
    <rect x="56" y="568" width="628" height="86" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.5" stroke-dasharray="10 7"/>
    <text x="76" y="603" font-size="25" font-weight="600">HarmBench judge [8]</text>
    <text x="76" y="636" font-size="21" fill="#4d5868">labels each answer: complied or refused</text>
    <line x1="370" y1="566" x2="370" y2="540" stroke="#16202c" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ah)"/>
  </g>
</svg>
"""

v = dict(
    fig1=FIG1, loto_best=a2(loto[best_l]),
    agree_main=round(jv["llm_main_all"]["raw_agreement"] * 100), agree_pair=round(jv["llm_pair_all"]["raw_agreement"] * 100),
    main_n2=f"{s_main['n']:,}", main_rate=pct(s_main["harmful_compliance_overall"]["rate"]),
    pair_n=s_pair["harmful_compliance_overall"]["n"], pair_rate=pct(s_pair["harmful_compliance_overall"]["rate"]),
    k0_lo=a2(min(k0[n]["test_auc"] for n in names)), k0_hi=a2(max(k0[n]["test_auc"] for n in names)),
    k_high=a2(k_high), rnd_lo=a2(min(rnd_meds)), rnd_hi=a2(max(rnd_meds)),
    n_test=m_pair["n"]["harmful_test"], n_test_beh=m_pair["n"]["harmful_test"] // 10,
    test_comp=round(m_pair["n"]["harmful_test"] * m_pair["test_compliance_rate"]),
    r0=rnd_cell(k0), r5=rnd_cell(k5), e0=cell(k0["emotion"]), e5=cell(k5["emotion"]),
    c0=cell(k0["refusal"]), c5=cell(k5["refusal"]), d0=cell(k0["refusal+emotion"]), d5=cell(k5["refusal+emotion"]),
    dd0=dcell(delta["0"]), dd5=dcell(delta["5"]),
    d_min=f"{min(d['point'] for d in delta.values()):+.2f}", d_max=f"{max(d['point'] for d in delta.values()):+.2f}",
    dz_desp=f"{win['desperate']['d_z']:+.2f}", dz_calm=f"{win['calm']['d_z']:+.2f}", dz_rnd=a2(win["random_abs_dz_p95"]),
    shift_desp=f"{w_emo['desperate']['in_sd_units']:+.2f}", shift_calm=f"{w_emo['calm']['in_sd_units']:+.2f}",
    neu_comp=pct(wr["neutral"]["harmful_compliance"]), emo_comp=pct(wr["emotional_pressure"]["harmful_compliance"]),
    rp_comp=pct(wr["roleplay"]["harmful_compliance"]),
    neu_over=pct(wr["neutral"]["benign_over_refusal"], 0), emo_over=pct(wr["emotional_pressure"]["benign_over_refusal"], 0),
    fa0_ref_benign=pct(fa0["refusal"]["alarm_rate_on_benign"]["rate"], 0),
    fa5_emo_benign=pct(fa5["emotion"]["alarm_rate_on_benign"]["rate"], 0),
    refs="\n".join(f"<li>{r}</li>" for r in refs),
)
(HERE / "poster_v2.html").write_text(Template((HERE / "poster_v2_template.html").read_text()).substitute(v))
print("wrote poster/poster_v2.html")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": W_PX, "height": H_PX}, device_scale_factor=1)
    page.goto((HERE / "poster_v2.html").as_uri())
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready")
    report = page.evaluate("""() => {
        const poster = document.querySelector('.poster');
        const limit = poster.getBoundingClientRect().bottom - parseFloat(getComputedStyle(poster).paddingBottom);
        const out = [];
        document.querySelectorAll('.col, .subcol, .masthead').forEach((el, i) => {
          let bottom = 0;
          for (const c of el.children) bottom = Math.max(bottom, c.getBoundingClientRect().bottom);
          out.push(el.className + i + (bottom > limit + 1 ? ' OVERFLOWS by ' + Math.round(bottom - limit) : ' free ' + Math.round(limit - bottom)));
        });
        document.querySelectorAll('p, li, td, h1, h2').forEach(el => {
          if (el.scrollWidth > el.clientWidth + 2) out.push('too wide: ' + el.textContent.slice(0, 50));
        });
        return out;
    }""")
    print("layout:", report)
    page.pdf(path=str(HERE / "poster_v2_A1.pdf"), width=f"{W_MM}mm", height=f"{H_MM}mm", print_background=True,
             margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
    page.screenshot(path=str(HERE / "poster_v2_preview.png"))
    b.close()
print("wrote poster/poster_v2_A1.pdf, poster/poster_v2_preview.png")
