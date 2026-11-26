"""Build poster/poster.html (A0 portrait) from the results JSON files, so every number on the poster comes
from the analysis outputs. Then run poster/export.py for the PDF and PNG.

Usage: python poster/build_poster.py && python poster/export.py
"""
import json
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "results"

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


mon = m_pair["monitors"]
k0, k5 = mon["0"], mon["5"]
delta = {k: mon[k]["delta_refusal+emotion_minus_refusal"] for k in mon}
wr = m_main["wrappers"]
w_emo = wr["emotional_pressure"]["prompt_projection_shift_vs_neutral"]
win = m_pair["h1"]["within_prompt"]["tokens_1-5"]
loto = p1["heldout"]["loto_acc_denoised"]
best_l = max(range(7, 28), key=lambda l: loto[l])
fa0, fa5 = k0["false_alarms"], k5["false_alarms"]
k_list = [int(k) for k in mon]
k_low = min(mon[str(k)][n]["test_auc"] for k in k_list if k > 0 for n in ("emotion", "refusal", "refusal+emotion"))
k_high = max(mon[str(k)][n]["test_auc"] for k in k_list if k > 0 for n in ("emotion", "refusal", "refusal+emotion"))
rnd_meds = [mon[str(k)]["random"]["median"] for k in k_list if k > 0]

delta_rows = "\n".join(
    f"<tr><td>{k}</td><td>{delta[k]['point']:+.2f}</td><td>[{delta[k]['ci95'][0]:+.2f}, {delta[k]['ci95'][1]:+.2f}]</td>"
    f"<td>{delta[k]['p_two_sided']:.2f}</td></tr>" for k in mon)

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
refs_html = "\n".join(f"<li>{r}</li>" for r in refs)

FIG1 = """
<svg viewBox="0 0 920 872" xmlns="http://www.w3.org/2000/svg" font-family="Seravek, 'Avenir Next', 'Gill Sans', sans-serif" role="img" aria-labelledby="fig1t">
  <title id="fig1t">Method: request, model activations, projection on directions, logistic monitor, judge labels</title>
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#16202c"/></marker>
  </defs>
  <g font-size="28" fill="#16202c">
    <circle cx="30" cy="58" r="25" fill="#16202c"/><text x="30" y="67" fill="#fff" text-anchor="middle" font-size="27" font-weight="600">1</text>
    <rect x="76" y="4" width="840" height="108" rx="14" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="100" y="48" font-weight="600" font-size="31">Request from JailbreakBench</text>
    <text x="100" y="90" fill="#4d5868">harmful or benign; fixed wrapper or published PAIR prompt</text>
    <line x1="496" y1="114" x2="496" y2="152" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>

    <circle cx="30" cy="214" r="25" fill="#16202c"/><text x="30" y="223" fill="#fff" text-anchor="middle" font-size="27" font-weight="600">2</text>
    <rect x="76" y="160" width="840" height="108" rx="14" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="100" y="204" font-weight="600" font-size="31">Qwen2.5-Omni-7B reads it and answers</text>
    <text x="100" y="246" fill="#4d5868">activations of 28 layers: end of prompt + first 20 tokens</text>
    <line x1="496" y1="270" x2="496" y2="308" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>

    <circle cx="30" cy="370" r="25" fill="#16202c"/><text x="30" y="379" fill="#fff" text-anchor="middle" font-size="27" font-weight="600">3</text>
    <rect x="76" y="316" width="840" height="236" rx="14" fill="#fff" stroke="#16202c" stroke-width="2.5"/>
    <text x="100" y="360" font-weight="600" font-size="31">Project onto fixed directions</text>
    <rect x="100" y="380" width="250" height="74" rx="10" fill="#fde9df"/><rect x="100" y="380" width="10" height="74" fill="#f47230"/>
    <text x="124" y="412" font-size="26">5 emotion</text><text x="124" y="442" font-size="26">vectors</text>
    <rect x="372" y="380" width="250" height="74" rx="10" fill="#d9f4ee"/><rect x="372" y="380" width="10" height="74" fill="#00b894"/>
    <text x="396" y="412" font-size="26">refusal</text><text x="396" y="442" font-size="26">direction [2]</text>
    <rect x="644" y="380" width="250" height="74" rx="10" fill="#efefef"/><rect x="644" y="380" width="10" height="74" fill="#9a9a9a"/>
    <text x="668" y="412" font-size="26">200 random</text><text x="668" y="442" font-size="26">directions</text>
    <text x="100" y="494" fill="#4d5868" font-size="26">emotion vectors: 150 stories the model wrote, emotion</text>
    <text x="100" y="527" fill="#4d5868" font-size="26">felt but never named; mean(emotion) &#8722; mean(all) [1]</text>
    <line x1="496" y1="554" x2="496" y2="592" stroke="#16202c" stroke-width="2.5" marker-end="url(#ah)"/>

    <circle cx="30" cy="654" r="25" fill="#16202c"/><text x="30" y="663" fill="#fff" text-anchor="middle" font-size="27" font-weight="600">4</text>
    <rect x="76" y="600" width="840" height="108" rx="14" fill="#fff" stroke="#16202c" stroke-width="3.5"/>
    <text x="100" y="644" font-weight="600" font-size="31">Logistic monitor: P(model will comply)</text>
    <text x="100" y="686" fill="#4d5868">fit on 50 % of behaviours, layer picked on 20 %, tested on 30 %</text>

    <circle cx="30" cy="814" r="25" fill="#16202c"/><text x="30" y="823" fill="#fff" text-anchor="middle" font-size="27" font-weight="600">5</text>
    <rect x="76" y="760" width="840" height="108" rx="14" fill="#fff" stroke="#16202c" stroke-width="2.5" stroke-dasharray="10 7"/>
    <text x="100" y="804" font-weight="600" font-size="31">HarmBench judge [8] labels each answer</text>
    <text x="100" y="846" fill="#4d5868">complied or refused: the target the monitor must predict</text>
    <line x1="496" y1="758" x2="496" y2="716" stroke="#16202c" stroke-width="2.5" stroke-dasharray="8 6" marker-end="url(#ah)"/>
  </g>
</svg>
"""

v = dict(
    fig1=FIG1,
    loto_best=a2(loto[best_l]), loto_layer=best_l,
    loto_ci=f"[{a2(p1['heldout']['loto_ci95_denoised'][best_l][0])}, {a2(p1['heldout']['loto_ci95_denoised'][best_l][1])}]",
    main_rate=pct(s_main["harmful_compliance_overall"]["rate"]), main_n=s_main["harmful_compliance_overall"]["n"],
    main_k=s_main["harmful_compliance_overall"]["k"],
    pair_rate=pct(s_pair["harmful_compliance_overall"]["rate"]), pair_n=s_pair["harmful_compliance_overall"]["n"],
    k0_lo=a2(min(k0[n]["test_auc"] for n in ("emotion", "refusal", "refusal+emotion"))),
    k0_hi=a2(max(k0[n]["test_auc"] for n in ("emotion", "refusal", "refusal+emotion"))),
    k_low=a2(k_low), k_high=a2(k_high), rnd_lo=a2(min(rnd_meds)), rnd_hi=a2(max(rnd_meds)),
    k5_emo=a2(k5["emotion"]["test_auc"]), k5_rnd=a2(k5["random"]["median"]), k5_rnd_hi=a2(k5["random"]["p2.5_p97.5"][1]),
    k5_pct=f"{k5['random']['emotion_percentile']:.0f}",
    n_test=m_pair["n"]["harmful_test"], test_comp=round(m_pair["n"]["harmful_test"] * m_pair["test_compliance_rate"]),
    n_test_beh=m_pair["n"]["harmful_test"] // 10,
    dz_desp=f"{win['desperate']['d_z']:+.2f}", dz_calm=f"{win['calm']['d_z']:+.2f}", dz_rnd=a2(win["random_abs_dz_p95"]),
    n_mixed=win["desperate"]["n_prompts"],
    shift_desp=f"{w_emo['desperate']['in_sd_units']:+.2f}", shift_calm=f"{w_emo['calm']['in_sd_units']:+.2f}",
    neu_comp=pct(wr["neutral"]["harmful_compliance"]), emo_comp=pct(wr["emotional_pressure"]["harmful_compliance"]),
    neu_over=pct(wr["neutral"]["benign_over_refusal"], 0), emo_over=pct(wr["emotional_pressure"]["benign_over_refusal"], 0),
    delta_rows=delta_rows,
    d_min=f"{min(d['point'] for d in delta.values()):+.2f}", d_max=f"{max(d['point'] for d in delta.values()):+.2f}",
    fa0_ref_benign=pct(fa0["refusal"]["alarm_rate_on_benign"]["rate"], 0),
    fa5_emo_benign=pct(fa5["emotion"]["alarm_rate_on_benign"]["rate"], 0),
    fa5_emo_recall=pct(fa5["emotion"]["test_recall"]["rate"], 0),
    kappa_main=a2(jv["llm_main_all"]["kappa_in_sample"]), kappa_pair=a2(jv["llm_pair_all"]["kappa_in_sample"]),
    agree_main=round(jv["llm_main_all"]["raw_agreement"] * 100), agree_pair=round(jv["llm_pair_all"]["raw_agreement"] * 100),
    refs=refs_html,
    n_comp_h1=m_pair['h1']['trajectories']['desperate']['compliant']['n'],
    n_ref_h1=m_pair['h1']['trajectories']['desperate']['refused']['n'],
)

TEMPLATE = Template((HERE / "poster_template.html").read_text())
(HERE / "poster.html").write_text(TEMPLATE.substitute(v))
print("wrote poster/poster.html")
