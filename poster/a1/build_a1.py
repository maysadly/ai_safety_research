"""Build the A1 landscape poster (poster/a1/poster_a1.html) from the results JSON files.
Every number on the poster is read here; nothing is typed by hand in the template.

Usage: python scripts/13_a1_figures.py && python poster/a1/build_a1.py && python poster/a1/export_a1.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / "results"
J = lambda name: json.load(open(R / name))  # noqa: E731

p1q, p1m = J("phase1_vectors.json"), J("phase5_mistral_phase1_vectors.json")
s_main, s_pair = J("phase2_summary_main.json"), J("phase2_summary_pair.json")
m_pair = J("phase3_metrics_pair.json")
jv = J("judge_validation.json")
M = {m: J(f"phase5_metrics_{m}.json") for m in ("qwen", "mistral")}
PH = {m: J(f"phase5_posthoc_k0_{m}.json") for m in ("qwen", "mistral")}
S5 = {m: J(f"phase2_summary_p5_{m}.json") for m in ("qwen", "mistral")}
ST = {m: J(f"phase6_steer_{m}.json") for m in ("qwen", "mistral")}


def pct(x, d=0):
    return f"{x * 100:.{d}f}"


def a2(x):
    return f"{x:.2f}"


def fam_rates(tag):
    lab = list(csv.DictReader(open(ROOT / f"data/labels_{tag}.csv")))
    harm = defaultdict(list)
    for l in lab:
        if l["kind"] != "harmful":
            continue
        w = l["wrapper"]
        k = "pair" if w.startswith("pair") else "prs" if w.startswith("prs") else w
        harm[k].append(int(l["compliance"]))
    return {k: sum(v) / len(v) for k, v in harm.items()}


FR = {m: fam_rates(f"p5_{m}") for m in ("qwen", "mistral")}


def ok(b):
    return '<span class="ok">&#10003;</span>' if b else '<span class="no">&#10007;</span>'


q, mi = M["qwen"], M["mistral"]
c1q, c1m = q["C1_added_value"], mi["C1_added_value"]
c2q, c2m = q["C2_emotion_vs_random"], mi["C2_emotion_vs_random"]
c4q, c4m = q["C4_within_prompt"], mi["C4_within_prompt"]
c5q, c5m = q["C5_wrappers"], mi["C5_wrappers"]
rng = lambda c, d: f"+{min(v['point'] for v in c.values()):.{d}f}&#8211;{max(v['point'] for v in c.values()):.{d}f}"  # noqa: E731
eq = lambda c: all(v["equivalent_within_margin"] for v in c.values())  # noqa: E731
c5_ok = lambda c: c["benign_over_refusal"]["emotional_pressure"]["ci95"][0] > 0 and c["harmful_compliance"]["emotional_pressure"]["ci95"][0] <= 0  # noqa: E731
rows_c = [
    ("C1", "Emotion adds &#8804; 0.03 AUROC to refusal (H2)",
     f"{ok(eq(c1q))} &#916; {rng(c1q, 3)}", f"{ok(eq(c1m))} &#916; {rng(c1m, 2)}"),
    ("C2", "Emotion no better than random directions",
     f"{ok(not c2q['0']['above_random_p95'])} k0: {c2q['0']['percentile']:.0f}th; {ok(not c2q['5']['above_random_p95'])} k1&#8211;5: {c2q['1']['percentile']:.0f}&#8211;{c2q['5']['percentile']:.0f}th",
     f"{ok(not c2m['0']['above_random_p95'])} k0: {c2m['0']['percentile']:.0f}th; {ok(not c2m['5']['above_random_p95'])} k1&#8211;5: {c2m['1']['percentile']:.0f}&#8211;{c2m['5']['percentile']:.0f}th"),
    ("C3", "No warning before generation (AUROC &lt; 0.65)",
     f"{ok(q['C3_pre_generation']['upper_below_bound'])} {a2(q['C3_pre_generation']['emotion_auc_k0'])}",
     f"{ok(mi['C3_pre_generation']['upper_below_bound'])} {a2(mi['C3_pre_generation']['emotion_auc_k0'])}"),
    ("C4", "Refusals more desperate, less calm (H1 reversed)",
     f"{ok(c4q['confirmed'])} d<sub>z</sub> {c4q['desperate']['d_z']:+.2f} / {c4q['calm']['d_z']:+.2f}",
     f"{ok(c4m['confirmed'])} d<sub>z</sub> {c4m['desperate']['d_z']:+.2f} / {c4m['calm']['d_z']:+.2f}"),
    ("C5", "Pressure: over-refusal up, compliance not up (H3)",
     f"{ok(c5_ok(c5q))} +{pct(c5q['benign_over_refusal']['emotional_pressure']['diff_vs_neutral'])} / {c5q['harmful_compliance']['emotional_pressure']['diff_vs_neutral'] * 100:+.1f} pp",
     f"{ok(c5_ok(c5m))} +{pct(c5m['benign_over_refusal']['emotional_pressure']['diff_vs_neutral'])} / {c5m['harmful_compliance']['emotional_pressure']['diff_vs_neutral'] * 100:+.0f} pp"),
]
pp = lambda x: f"{x * 100:+.1f}"  # noqa: E731
sq, sm = ST["qwen"], ST["mistral"]
rows_c += [
    ("S1", "&#8722;desperate / +calm cut distressed over-refusal (E4)",
     f"{ok(all(v['supported'] for v in sq['S1_mitigation'].values()))} {pp(sq['S1_mitigation']['desp_minus_12']['diff'])} / {pp(sq['S1_mitigation']['calm_plus_12']['diff'])} pp",
     f"{ok(all(v['supported'] for v in sm['S1_mitigation'].values()))} {pp(sm['S1_mitigation']['desp_minus_12']['diff'])} / {pp(sm['S1_mitigation']['calm_plus_12']['diff'])} pp"),
    ("S2", "&#8230;without raising harmful compliance (&lt; +3 pp)",
     f"{ok(sq['S2_safety_cost']['desp_minus_12']['non_inferior_3pp'])} {pp(sq['S2_safety_cost']['desp_minus_12']['diff'])}; {ok(sq['S2_safety_cost']['calm_plus_12']['non_inferior_3pp'])} +calm {pp(sq['S2_safety_cost']['calm_plus_12']['diff'])} pp",
     f"{ok(sm['S2_safety_cost']['desp_minus_12']['non_inferior_3pp'] and sm['S2_safety_cost']['calm_plus_12']['non_inferior_3pp'])} {pp(sm['S2_safety_cost']['desp_minus_12']['diff'])} / {pp(sm['S2_safety_cost']['calm_plus_12']['diff'])} pp"),
    ("S4", "&#8230;beyond a random direction",
     f"{ok(all(v['p_less'] < 0.05 for v in sq['S4_vs_random'].values()))} {pp(sq['S4_vs_random']['desp_minus_12']['diff_vs_random'])} / {pp(sq['S4_vs_random']['calm_plus_12']['diff_vs_random'])} pp",
     f"{ok(all(v['p_less'] < 0.05 for v in sm['S4_vs_random'].values()))} {pp(sm['S4_vs_random']['desp_minus_12']['diff_vs_random'])} / {pp(sm['S4_vs_random']['calm_plus_12']['diff_vs_random'])} pp"),
]
table_c = "\n".join(f"<tr><th>{a}</th><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows_c)
rate = lambda m, c, k: ST[m]["rates"][c][k]["rate"]  # noqa: E731
gap = lambda m, c: rate(m, c, "over_refusal:emotional_pressure") - rate(m, c, "over_refusal:neutral")  # noqa: E731

wfq, wfm = PH["qwen"]["within_family"], PH["mistral"]["within_family"]
direct_q = max(wfq["D1 plain + wrappers"]["refusal"], wfq["HarmBench direct"]["refusal"])
pair_q = wfq["PAIR"]["refusal"]

refs = [
    'N. Sofroniew et al., "Emotion concepts and their function in a large language model," <i>Transformer Circuits Thread</i>, Anthropic, 2026.',
    'A. Arditi et al., "Refusal in language models is mediated by a single direction," in <i>Proc. NeurIPS</i>, 2024.',
    'P. Chao et al., "JailbreakBench: An open robustness benchmark for jailbreaking LLMs," in <i>NeurIPS Datasets &amp; Benchmarks</i>, 2024.',
    'A. Korznikov et al., "The rogue scalpel: Activation steering compromises LLM safety," arXiv:2509.22067, 2025.',
    'Y. Wang et al., "Read-best is not steer-best: A probing&#8211;steering layer dissociation in omni-modal LLMs," arXiv:2609.22135, 2026.',
    'J. Xu et al., "Qwen2.5-Omni technical report," arXiv:2503.20215, 2025.',
    'A. Goldenberg and J. J. Gross, "Do large language models have emotions?" arXiv:2606.14742, 2026.',
    'M. Mazeika et al., "HarmBench: A standardized evaluation framework for automated red teaming and robust refusal," in <i>Proc. ICML</i>, 2024.',
    'P. Chao et al., "Jailbreaking black box large language models in twenty queries," in <i>Proc. IEEE SaTML</i>, 2025.',
    'A. Q. Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023.',
    'M. Andriushchenko, F. Croce, and N. Flammarion, "Jailbreaking leading safety-aligned LLMs with simple adaptive attacks," arXiv:2404.02151, 2024.',
    'A. Zou et al., "Representation engineering: A top-down approach to AI transparency," arXiv:2310.01405, 2023.',
]

FIG1 = """
<svg viewBox="0 0 700 710" xmlns="http://www.w3.org/2000/svg" font-family="Seravek, 'Avenir Next', 'Gill Sans', sans-serif" role="img" aria-labelledby="f1t">
  <title id="f1t">System architecture: request, model, projections, monitor, judge</title>
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#16202c"/></marker></defs>
  <g font-size="22" fill="#16202c">
    <circle cx="20" cy="48" r="18" fill="#16202c"/><text x="20" y="55" fill="#fff" text-anchor="middle" font-size="20" font-weight="600">1</text>
    <rect x="48" y="4" width="648" height="92" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.2"/>
    <text x="66" y="38" font-weight="600" font-size="24">Request (JailbreakBench, HarmBench)</text>
    <text x="66" y="74" fill="#4d5868">plain, wrapped or published jailbreak; benign controls</text>
    <line x1="372" y1="98" x2="372" y2="120" stroke="#16202c" stroke-width="2.2" marker-end="url(#ah)"/>

    <circle cx="20" cy="170" r="18" fill="#16202c"/><text x="20" y="177" fill="#fff" text-anchor="middle" font-size="20" font-weight="600">2</text>
    <rect x="48" y="126" width="648" height="92" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.2"/>
    <text x="66" y="160" font-weight="600" font-size="24">LLM reads it and answers</text>
    <text x="66" y="196" fill="#4d5868">Qwen2.5-Omni-7B or Mistral-7B; activations, tokens 0&#8211;20</text>
    <line x1="372" y1="220" x2="372" y2="242" stroke="#16202c" stroke-width="2.2" marker-end="url(#ah)"/>

    <circle cx="20" cy="292" r="18" fill="#16202c"/><text x="20" y="299" fill="#fff" text-anchor="middle" font-size="20" font-weight="600">3</text>
    <rect x="48" y="248" width="648" height="196" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.2"/>
    <text x="66" y="282" font-weight="600" font-size="24">Project onto fixed directions</text>
    <rect x="66" y="300" width="196" height="64" rx="8" fill="#fde9df"/><rect x="66" y="300" width="8" height="64" fill="#f47230"/>
    <text x="84" y="327" font-size="21">5 emotion</text><text x="84" y="353" font-size="21">vectors [1]</text>
    <rect x="274" y="300" width="196" height="64" rx="8" fill="#d9f4ee"/><rect x="274" y="300" width="8" height="64" fill="#00b894"/>
    <text x="292" y="327" font-size="21">refusal</text><text x="292" y="353" font-size="21">direction [2]</text>
    <rect x="482" y="300" width="196" height="64" rx="8" fill="#efefef"/><rect x="482" y="300" width="8" height="64" fill="#9a9a9a"/>
    <text x="500" y="327" font-size="21">200 random</text><text x="500" y="353" font-size="21">directions</text>
    <text x="66" y="394" fill="#4d5868" font-size="21">emotion vectors: model-written stories, emotion never named</text>
    <text x="66" y="424" fill="#4d5868" font-size="21">baseline without activations: text of the first k tokens</text>
    <line x1="372" y1="446" x2="372" y2="468" stroke="#16202c" stroke-width="2.2" marker-end="url(#ah)"/>

    <circle cx="20" cy="518" r="18" fill="#16202c"/><text x="20" y="525" fill="#fff" text-anchor="middle" font-size="20" font-weight="600">4</text>
    <rect x="48" y="474" width="648" height="92" rx="12" fill="#fff" stroke="#16202c" stroke-width="3.2"/>
    <text x="66" y="508" font-weight="600" font-size="24">Logistic monitor &#8594; P(model will comply)</text>
    <text x="66" y="544" fill="#4d5868">alarm above a threshold &#8594; human review</text>

    <circle cx="20" cy="660" r="18" fill="#16202c"/><text x="20" y="667" fill="#fff" text-anchor="middle" font-size="20" font-weight="600">5</text>
    <rect x="48" y="616" width="648" height="90" rx="12" fill="#fff" stroke="#16202c" stroke-width="2.2" stroke-dasharray="9 6"/>
    <text x="66" y="650" font-weight="600" font-size="24">HarmBench judge [8] labels every answer</text>
    <text x="66" y="686" fill="#4d5868">complied / refused: the target the monitor must predict</text>
    <line x1="372" y1="614" x2="372" y2="574" stroke="#16202c" stroke-width="2.2" stroke-dasharray="7 5" marker-end="url(#ah)"/>
  </g>
</svg>
"""

v = dict(
    fig1=FIG1, table_c=table_c, refs="\n".join(f"<li>{r}</li>" for r in refs),
    s1q=f"{-sq['S1_mitigation']['desp_minus_12']['diff'] * 100:.1f}", s1m=f"{-sm['S1_mitigation']['desp_minus_12']['diff'] * 100:.1f}",
    q_or_base=pct(rate("qwen", "base", "over_refusal:emotional_pressure")), q_or_desp=pct(rate("qwen", "desp_minus_12", "over_refusal:emotional_pressure")),
    m_or_base=pct(rate("mistral", "base", "over_refusal:emotional_pressure")), m_or_desp=pct(rate("mistral", "desp_minus_12", "over_refusal:emotional_pressure")),
    q_or_rand=pct(rate("qwen", "rand_minus_12", "over_refusal:emotional_pressure")), m_or_rand=pct(rate("mistral", "rand_minus_12", "over_refusal:emotional_pressure")),
    q_or_d4=pct(rate("qwen", "desp_minus_4", "over_refusal:emotional_pressure")), m_or_d4=pct(rate("mistral", "desp_minus_4", "over_refusal:emotional_pressure")),
    m_or_dplus=pct(rate("mistral", "desp_plus_12", "over_refusal:emotional_pressure")),
    q_gap_closed=pct(1 - gap("qwen", "desp_minus_12") / gap("qwen", "base")), m_gap_closed=pct(1 - gap("mistral", "desp_minus_12") / gap("mistral", "base")),
    q_harm_d=pp(sq["S2_safety_cost"]["desp_minus_12"]["diff"]), q_harm_ci=f"[{sq['S2_safety_cost']['desp_minus_12']['ci95'][0] * 100:+.1f}, {sq['S2_safety_cost']['desp_minus_12']['ci95'][1] * 100:+.1f}]",
    q_calm_harm=pp(sq["S2_safety_cost"]["calm_plus_12"]["diff"]),
    m_rand_harm=pp(sm["descriptive"]["harmful_vs_base"]["rand_minus_12"]["diff"]),
    n6=f"{sum(len(open(ROOT / f'data/raw/p6_{m}/generations.jsonl').readlines()) for m in ('qwen', 'mistral')):,}",
    n_all=f"{q['n']['harmful'] + mi['n']['harmful'] + 3000 + sum(len(open(ROOT / f'data/raw/p6_{m}/generations.jsonl').readlines()) for m in ('qwen', 'mistral')):,}",
    loto_q=a2(max(p1q["heldout"]["loto_acc_denoised"][7:28])), loto_m=a2(max(p1m["heldout"]["loto_acc_denoised"][8:32])),
    e1_rate=pct(s_main["harmful_compliance_overall"]["rate"], 1), e1_k=s_main["harmful_compliance_overall"]["k"],
    e1_n=f"{s_main['harmful_compliance_overall']['n']:,}",
    e2_rate=pct(s_pair["harmful_compliance_overall"]["rate"], 0), e2_n=s_pair["harmful_compliance_overall"]["n"],
    e2_k0=a2(m_pair["monitors"]["0"]["emotion"]["test_auc"]), e2_k5=a2(m_pair["monitors"]["5"]["emotion"]["test_auc"]),
    e2_k5_rnd=a2(m_pair["monitors"]["5"]["random"]["median"]),
    n5=f"{q['n']['harmful'] + mi['n']['harmful'] + 2 * 1500:,}", beh=q["n"]["behaviours"],
    q_comp=f"{q['n']['compliant']:,}", q_harm=f"{q['n']['harmful']:,}", m_comp=f"{mi['n']['compliant']:,}",
    q_rate=pct(q["n"]["compliant"] / q["n"]["harmful"], 1), m_rate=pct(mi["n"]["compliant"] / mi["n"]["harmful"], 0),
    q_mixed=S5["qwen"]["harmful_prompts_mixed"], m_mixed=S5["mistral"]["harmful_prompts_mixed"],
    c1q_max=f"{max(v['point'] for v in c1q.values()):.3f}",
    q_k1_ref=a2(q["monitors"]["1"]["refusal"]["auc"]), q_k1_emo=a2(q["monitors"]["1"]["emotion"]["auc"]),
    q_k1_rnd=a2(q["monitors"]["1"]["random"]["median"]), q_k1_txt=a2(q["monitors"]["1"]["text_tfidf"]["auc"]),
    q_k5_txt=f'{q["monitors"]["5"]["text_tfidf"]["auc"]:.3f}', q_k5_both=f'{q["monitors"]["5"]["refusal+emotion"]["auc"]:.3f}',
    pair_q_emo=a2(wfq["PAIR"]["emotion"]),
    q_k0_ref=a2(q["monitors"]["0"]["refusal"]["auc"]), m_k0_emo=a2(mi["monitors"]["0"]["emotion"]["auc"]),
    direct_q=a2(direct_q), pair_q=a2(pair_q), fam_only_q=a2(PH["qwen"]["pooled"]["family_only"]),
    m_ref_k1=a2(mi["monitors"]["1"]["refusal"]["auc"]), m_emo_k1=a2(mi["monitors"]["1"]["emotion"]["auc"]),
    m_rnd_k1=a2(mi["monitors"]["1"]["random"]["median"]),
    q_plain=pct(FR["qwen"]["neutral"]), q_pair=pct(FR["qwen"]["pair"]), m_plain=pct(FR["mistral"]["neutral"]),
    m_prs=pct(FR["mistral"]["prs"]), m_role=pct(FR["mistral"]["roleplay"]),
    q_over=pct(c5q["benign_over_refusal"]["emotional_pressure"]["diff_vs_neutral"]),
    m_over=pct(c5m["benign_over_refusal"]["emotional_pressure"]["diff_vs_neutral"]),
    m_comp_drop=pct(-c5m["harmful_compliance"]["emotional_pressure"]["diff_vs_neutral"]),
    kappa_main=a2(jv["llm_main_all"]["kappa_in_sample"]), kappa_pair=a2(jv["llm_pair_all"]["kappa_in_sample"]),
    fa5=pct(m_pair["monitors"]["5"]["false_alarms"]["emotion"]["alarm_rate_on_benign"]["rate"]),
    fa0=pct(m_pair["monitors"]["0"]["false_alarms"]["refusal"]["alarm_rate_on_benign"]["rate"]),
)
html = Template((HERE / "template_a1.html").read_text()).substitute(v)
(HERE / "poster_a1.html").write_text(html)
print("wrote poster/a1/poster_a1.html")
