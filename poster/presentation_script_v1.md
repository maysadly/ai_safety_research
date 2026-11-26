# Poster defence script for poster v1 (A0 portrait, `poster/poster.pdf`), 7 min + 3 min questions

This script matches the **first poster**: Qwen2.5-Omni-7B only, Phases 1–3 (pre-registered set plus the exploratory PAIR set). Figure numbers are the ones printed on that poster. It is about 900 words, roughly 7 minutes at 130 words per minute.

| Time | Speaker | Part (rubric) |
|---|---|---|
| 0:00–1:00 | **Danat Bakzhanov** | Problem and research gap ("Why look at emotions?", hypotheses) |
| 1:00–2:30 | **Yerkin Maratuly** | Architecture and methodology (Fig. 1 Method) |
| 2:30–4:00 | **Tleukhan Makhmutov** | Experiments and results (Fig. 4 top band, Fig. 2, Fig. 3) |
| 4:00–6:00 | **Kasymzhomart Beketov** | Fig. 5, monitoring, syllabus integration, safety improvements |
| 6:00–7:00 | **Danat Bakzhanov** | Conclusions, contribution and limitations |

---

## 1. Danat: problem and research gap (1:00)

> Good afternoon. We are Danat, Yerkin, Kasymzhomart and Tleukhan. Our question is on the title: can a model's internal emotion signals warn us *before* it breaks its safety rules?
>
> Language models now act as agents, and output filters only catch harm after it is written. We want a monitor that raises a flag before the answer appears.
>
> This year Anthropic showed that a language model has emotion directions, for states like "desperate" or "calm", that causally change its behaviour, and suggested monitoring them [1]. Arditi et al. showed that refusal is controlled by a single direction [2].
>
> The gap: nobody has tested whether emotion signals predict harmful compliance, or add anything beyond the refusal direction. Our three hypotheses are on the left: H1, compliant answers start more desperate; H2, emotion adds to refusal; H3, emotional pressure raises compliance.
>
> Yerkin will show how we tested them.

## 2. Yerkin: architecture and methodology (1:30), Fig. 1

> Figure 1 shows our pipeline in five steps.
>
> One: a request from JailbreakBench, either with a fixed wrapper or as a published PAIR jailbreak prompt. We created no new attacks.
>
> Two: Qwen2.5-Omni-7B reads it and answers. We record its activations at the end of the prompt and at each of the first 20 tokens.
>
> Three: we project them onto fixed directions. There are five emotion vectors, built from 150 stories the model wrote with the emotion never named; they recognise the emotion of unseen stories with 65 percent accuracy, against 20 percent chance. There is also the refusal direction and, as a control, 200 random directions.
>
> Four: a logistic monitor predicts whether the model will comply.
>
> Five: the HarmBench judge labels each answer. An independent review agreed on 99 of 100 plain answers and 88 of 100 jailbreak answers.
>
> We ran two sets. The pre-registered set was 100 harmful and 100 benign behaviours, each with a neutral, an emotional-pressure and a role-play version, five samples each: 3,000 answers. Only 0.9 percent complied, too few to test a monitor. So we added an exploratory set of JailbreakBench's published PAIR prompts: 790 answers, 19.8 percent of which complied.
>
> Tleukhan, what did we find?

## 3. Tleukhan: experiments and results (1:30), Fig. 4 top band, Fig. 2, Fig. 3

> Start with the band at the top, Figure 4. At the end of the prompt, answers that will comply and answers that will refuse look the same on the "desperate" and "calm" vectors. They split within one to three tokens. Refusals turn desperate and lose calm, the opposite of H1. Within the same prompt, the effect is about 1.2 standard deviations, above the random-direction band of 0.94.
>
> Figure 2 is the main test. Before the model writes anything, every monitor is at chance: AUROC 0.42 to 0.55. After five tokens, the emotion monitor reaches 0.87. But that is only the 94th percentile of 200 random monitors, whose median is 0.81 and which reach up to 0.88.
>
> Figure 3 shows the same thing over time: from the first generated token compliance becomes readable, by every direction we tried, random ones included.
>
> And adding emotion to the refusal direction changes AUROC by −0.10 to +0.14 across token counts. None of these differences is significant after Bonferroni correction. So H2 is not supported.
>
> In the pre-registered set, the test split held only one compliant answer, so its AUROCs cannot be estimated. We say so on the poster.
>
> Kasymzhomart will explain what this means for safety.

## 4. Kasymzhomart: Fig. 5, monitoring, syllabus integration, safety improvements (2:00)

**H3 and safety evaluation (about 35 s), Fig. 5**

> Figure 5 tests H3. Emotional pressure moves the internal state as intended: "desperate" up and "calm" down, by about 0.9 standard deviations. But harmful compliance stays at 0.8 percent, while refusals of *harmless* requests rise from 29 to 49 percent. The model becomes more careful. PAIR jailbreaks, by contrast, raise compliance to about 20 percent.

**Monitoring (about 30 s)**

> Look at what a monitor would do in practice. If we tune the refusal-direction monitor to catch 80 percent of compliant answers before generation, it also flags 92 percent of benign requests. After five tokens the emotion monitor flags 29 percent of benign answers. It detects the model *going along*, not harm. So a real deployment needs a classifier on the request, an early-token monitor, and human escalation, with alarms logged.

**Syllabus integration (about 35 s)**

> The project covers most of the course.
> - Week 1, risk and governance.
> - Week 2, alignment: the refusal direction.
> - Week 3, data provenance: public benchmarks and a validated judge.
> - Weeks 4 and 5, adversarial robustness and red teaming: published PAIR jailbreaks.
> - Week 6, interpretability: activation directions.
> - Week 9, evaluation: AUROC, confidence intervals, false alarms.
> - Week 10, oversight: an automatic judge plus a human check.
> - Weeks 7 and 8 we touch only indirectly, through over-refusal and agents.

**Safety improvements (about 20 s)**

> - Do not rely on emotion vectors as a pre-generation alarm.
> - Watch the first tokens of the answer, but pair that with a check on the request.
> - Treat over-refusal under emotional pressure as a usability risk.
>
> Danat will close.

## 5. Danat: conclusions, contribution and limitations (1:00)

> Our conclusions are at the bottom of the poster.
> - Emotion vectors gave no pre-generation warning in Qwen2.5-Omni-7B, and added nothing robust to the refusal direction.
> - Early-token monitors work, but they read what the model is already saying; any direction does.
> - Emotional pressure is a usability risk here, not a jailbreak.
>
> Our contribution is the first test of emotion vectors as a safety early-warning monitor, against refusal-direction and random-direction baselines, with an open pipeline behind the QR code.
>
> Limitations:
> - one 7B model, text only, answers cut at 128 tokens;
> - an automatic judge, with human validation pending;
> - the decisive test is exploratory, with 26 test behaviours and confidence intervals of about ±0.1;
> - fear and desperate overlap.
>
> Thank you, we are happy to take questions.

---

## Questions: who answers what

| Topic | Answers | Short answer |
|---|---|---|
| Why emotions? | Danat | Anthropic showed causal emotion directions and proposed monitoring them [1]; we tested that idea on safety. |
| What is AUROC? | Tleukhan | The chance that a randomly chosen compliant answer scores higher than a refused one. 0.5 is chance, 1 is perfect. |
| Why random directions? | Tleukhan | A control. If random directions do as well as emotion vectors, emotional meaning is not what predicts. |
| Why only 0.9 % compliance? | Yerkin | The model is safety-tuned and the wrappers were simple and fixed by design. That is why we added published PAIR prompts, labelled exploratory. |
| Is the PAIR result reliable? | Tleukhan | It is exploratory: 26 test behaviours and wide CIs. (We have since run a pre-registered confirmatory study with 2 models and 282 behaviours. It confirmed that emotion adds little in Qwen, but showed a pre-generation signal for *direct* requests. That is on the updated poster.) |
| How reliable are the labels? | Yerkin | HarmBench classifier; agreement with an independent review of 99/100 and 88/100. Human labels are in progress. |
| Ethics? | Yerkin | Only published benchmark prompts; no model answers to harmful requests are shown or committed. |
| Deployment? | Kasymzhomart | Request classifier + early-token monitor + human escalation and logging. |
