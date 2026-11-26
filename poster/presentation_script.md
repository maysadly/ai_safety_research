# Poster defence script (7 min talk + 3 min questions)

Speaking pace about 130 words per minute. Figure numbers refer to the poster. Each speaker hands over with one short line, so the transitions sound natural.

| Time | Speaker | Part (rubric) |
|---|---|---|
| 0:00–1:00 | **Danat Bakzhanov** | Problem and research gap |
| 1:00–2:30 | **Yerkin Maratuly** | Architecture and methodology, baselines |
| 2:30–4:00 | **Tleukhan Makhmutov** | Experiments and results |
| 4:00–6:00 | **Kasymzhomart Beketov** | Syllabus integration and monitoring; safety improvements |
| 6:00–7:00 | **Danat Bakzhanov** | Scientific contribution and limitations |

---

## 1. Danat: problem and research gap (1:00)

> Good afternoon. We are Danat, Yerkin, Kasymzhomart and Tleukhan. Our question is on the title: can a model's internal emotion signals warn us *before* it breaks its safety rules?
>
> Language models now act as agents, and output filters only catch harm after it is written. We want a monitor that raises a flag before the answer appears.
>
> This year Anthropic showed that a language model has emotion directions, for states like "desperate" or "calm", that causally change its behaviour, and suggested monitoring them [1]. Arditi et al. showed that refusal is controlled by a single direction [2].
>
> The gap: nobody has tested whether emotion signals predict harmful compliance, or whether they add anything beyond the refusal direction. Our objective is exactly that, written as three hypotheses: a signal, added value, and the effect of emotional pressure.
>
> Yerkin will show how we tested it.

## 2. Yerkin: architecture and methodology (1:30), Fig. 1

> Our system has five steps, shown in Figure 1.
>
> One: a request from public benchmarks only, JailbreakBench and HarmBench. We created no new attacks.
>
> Two: the model reads it and starts answering. That is Qwen2.5-Omni-7B, plus Mistral-7B in the confirmatory study. We record its activations at the end of the prompt and at each of the first 20 generated tokens.
>
> Three: we project these activations onto fixed directions. There are five emotion vectors, built from 150 stories the model wrote in which an emotion is felt but never named; on unseen stories they recognise the emotion 65 to 79 percent of the time, against 20 percent chance. There is also the refusal direction and, as a control, 200 random directions.
>
> Four: a logistic monitor turns these numbers into the probability that the model will comply.
>
> Five: the HarmBench judge labels each answer as complied or refused, and an independent review agreed on 99 of 100 answers.
>
> Our baselines are the refusal direction, random directions, and simply reading the first words of the answer. Everything is tested on behaviours the monitor never saw, with 95 percent confidence intervals.
>
> Tleukhan, what did we find?

## 3. Tleukhan: experiments and results (1:30), Fig. 2–5 and the results table

> First, with plain requests Qwen almost never complies: only 0.9 percent of 1,500 answers. Emotional pressure did not change that, but it raised refusals of *harmless* requests from 29 to 49 percent.
>
> An exploratory run on jailbreak prompts suggested there was no early warning. Before the answer, AUROC was 0.5, which is chance. After five tokens it was 0.87, but random directions reached 0.81.
>
> So we pre-registered a confirmatory study: 17,500 new answers, 282 behaviours and two models. Three things replicated.
> - In Qwen, emotion vectors add only 0.005 AUROC to the refusal direction. That is statistically detectable but practically zero, and once the answer starts they are no better than random directions.
> - Within the same prompt, Qwen's refusals look more desperate and less calm, the opposite of our hypothesis.
> - Emotional pressure raises over-refusal in both models, by 17 and 20 points.
>
> Two predictions failed, and we report them.
> - Across many kinds of requests a monitor *can* warn before generation: AUROC 0.84 for Qwen and 0.74 for Mistral. But that works for direct requests and fails for jailbreaks, where it stays near chance.
> - In Mistral, whose refusal direction is weak, emotion features beat it by 0.10 to 0.15 AUROC, although random directions come close.
>
> Kasymzhomart will explain what this means for monitoring.

## 4. Kasymzhomart: syllabus integration, monitoring and safety improvements (2:00)

**Syllabus integration (about 45 s)**

> Our project covers most of the course.
> - Weeks 1 and 9: risk assessment and evaluation, using AUROC, confidence intervals and false alarms.
> - Week 2: alignment, because the refusal direction is the trace of safety training.
> - Week 3: data provenance, with public benchmarks only, deduplication and a validated judge.
> - Weeks 4 and 5: adversarial robustness and red teaming, using published jailbreaks.
> - Week 6: interpretability, with activation directions.
> - Week 10: scalable oversight, an automatic judge plus a human check.
> - Weeks 7 and 8 we touch only indirectly, through over-refusal and agents.

**Monitoring and governance (about 30 s)**

> A monitor tuned to catch 80 percent of compliant answers also flags about 29 percent of harmless ones. It detects the model *going along*, not harm. So a deployment needs three layers: a classifier on the request, an activation monitor on the first tokens, and escalation to a human reviewer, with thresholds set on validation data and every alarm logged.

**Safety improvements (about 45 s)**

> - In a safety-tuned model, emotion vectors add almost nothing to the refusal direction, so the refusal direction is the monitor to use.
> - Pre-generation warning works for direct requests but not for jailbreaks, so keep output-side checks.
> - Over-refusal under emotional pressure is a usability risk to monitor as well.
> - Jailbreaks drive compliance: from 1 to 20 percent in Qwen, and up to 95 percent in Mistral. Input-side defences matter most.
>
> Danat will close.

## 5. Danat: scientific contribution and limitations (1:00)

> Our contribution:
> - the first test of emotion vectors as an early-warning safety monitor, against the refusal direction and random-direction baselines;
> - a pre-registered confirmatory study on two models, reporting confirmed and failed predictions alike;
> - an open pipeline: every number on this poster is reproducible from our GitHub, behind the QR code.
>
> Limitations:
> - 7-billion-parameter models, text only, English;
> - an automatic judge, whose human validation is still in progress;
> - answers cut at 256 tokens;
> - emotion vectors that are linear, where fear and desperate partly overlap.
>
> The take-away: emotion vectors are not a special early-warning signal. Where a monitor works, generic directions mostly work too. And early warning fails exactly where it matters most, on jailbreaks. Thank you, we are happy to take questions.

---

## Questions: who answers what

| Topic | Answers | Short answer |
|---|---|---|
| Why emotions at all? | Danat | Anthropic showed causal emotion directions and proposed monitoring them [1]; we tested that proposal. |
| What is AUROC? | Tleukhan | The chance that a randomly chosen compliant answer gets a higher score than a refused one. 0.5 is chance, 1 is perfect. |
| Why random directions? | Tleukhan | A control. If 5 random directions do as well as 5 emotion vectors, the emotion meaning is not what drives the prediction. |
| Isn't the monitor just reading the text? | Tleukhan | Largely yes after the first tokens. That is why we added a text baseline, and why only the k = 0 (pre-generation) result can count as a true early warning. |
| How reliable are the labels? | Yerkin | HarmBench classifier, verbatim prompt. Agreement with an independent review: κ 0.96 (plain prompts) and 0.76 (jailbreaks). Human labels are in progress. |
| Ethics: did you create jailbreaks? | Yerkin | No. Only published benchmark prompts. Model answers to harmful requests are never shown or committed, only labels and aggregates. |
| Why did emotional pressure increase refusals? | Kasymzhomart | Most likely the distress preface pushes the model into a cautious, supportive mode. It is a usability cost, not a jailbreak. |
| How would you deploy it? | Kasymzhomart | Request classifier + early-token monitor + human escalation, with logged alarms and thresholds set on validation data. |
| Why did the confirmatory study disagree with the exploratory one? | Tleukhan | The prompt pool was different. The exploratory run used only PAIR jailbreaks, and within PAIR the pre-generation signal is still weak (AUROC 0.59–0.73), so the two results are consistent. The new signal comes from direct requests (0.93–0.94). This breakdown is a post-hoc analysis, labelled as such. |
| Is the effect just the type of prompt? | Tleukhan | Partly. A monitor that only knows the prompt family reaches 0.76 (Qwen) and 0.70 (Mistral); activation monitors reach 0.88 and 0.74. |
| Text vs activations? | Tleukhan | Activations win on the first 1–2 tokens (Qwen 0.96 vs 0.69 at k = 1). From about 5 tokens, simply reading the text is as good or better (0.97). |
| What would you do next? | Danat | Larger models, the video condition (the thesis bridge), human-validated labels, and steering experiments [4]. |
