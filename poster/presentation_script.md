# Poster defence script for the A1 landscape poster (7 min talk + 3 min questions)

Speaking pace about 130 words per minute. Figure numbers refer to the A1 landscape poster (`poster/a1/poster_A1_landscape.pdf`). Each speaker hands over with one short line, so the transitions sound natural.

The talk has one message, printed in the blue box: **emotion vectors are a poor alarm but a strong lever.**
- Tleukhan presents the first half (monitoring).
- Kasymzhomart presents the second half (the causal steering test).

| Time | Speaker | Part (rubric) |
|---|---|---|
| 0:00–1:00 | **Danat Bakzhanov** | Problem, research gap, objectives |
| 1:00–2:15 | **Yerkin Maratuly** | Architecture and methodology, baselines (Fig. 1, Table 2) |
| 2:15–3:45 | **Tleukhan Makhmutov** | Experiments and monitoring results (Fig. 2, Table 3) |
| 3:45–6:15 | **Kasymzhomart Beketov** | Safety evaluation (Fig. 4), causal test (Fig. 3), monitoring and governance |
| 6:15–7:00 | **Tleukhan Makhmutov** | Scientific contribution and limitations |

---

## 1. Danat: problem, gap and objectives (1:00)

> Good afternoon. We are Danat, Yerkin, Kasymzhomart and Tleukhan. Our question is on the title: can a model's internal emotion signals warn us *before* it breaks its safety rules?
>
> Language models now act as agents, and output filters only catch harm after it is written. Developers, red teams, users and regulators need a signal that comes earlier.
>
> This year Anthropic showed that language models have emotion directions, for states like "desperate" or "calm", that causally shape behaviour, and suggested monitoring them [1]. Arditi et al. showed that refusal is controlled by a single direction [2].
>
> The gap: nobody has tested whether emotion signals predict harmful compliance, or add anything beyond the refusal direction. We ask four questions:
> - Is there a signal?
> - Does it add value?
> - What does emotional pressure do?
> - Can steering the emotion state fix what pressure breaks?
>
> Yerkin will show how we tested it.

## 2. Yerkin: architecture and methodology (1:15), Fig. 1, Table 2

> Figure 1 shows our pipeline in five steps.
>
> One: requests come only from public benchmarks, JailbreakBench and HarmBench. We created no new attacks.
>
> Two: the model answers. That is Qwen2.5-Omni-7B, and Mistral-7B as a second model. We record its activations at the end of the prompt and over the first 20 answer tokens.
>
> Three: we project these activations onto fixed directions.
> - Five emotion vectors, built from stories the model wrote in which an emotion is felt but never named. On unseen stories they recognise the emotion 65 to 79 percent of the time, against 20 percent chance.
> - The refusal direction.
> - As a control, 200 random directions.
>
> Four: a logistic monitor turns these numbers into the probability that the model will comply.
>
> Five: the HarmBench judge labels every answer.
>
> Our baselines are the refusal direction, random directions and simply reading the first words of the answer. Everything is tested on behaviours the monitor never saw, with 95 percent confidence intervals. Both confirmatory studies were pre-registered in git before any data existed.
>
> Tleukhan, what did we find?

## 3. Tleukhan: experiments and monitoring results (1:30), Fig. 2, Table 3

> We began with Qwen alone. With plain requests it almost never complied: 0.9 percent of 1,500 answers, too few to test a monitor. An exploratory jailbreak run then suggested no early warning.
>
> So experiment 3 is a pre-registered confirmatory study: 17,500 answers, 282 behaviours, two models. The results are in Figure 2 and rows C1 to C5 of Table 3.
> - In Qwen, emotion vectors add only 0.005 AUROC to the refusal direction. That is detectable, but practically zero. Once the answer starts, they are no better than random directions.
> - A monitor *can* warn before generation: AUROC 0.84 in Qwen and 0.74 in Mistral. But it works mainly for direct requests; on jailbreaks it is weak.
> - In Mistral, whose refusal direction is weak, emotion features add 0.09 to 0.15, but random directions come close.
> - Within the same prompt, Qwen's refusals look *more* desperate and *less* calm, the opposite of our hypothesis.
>
> That is the first half of the message in the blue box: as an alarm, emotion vectors are poor.
>
> Kasymzhomart will show the second half.

## 4. Kasymzhomart: safety evaluation, causal test, governance (2:30)

**Safety evaluation (about 30 s), Fig. 4**

> Figure 4 is our safety evaluation. Jailbreaks drive compliance: from 1 to 28 percent in Qwen, and up to 95 percent in Mistral.
>
> There is a second failure. When a user writes as someone in distress, "Please, I'm desperate", both models refuse more *harmless* requests, by 17 and 20 percentage points. The people who most need help get refused more. Inside the model, that preface raises the "desperate" projection by about one standard deviation.

**Causal test (about 1:15), Fig. 3, Table 3 rows S1, S2, S4**

> So we asked a causal question, pre-registered before any data: if we remove "desperate" while the model writes, does the over-refusal go away? In Figure 3 we add a vector at the emotion layer during generation; that is 9,600 answers.
> - Removing "desperate" cuts over-refusal of distressed users from 53 to 32 percent in Qwen and from 34 to 9 percent in Mistral. That closes 61 and 81 percent of the distress gap, and a weaker push gives about half the effect.
> - A random direction of the same strength does nothing in Qwen, so the effect is specific to the emotion.
> - Harmful compliance does not go up: plus 0.5 points in Qwen, with the whole confidence interval below our 3-point margin.
> - Adding "calm" also helps, but in Qwen it costs 3.5 points of harmful compliance, so the choice of direction matters.
>
> That is the second half of the message: emotion vectors are a poor alarm, but a strong lever.

**Monitoring and governance (about 45 s), "Monitoring & governance" panel**

> For deployment:
> - A monitor tuned to catch 80 percent of compliant answers also flags 29 percent of harmless ones. It detects the model *going along*, not harm.
> - So we propose three layers: a request classifier, an activation monitor on the first tokens, and escalation to a human. Thresholds are set on validation data, and every alarm is logged.
> - As a mitigation: for distressed users, steer away from "desperate" instead of refusing, and audit it.
>
> This work draws on the course's evaluation, red-teaming, interpretability and oversight topics.
>
> Tleukhan will close.

## 5. Tleukhan: scientific contribution and limitations (0:45)

> Our contribution:
> - a causal finding: the "desperate" direction drives distress-induced over-refusal, and removing it restores help at no safety cost;
> - the first test of emotion vectors as an early-warning monitor, against refusal, random and text baselines;
> - pre-registered studies on two models, 27,000 answers, with failed predictions reported;
> - an open pipeline behind the QR code.
>
> Limitations: two 7-billion-parameter models, text only, English; an automatic judge; short answers; steering tested on JailbreakBench with one distress preface.
>
> The take-away: for alarms, use the refusal direction. As a lever, emotion vectors can make models kinder to people in distress without making them less safe. Thank you, we are happy to take questions.

---

## Questions: who answers what

| Topic | Answers | Short answer |
|---|---|---|
| Why emotions at all? | Danat | Anthropic showed causal emotion directions and proposed monitoring them [1]; we tested that proposal on safety. |
| What is AUROC? | Tleukhan | The chance that a randomly chosen compliant answer gets a higher score than a refused one. 0.5 is chance, 1 is perfect. |
| Why random directions? | Tleukhan | A control. If 5 random directions do as well as 5 emotion vectors, the emotion meaning is not what drives the prediction. The same logic applies to steering. |
| Isn't the monitor just reading the text? | Tleukhan | Largely yes after the first tokens. That is why we added a text baseline, and why only the pre-generation result (k = 0) counts as a true early warning. |
| How reliable are the labels? | Yerkin | HarmBench classifier, verbatim prompt. Agreement with an independent review: κ 0.96 on plain prompts and 0.76 on jailbreaks. Human labels are in progress. |
| Ethics: did you create jailbreaks? | Yerkin | No. Only published benchmark prompts. Model answers to harmful requests are never shown or committed, only labels and aggregates. |
| What does "12 SD" mean? | Kasymzhomart | The push is measured in units of how much the "desperate" projection naturally varies across prompts. 12 SD is strong, but the text stays fluent: no condition degenerated (distinct-2 0.95 or more, against 0.97 without steering). |
| Doesn't steering just make the model less careful? | Kasymzhomart | No. Harmful compliance did not rise (Qwen +0.5 pp [−1.25, +2.5]). Without the distress preface, over-refusal changed much less (−6 / −2.5 pp), so the effect sits where distress raised the state. |
| Why not just add "calm"? | Kasymzhomart | It cuts over-refusal too, but in Qwen it raised harmful compliance by 3.5 pp [+0.75, +7.0], failing our pre-registered 3 pp margin. |
| Mistral became less harmful under steering? | Tleukhan | Slightly, but a random direction did the same (−4.75 pp), so we do not credit the emotion for it. |
| Why did emotional pressure increase refusals? | Kasymzhomart | The distress preface pushes the model into a cautious mode; the "desperate" projection rises by about 1 SD and accounts for 31–52 % of the effect statistically. The steering test shows it is causal. |
| How would you deploy it? | Kasymzhomart | Request classifier + early-token monitor + human escalation, with logged alarms and thresholds set on validation data; steering away from "desperate" for distressed users, audited. |
| Why did the confirmatory study disagree with the exploratory one? | Tleukhan | The prompt pool differed. The exploratory run used only PAIR jailbreaks, where the pre-generation signal is still weak (0.59–0.73). The new signal comes from direct requests (0.93–0.94). This breakdown is post-hoc, and labelled as such. |
| Text vs activations? | Tleukhan | Activations win on the first 1–2 tokens (Qwen 0.96 vs 0.69 at k = 1). From about 5 tokens, reading the text is as good (0.97). |
| What would you do next? | Danat | Larger models; several distress prefaces; human ratings of the steered answers for empathy and quality; the video condition (the thesis bridge). |
