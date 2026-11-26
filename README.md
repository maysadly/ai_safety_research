# Can Emotion Vectors Warn Us Before a Model Breaks Its Safety Rules?

AI safety course project (QAIRU, AI+X master's programme) by Tileukhan Makhmutov.

We test whether internal emotion representations ("emotion vectors") in **Qwen2.5-Omni-7B** predict that the model is about to comply with a harmful request, and whether they add information beyond the known refusal direction.

- Plan and phases: [`TASK.md`](TASK.md)
- Working rules for Claude Code: [`CLAUDE.md`](CLAUDE.md)

> Ethics: harmful prompts come from JailbreakBench. Model completions to harmful prompts are never committed or published; only labels and aggregate metrics are.
