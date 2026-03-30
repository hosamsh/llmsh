# Guidelines for the Human
You make the same kinds of mistakes the agents do — just at a different layer.

Before requesting work, ask yourself:
- What approach do I NOT want?. Say it.
- Are there UX or interface constraints? State them.
- Is there an existing pattern in the codebase this should follow? Point to it.
- Are there integration/async/streaming concerns? Name them.

A two-sentence constraint saves a two-round retry.

## Describe edge cases before implementation, not during debugging
If you know the feature touches:
- Streaming, async, or real-time data — say what happens on disconnect, timeout, partial data, empty response.
- External APIs — say what happens when they return errors, slow responses, unexpected formats.
- User input — say what happens with empty input, huge input, malformed input.

You already know where things break. The agent doesn't. Tell it.

## Scope requests to what you'll actually review
When you ask for 14-18 tasks in a batch, you're trusting the pipeline to catch everything. When it doesn't, the cost compounds — one bad early task poisons later ones.

- Large batches work well for mechanical tasks (rename, refactor, add a field).
- For tasks with design judgment, run smaller batches or add checkpoints.
- If you won't review the output of a task before the next one starts, make sure the task spec is airtight.

## Review before sharing externally
The agent doesn't know what's safe to share — you do.

Before forwarding agent output to others:
- Did anyone verify it works?
- Is it a recommendation or a tested fact?
- Would you stake your name on it?

## Don't confuse tolerance for strategy
You tolerate buggy first implementations as a cost of rapid iteration. That's a valid trade-off — but only if you're actually iterating faster. If a feature takes 3 rounds because the spec was thin, you didn't save time by skipping the spec.

The math: 5 minutes writing constraints < 20 minutes debugging a wrong approach.

## Sleep on it
Night work isn't inherently bad, but your prompts get shorter and your patience gets thinner. The agent doesn't get tired — but your ability to steer it does.
