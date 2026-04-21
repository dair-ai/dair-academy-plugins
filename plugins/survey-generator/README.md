# Survey Generator (Fireworks AI)

A Claude Code plugin that generates polished, single-file HTML survey papers on any AI/ML topic. The invoking agent curates a research bundle from a public anchor resource; [Kimi K2.6](https://fireworks.ai/models/fireworks/kimi-k2p6) on [Fireworks AI](https://fireworks.ai/) then produces an academic-style HTML artifact with inline SVG figures, numbered sections, and a References list in a single API call.

## How It Works

The skill splits the work cleanly between the agent (research curation) and the model (one-shot artifact generation):

1. **Anchor resource read** - The agent fetches a public index you provide (an arXiv survey, a curated GitHub list, a canonical blog post, etc.) and extracts subtopics and referenced works.
2. **Taxonomy and sections** - The agent drafts a 4 to 8 branch taxonomy and 6 to 10 numbered sections covering introduction, foundations, methods, evaluation, and open problems.
3. **Bibliography curation** - The agent assembles 20 to 100 real papers (default 20 for a quick survey, 40 to 50 for a comprehensive one, 80 to 100 for an exhaustive one). Every paper needs key, authors, year, title, venue, and a 1 to 2 sentence summary.
4. **Artifact generation** - `build_artifact.py` sends `research_bundle.json` and `style_spec.json` to Kimi K2.6 on Fireworks, which returns a single self-contained HTML document with inline CSS, inline SVG figures, and parenthetical citations.

The style spec is topic-agnostic - figure geometries (taxonomy tree, paradigm panels, system stack) are parametrized off the research bundle and scale automatically as the taxonomy grows.

## Setup

### 1. Get a Fireworks AI API Key

Create a Fireworks AI account at [fireworks.ai](https://fireworks.ai/) and grab your API key from the dashboard.

### 2. Export the API Key

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export FIREWORKS_API_KEY="your_api_key_here"
```

Then restart your terminal or run `source ~/.zshrc`.

### 3. Install the Plugin

```bash
/plugin marketplace add dair-ai/dair-academy-plugins
/plugin install survey-generator@dair-academy-plugins
```

## Usage

Once installed, ask Claude Code to run the survey generator:

```
Run the survey generator on the topic "Reasoning Models"
using https://github.com/dair-ai/AI-Papers-of-the-Week
as the anchor source. Target 50 papers.
```

DAIR.AI's [AI Papers of the Week](https://github.com/dair-ai/AI-Papers-of-the-Week) is a good default anchor for broad AI/ML topics - it is an open-source, continuously updated index of notable papers with short summaries. Any public curated list, arXiv survey, or canonical blog post works too.

Claude will:

1. Read the anchor source and extract the landscape of relevant work.
2. Draft a taxonomy, section list, and bibliography into `research_bundle.json`.
3. Run `python3 build_artifact.py` from the skill directory.
4. Write the generated survey to `output/survey_kimi-k2p6_v{N}.html`.

Open the HTML file in your browser to read the finished artifact.

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `topic` | yes | Concise survey topic, e.g. "Agentic Engineering", "Diffusion Language Models" |
| `source_url` | yes | Public anchor resource (arXiv survey, curated list, canonical blog post) |
| `bibliography_size` | no | Default 20. Use 40 to 50 for comprehensive, 80 to 100 for exhaustive |
| `section_count` | no | Default 6 to 10 numbered sections |

If you do not supply these, the agent will ask.

## Budget and Scaling

The skill has been validated from 20 to 100 papers on a single Kimi K2.6 call with `max_tokens=81920`.

| Bibliography | Completion tokens (typical) | Elapsed (typical) |
|--------------|-----------------------------|-------------------|
| 20 papers | ~12k to 16k | ~60 to 90 s |
| 50 papers | ~25k to 32k | ~120 to 180 s |
| 100 papers | ~45k to 55k | ~240 to 360 s |

Prompt structure stays stable across runs, so Fireworks prompt caching keeps repeat runs on the same topic cheap.

## Iterating

If a figure looks weak, sharpen `style_spec.json` (the `required_figures` and `figure_quality_note` keys) and rerun. If prose is thin or sections are missing, tighten the `guidance` fields in `research_bundle.json`.

Do not edit the generated HTML directly. Iterate on inputs and rerun. Each run writes a new versioned output file so you can compare versions.

To compare models side by side:

```bash
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p5 python3 build_artifact.py
```

Output filenames are slugged by model (`survey_kimi-k2p5_v1.html`, `survey_kimi-k2p6_v1.html`, etc.).

## Reference Example

A worked 100-paper example on Agentic Engineering lives under `skills/survey-generator/examples/agentic-engineering/`:

- `research_bundle.json` - the curated input with 8 taxonomy branches, 10 sections, 100 real papers.
- `survey.html` - the generated single-file artifact (viewBox 820x880 auto-scaled for 30 leaves).

Use it as a structural starting point for your own topic.

## Why Fireworks AI?

Generating a 5000 to 6500 word survey with inline SVG figures in a single call needs a model that can handle a long structured prompt and emit a long structured output. Fireworks makes this practical:

- **Context and completion headroom** - Kimi K2.6 handles the full research bundle plus style spec plus an 80k-token completion budget.
- **Speed** - Optimized inference keeps even 100-paper runs under six minutes.
- **Cost** - Open-weight pricing keeps iteration on style and prose affordable.

## Credits

- Built by [DAIR.AI](https://github.com/dair-ai)
- Powered by [Fireworks AI](https://fireworks.ai/) and [Kimi K2.6](https://fireworks.ai/models/fireworks/kimi-k2p6)
