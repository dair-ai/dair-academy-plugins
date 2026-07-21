# DAIR Academy Plugins

Open-source plugin marketplace for Claude Code by [DAIR.AI Academy](https://academy.dair.ai/).

## Installation

Add the marketplace to Claude Code:

```bash
/plugin marketplace add dair-ai/dair-academy-plugins
```

Then install any plugin:

```bash
/plugin install <plugin-name>@dair-academy-plugins
```

## Available Plugins

| Plugin | Description |
|--------|-------------|
| [image-generator](./plugins/image-generator) | Generate and edit images using Gemini's Nano Banana Pro model |
| [lesson-generator](./plugins/lesson-generator) | Build compact, standalone multi-lesson course artifacts with navigation, objectives, flashcards, quizzes, and source links |
| [learn](./plugins/learn) | Help users learn topics through adaptive tutoring, practice exercises, study plans, and feedback |
| [llm-council](./plugins/llm-council) | Orchestrate multiple open-weight LLMs via Fireworks AI to deliberate on queries using the LLM Council approach |
| [survey-generator](./plugins/survey-generator) | Generate polished, single-file HTML survey papers on any AI/ML topic, powered by Kimi K2.6 on Fireworks AI |
| [youtube-notetaker](./plugins/youtube-notetaker) | Turn YouTube talks into local study deep-dives with extracted slides, timestamped transcripts, editable notes, and a bundled markdown-backed viewer |
| [wiki-builder](./plugins/wiki-builder) | Start, structure, grow, query, and maintain reusable research wikis with per-wiki configurable structure and flavors |
| [x-agent-intelligence](./plugins/x-agent-intelligence) | Build readable AI and agent intelligence feeds from the official X MCP server |

## Try These Without Setup

Most of these skills also run inside the [DAIR Academy AI Builder](https://academy.dair.ai/) — the easiest way to try them without installing anything locally.

## Contributing

We welcome community contributions. To add a plugin:

1. Create a new directory under `plugins/` following the standard structure
2. Include a `.claude-plugin/plugin.json` with plugin metadata
3. Add your skill(s) under `skills/<skill-name>/SKILL.md`
4. Include a `README.md` with setup instructions and usage examples
5. Submit a pull request

### Plugin Structure

```
plugins/<plugin-name>/
├── .claude-plugin/
│   └── plugin.json          # Plugin metadata (name, description, version, author)
├── README.md                # User-facing documentation
└── skills/
    └── <skill-name>/
        ├── SKILL.md          # Skill definition with YAML frontmatter
        ├── .env.example      # Environment variable template (if needed)
        └── [other files]     # Reference docs, templates, etc.
```

## License

Licensed under the [MIT License](./LICENSE). The root `LICENSE` covers the entire marketplace, and each plugin also includes its own `LICENSE` file.
