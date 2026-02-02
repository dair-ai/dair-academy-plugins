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

Open source - see individual plugins for their specific licenses.
