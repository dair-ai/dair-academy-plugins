# Wiki Builder

A Claude Code plugin for starting, structuring, growing, querying, and maintaining reusable research wikis. Each wiki is a standalone folder with its own sources, compiled pages, derived artifacts, prompts, and local configuration — built around a per-wiki `wiki.config.md` so structure and style can vary by use case.

> 🎥 **Watch the walkthrough**: [How to use the wiki-builder skill](https://academy.dair.ai/events/cmnivyzyp001n04k1rnozju2n) on DAIR Academy.

## How It Works

The skill is intentionally general. Rather than hard-coding a single wiki layout, it scaffolds a clean starting structure and then defers to each wiki's local `wiki.config.md` for purpose, audience, page types, style rules, and update workflow.

Supported wiki flavors out of the box:

- `research` — ongoing research notes and curated topics
- `paper` — single-paper deep dives
- `domain` — broad subject areas
- `product` — product / tool knowledge bases
- `person` — researcher / founder dossiers
- `organization` — company / lab profiles
- `project` — internal project knowledge bases

## Setup

Install the plugin:

```bash
/plugin marketplace add dair-ai/dair-academy-plugins
/plugin install wiki-builder@dair-academy-plugins
```

By default, wikis are created under `~/dair-wikis/`. To use a different location, set the `WIKI_ROOT` environment variable in your shell profile:

```bash
export WIKI_ROOT="$HOME/my-wikis"
```

Or pass `--root /path/to/wikis` to the init script on a per-call basis.

## Usage

Once installed, ask Claude Code things like:

```
Start a new wiki on agent memory using the research flavor.
```

```
Add this arXiv paper to my ai-evals wiki and compile a source page.
```

```
Compile a concept page on tool-use benchmarks in my agents wiki and link it from the index.
```

Claude will:

1. Resolve the task (start, ingest, compile, query, restructure, lint, or export).
2. Read the target wiki's `wiki.config.md` and prompts before making changes.
3. Preserve provenance for every claim using `sources.md`.
4. Generate durable pages under `wiki/` and update `wiki/index.md` and `logs/maintenance-log.md`.

## Wiki Layout

Each new wiki starts with this layout:

```text
<wiki-slug>/
├── wiki.config.md
├── raw/
├── wiki/
│   └── index.md
├── derived/
├── prompts/
│   ├── compile-index.md
│   ├── compile-source-page.md
│   ├── compile-concept-page.md
│   ├── query-and-file.md
│   └── lint-wiki.md
├── logs/
│   └── maintenance-log.md
└── sources.md
```

Common additions per flavor: `wiki/papers`, `wiki/concepts`, `wiki/people`, `wiki/products`, `wiki/organizations`, `wiki/timelines`, `wiki/questions`, `wiki/maps`, and `assets`.

## Manual Scaffold

If you prefer to scaffold a wiki without going through Claude:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/skills/wiki-builder/scripts/init_wiki.sh" \
  agent-memory \
  --title "Agent Memory" \
  --flavor research
```

Use `--root /custom/path` to override the default `~/dair-wikis` location.

## Quality Bar

- Make the first page useful immediately.
- Prefer explicit filenames and stable slugs.
- Separate raw source material from compiled interpretation.
- Link related wiki pages.
- Mark speculation and unknowns clearly.
- Avoid rewriting the same source summary in many places.
- Keep generated pages navigable for future agents and humans.

## Credits

Built by [DAIR.AI](https://github.com/dair-ai).
