# LLM Council (Fireworks AI)

A Claude Code plugin that orchestrates multiple open-weight LLMs to deliberate on queries using Karpathy's [LLM Council](https://x.com/karpathy/status/1886955788325941490) concept. Powered entirely by fast, affordable inference on [Fireworks AI](https://fireworks.ai/).

## How It Works

The LLM Council runs a 3-phase deliberation process:

1. **Phase 1 - Individual Responses**: All selected models respond to your query independently (in parallel)
2. **Phase 2 - Cross-Model Ranking**: Each model reviews and ranks the other models' anonymized responses
3. **Phase 3 - Chairman Synthesis**: A designated Chairman model synthesizes the best final answer using all responses and rankings

The result is a more thorough, balanced, and high-quality answer than any single model would produce alone.

## Available Models

All models are open-weight and run on Fireworks AI:

| Model | Provider | Parameters |
|-------|----------|------------|
| GLM 5 | Z.ai | 744B (40B active MoE) |
| DeepSeek V3.1 | DeepSeek | - |
| Kimi K2.5 | Moonshot | - |
| Qwen3 235B | Alibaba | 235B (22B active MoE) |
| Llama 4 Maverick | Meta | - |

## Setup

### 1. Get a Fireworks AI API Key

Sign up for a free account at [app.fireworks.ai](https://app.fireworks.ai/) and grab your API key.

### 2. Export the API Key

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export FIREWORKS_API_KEY="your_api_key_here"
```

Then restart your terminal or run `source ~/.zshrc`.

### 3. Install the Plugin

```bash
claude plugin add /path/to/dair-academy-plugins/plugins/llm-council
```

## Usage

Once installed, just ask Claude Code to run the LLM Council:

```
Run the LLM council on this query: "What are the most important considerations when designing a retrieval-augmented generation system?"
```

Claude will:
1. Ask you to pick which models should participate
2. Ask you to pick a Chairman model
3. Run all three phases automatically
4. Display the full deliberation with individual responses, rankings, and the Chairman's synthesis

## Why Fireworks AI?

Running 5 models in parallel with ranking and synthesis requires a lot of inference calls. Fireworks makes this practical:

- **Speed**: Open-weight models run fast on Fireworks' optimized infrastructure
- **Cost**: Competitive pricing means running a full council deliberation stays affordable
- **Model variety**: Access to the latest open-weight models from multiple providers in one place

## Credits

- LLM Council concept by [Andrej Karpathy](https://x.com/karpathy/status/1886955788325941490)
- Built by [DAIR.AI](https://github.com/dair-ai)
- Powered by [Fireworks AI](https://fireworks.ai/)
