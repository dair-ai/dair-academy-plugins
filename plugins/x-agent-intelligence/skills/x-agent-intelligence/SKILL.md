---
name: x-agent-intelligence
description: Build or refresh a readable local AI and agent intelligence feed from the official X MCP server. Use when the user asks for an X-based digest, monitoring dashboard, daily AI feed, source-account timeline, or a self-contained HTML artifact backed by X posts.
---

# X Agent Intelligence

Create a local, readable HTML intelligence feed from X posts retrieved through the official X API MCP server. The feed is a presentation artifact, not a hosted application.

## First check

Confirm that the agent can see an X MCP server. The expected official server is `https://api.x.com/mcp`. If no X tools are available, stop and point the user to `references/x-mcp-setup.md`; do not invent API responses or silently substitute scraping.

Read `../../references/x-mcp-setup.md` when setup, authentication, client configuration, or portability matters.

## Source configuration

Ask for or infer:

- Source handles, without `@`.
- Lookback window, normally 24 hours for a daily feed or 7 days for a backlog.
- Categories such as Coding Agents, Frameworks, Agent Research, Papers, Models, and Meta.
- Maximum stories per day, normally 10 to 25.
- Output path and whether the user wants a static snapshot or a refreshable data-backed shell.

Keep source configuration separate from story content so users can edit it later.

If the user asks to begin with the shared public list, read `../../references/starter-sources.md`. It is an optional starting point, not a requirement or a hidden default; users may edit or replace it.

## Collect with X MCP

Use the official X MCP tools exposed by the user's client. Tool names may be namespaced by the client, but the expected operations are:

1. Resolve handles with `get_users_by_usernames`, requesting `id`, `name`, `username`, `profile_image_url`, and `verified`.
2. Fetch posts with `get_users_posts` for each resolved user. Use `post.fields` including `created_at`, `text`, `author_id`, `public_metrics`, `entities`, and `attachments` when supported. Use expansions for `author_id`, `attachments.media_keys`, and `referenced_tweets.id` when supported.
3. If account feeds are unavailable, use `search_posts_all` with `from:handle` queries and explicit `start_time`/`end_time` windows.
4. Fetch missing post details with `get_posts_by_ids` or `get_posts_by_id`.
5. Use `get_users_by_id` only when an expanded author record is missing.

Do not use bookmark writes, Article publishing, likes, reposts, or other write tools for this skill. This workflow is read-only.

## Normalize and select

Convert each post into this schema:

```json
{
  "id": "post-id",
  "date": "YYYY-MM-DD",
  "created_at": "ISO timestamp",
  "category": "Models",
  "handle": "OpenAI",
  "author_name": "OpenAI",
  "verified": true,
  "profile_image_url": "https://...",
  "tweet_url": "https://x.com/OpenAI/status/post-id",
  "text": "raw text only when the user wants it retained",
  "title": "Short factual headline",
  "summary": "One or two sentences explaining why it matters.",
  "media_url": null,
  "media_type": null
}
```

Deduplicate by post ID. Exclude replies and reposts unless explicitly requested. Defensively remove any item with `in_reply_to_user_id` or a `referenced_tweets` entry of type `replied_to` or `retweeted`, even when the X MCP response ignored the requested exclusion. Prefer original announcements, papers, releases, benchmarks, demos, and concrete engineering reports. Rank using relevance, source quality, recency, engagement, and diversity across accounts. Do not let engagement alone determine importance.

Write titles and summaries as faithful paraphrases. Do not fabricate benchmarks, product capabilities, paper claims, or dates. Preserve the original post URL on every story.

## Render the artifact

Build one self-contained `feed.html`, using `../../assets/reference-artifact.html` as the design reference. Match its editorial reading-list layout closely: warm off-white canvas with a thin dark top rule; compact serif masthead; segmented story stats; a sticky, borderless category-chip toolbar; a narrow left date rail; and reading-list items with the title and "Why it matters" copy on the left and a fixed-size image or video preview on the right. Make the first story of the newest day the lead, with a larger thumbnail and headline.

Requirements:

- Inline all normalized story data in the artifact. No backend, build step, scheduler controls, or external runtime dependencies; plain HTML, CSS, and JavaScript unless a framework is requested.
- Include search, category filters, date grouping, source handles, avatars, original-post links, a clear updated timestamp, and image or video previews when `media_url` is available.
- Include the source-settings control: it opens a panel where the user can add, remove, copy, and reset X handles. Persist handles only in browser local storage.
- Use `profile_image_url` from X when available. Do not require `unavatar.io`.
- Use X widget embeds only as an optional enhancement. The feed must remain readable if X widgets fail to load.
- Seed the handle list from the adopter's own source configuration. The reference may seed the public handles from `references/starter-sources.md`, but never include credentials, private handles, automation identifiers, or orchestration configuration.
- Do not expose MCP credentials in the artifact or browser JavaScript.

## Validate before handing off

Check:

- The output opens locally without a backend.
- Every story has a valid X URL and handle.
- Dates sort newest first.
- Filters and search work with zero matching results.
- Missing avatars, media, and embeds degrade gracefully.
- No API key, bearer token, client secret, OAuth token, home-directory path, or private MCP URL appears in the output.
- The artifact labels generated summaries as summaries and links to the original post.

If recurring updates are requested, explain that the user may run the same prompt from any scheduler or orchestrator they choose. X MCP remains the source-access layer.
