# YouTube Notetaker

Turn any YouTube talk into a self-contained, interactive study deep-dive backed by plain
markdown files. Each video becomes one markdown file with slide images at their timestamps, a
clean timestamped transcript, and editable notes. A small bundled server renders the whole
library as a single-page app (index grid + per-video split pane with slide deck, embedded
player, searchable transcript, and notes that save back to the files).

No database, no cloud service, no framework. Just files on disk you own, plus `yt-dlp`,
`ffmpeg`, and a ~180-line Python server.

## Install

```bash
/plugin marketplace add dair-ai/dair-academy-plugins
/plugin install youtube-notetaker@dair-academy-plugins
```

## Quick start

```bash
# deps
pip install yt-dlp pillow pyyaml        # plus ffmpeg via your package manager
export VIDEO_LIBRARY_DIR=~/video-deepdives
cd "$CLAUDE_PLUGIN_ROOT/skills/youtube-notetaker"

# add a video (see SKILL.md for the full step-by-step with curation)
scripts/setup.sh "https://www.youtube.com/watch?v=<id>"
scripts/download.sh <id> /tmp/ytnote-<id>
scripts/detect_slides.sh /tmp/ytnote-<id>/video.mp4 /tmp/ytnote-<id>
python3 scripts/contact_sheet.py /tmp/ytnote-<id>/video.mp4 /tmp/ytnote-<id>/scene_times.txt /tmp/ytnote-<id>/contact.jpg
# ...curate keep.txt, then:
python3 scripts/extract_slides.py <id> /tmp/ytnote-<id>/video.mp4 /tmp/ytnote-<id>/keep.txt > /tmp/ytnote-<id>/slides.json
python3 scripts/vtt_to_transcript.py /tmp/ytnote-<id>/*.vtt /tmp/ytnote-<id>/transcript.txt
# ...write notes into slides.json, then:
python3 scripts/write_library_item.py --id <id> --title "..." --speaker "..." \
  --tags a,b,c --slides /tmp/ytnote-<id>/slides.json --transcript /tmp/ytnote-<id>/transcript.txt

# view
python3 scripts/serve.py --dir "$VIDEO_LIBRARY_DIR" --port 8000
# open http://127.0.0.1:8000/
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VIDEO_LIBRARY_DIR` | `~/video-deepdives` | where markdown files + `_media/` live |
| `VIDEO_LIBRARY_PORT` | `8000` | default port for `serve.py` |

`serve.py` flags: `--dir`, `--port`, `--host`, `--artifact` (path to the HTML shell; defaults to
`reference/artifact.html`).

## Layout

```
$VIDEO_LIBRARY_DIR/
  <YTID>.md                      one file per video (frontmatter + transcript)
  _media/<YTID>-slide-NN.jpg     slide images
```

See `SKILL.md` for the full pipeline, the markdown file shape, and gotchas.
