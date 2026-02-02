# Image Generator Plugin

Generate and edit images using Google's Gemini Nano Banana Pro model (`gemini-3-pro-image-preview`).

## Features

- **Text-to-Image**: Generate high-quality images from text descriptions
- **Image Editing**: Edit existing images with text prompts (add/remove elements, style transfer)
- **Multi-Image Composition**: Combine elements from multiple images
- **High Resolution**: Support for 1K, 2K, and 4K output resolutions
- **Accurate Text Rendering**: Generate images with legible, well-placed text
- **Multiple Aspect Ratios**: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- **Google Search Grounding**: Generate images based on real-time information
- **Multi-turn Editing**: Iteratively refine images through conversation

## Setup (Required)

Before using this plugin, you must set your Gemini API key as an environment variable:

1. **Get a free API key** from [Google AI Studio](https://aistudio.google.com/)

2. **Export the key** in your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Restart your terminal** or run `source ~/.zshrc` (or `~/.bashrc`)

## Usage

Invoke this skill when you want to:
- "Generate an image of..."
- "Create a picture showing..."
- "Edit this image to..."
- "Add [element] to this image"
- "Transform this photo into [style]"
- "Create a logo for..."
- "Generate a product mockup..."

## Examples

### Text-to-Image
```
Generate a photorealistic image of a cozy coffee shop interior with warm lighting
```

### Image Editing
```
Take this photo and add a small wizard hat to the cat
```

### Style Transfer
```
Transform this city photo into Van Gogh's Starry Night style
```

### Logo Creation
```
Create a minimalist logo for a coffee shop called 'The Daily Grind'
```

## Model Capabilities

The `gemini-3-pro-image-preview` (Nano Banana Pro) model features:
- Advanced reasoning for complex prompts
- "Thinking" process for composition refinement
- Up to 14 reference images for input
- Google Search grounding for real-time data
- High-resolution output up to 4K

## Limitations

- Best performance with: EN, ar-EG, de-DE, es-MX, fr-FR, hi-IN, id-ID, it-IT, ja-JP, ko-KR, pt-BR, ru-RU, ua-UA, vi-VN, zh-CN
- No audio or video inputs
- Works best with up to 5 high-fidelity images (up to 14 total)
- All generated images include a SynthID watermark

## Version

- Version: 1.1.1
- Author: DAIR.AI
