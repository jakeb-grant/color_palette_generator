# Color Palette Generator

Generate functional color palettes from images for Zed editor themes (and other applications). Extracts colors using k-means clustering and enforces WCAG contrast requirements for readability.

## Features

- **Dual theme output** - Generates both dark and light themes from a single image
- **WCAG contrast enforcement** - All text colors meet minimum contrast ratios
- **24 terminal colors** - Base, bright, and dim variants for full terminal support
- **Smart border colors** - Blends toward accent colors when compatible, stays subtle when not
- **Zed theme output** - Ready-to-use JSON theme file for Zed editor
- **HTML preview** - Visual preview of the generated palette

## Installation

### Run directly from GitHub with uvx

```bash
uvx --from git+https://github.com/jakeb-grant/color_palette_generator color-palette-generator <image_path> [output_directory]
```

### Install locally

```bash
# Clone the repository
git clone https://github.com/jakeb-grant/color_palette_generator.git
cd color_palette_generator

# Run with uv
uv run color_palette_generator.py <image_path> [output_directory]
```

## Usage

```bash
# Generate theme from an image (outputs to same directory as image)
color-palette-generator my-wallpaper.png

# Specify output directory
color-palette-generator my-wallpaper.png ./my-theme/
```

## Output Files

For an image named `my-wallpaper.png`, the generator creates:

```
output_directory/
├── my-wallpaper.json          # Zed theme (both dark & light variants)
├── palette-dark.json          # Dark palette values
├── palette-light.json         # Light palette values
├── palette_preview-dark.html  # Visual preview (dark)
├── palette_preview-light.html # Visual preview (light)
├── readability_report-dark.txt
└── readability_report-light.txt
```

## Palette Structure

The generated palette includes:

### Backgrounds & Foregrounds
- `background`, `background_medium`, `background_light`, `background_disabled`
- `foreground`, `foreground_bright`, `foreground_medium`, `foreground_dim`

### Elements
- `element`, `element_hover`, `element_active`, `element_selected`, `element_disabled`

### Borders
- `border`, `border_variant`, `border_focused`, `border_selected`, `border_disabled`

### Accents
- `primary`, `primary_variant`, `secondary`, `secondary_variant`, `tertiary`
- `muted`, `selection`

### Semantic Colors
- `error`, `warning`, `success`, `info`

### Terminal Colors (24 total)
- Base: `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`
- Bright variants: `*_bright`
- Dim variants: `*_dim`

## Contrast Requirements

| Category | Minimum Contrast |
|----------|-----------------|
| Main text | 5.0:1 |
| Dim text | 4.0:1 |
| Terminal colors | 4.0:1 |
| Semantic colors | 4.5:1 |

## Examples

See the `out/` directory for example themes generated from the images in `images/`.

## License

MIT
