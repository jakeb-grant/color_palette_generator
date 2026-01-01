# Color Palette Generator

Generate functional color palettes from images for Zed editor themes (and other applications). Extracts colors using k-means clustering and enforces WCAG contrast requirements for readability.

## Features

- **Dual theme output** - Generates both dark and light themes from a single image
- **Transparent blur themes** - Auto-generates blur variants with contrast-safe transparency
- **WCAG contrast enforcement** - All text colors meet minimum contrast ratios
- **24 terminal colors** - Base, bright, and dim variants for full terminal support
- **Smart border colors** - Blends toward accent colors when compatible, stays subtle when not
- **Zed theme output** - Ready-to-use JSON theme files for Zed editor
- **HTML preview** - Visual preview of the generated palette

## Installation

### Run directly from GitHub with uvx

```bash
uvx --from git+https://github.com/jakeb-grant/color_palette_generator color-palette-generator <image_path> [-o output_directory]
```

### Install locally

```bash
# Clone the repository
git clone https://github.com/jakeb-grant/color_palette_generator.git
cd color_palette_generator

# Run with uv
uv run color-palette-generator <image_path> [-o output_directory]
```

## Usage

### Generate from image

```bash
# Generate theme from an image (outputs to same directory as image)
color-palette-generator my-wallpaper.png

# Specify output directory
color-palette-generator my-wallpaper.png -o ./my-theme/

# Override blur theme opacity (0.0-1.0)
color-palette-generator my-wallpaper.png -o ./my-theme/ --opacity 0.85
```

### Generate from existing palette JSON

You can also generate outputs from an existing `palette-*.json` file:

```bash
# Load palette from JSON and generate all outputs
color-palette-generator --from-palette palette-dark.json -o ./output --name "my-theme"

# With custom opacity
color-palette-generator --from-palette palette-dark.json -o ./output --name "my-theme" --opacity 0.85
```

This is useful for:
- Manually tweaking a generated palette and regenerating outputs
- Creating themes from scratch without an image
- Re-exporting with different opacity values

**Note:** When using `--from-palette`, only a single theme variant (dark or light) is generated, detected automatically from the palette's background luminance.

## Output Files

For an image named `my-wallpaper.png`, the generator creates:

```
output_directory/
├── my-wallpaper.json          # Zed theme (opaque, dark & light variants)
├── my-wallpaper-blur.json     # Zed theme (transparent blur, dark & light variants)
├── palette-dark.json          # Dark palette values
├── palette-light.json         # Light palette values
├── palette_preview-dark.html  # Visual preview (dark)
├── palette_preview-light.html # Visual preview (light)
├── readability_report-dark.txt
└── readability_report-light.txt
```

## Blur Themes

The generator automatically creates transparent blur variants (`*-blur.json`) with:

- **`background.appearance: "blurred"`** - Enables Zed's blur effect
- **Auto-calculated opacity** - Uses binary search to find the maximum transparency that maintains WCAG contrast against worst-case wallpapers (white for dark themes, black for light themes)
- **Cascading transparency** - Main surfaces use base opacity, overlapping elements (tabs) use 50% of base to layer properly

Use the `--opacity` flag to override the auto-calculated value if desired.

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
