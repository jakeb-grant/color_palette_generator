import json

from ..color import hex_to_rgb, create_color


def load_palette_from_json(json_path):
    """Load a palette dict from JSON, converting hex strings to Color objects.

    Args:
        json_path: Path to palette JSON file

    Returns:
        tuple: (palette dict with Color objects, is_dark_theme bool, blur_opacity or None)
    """
    with open(json_path) as f:
        data = json.load(f)

    palette = {}
    blur_opacity = None

    for key, value in data.items():
        # Skip metadata keys
        if key.startswith("_"):
            if key == "_blur_opacity" and isinstance(value, dict):
                blur_opacity = value.get("float")
            continue

        # Convert hex string to Color object
        if isinstance(value, str) and value.startswith("#"):
            rgb = hex_to_rgb(value)
            palette[key] = create_color(*rgb)

    # Detect dark/light theme from background luminance
    is_dark_theme = palette["background"].luminance < 0.5

    return palette, is_dark_theme, blur_opacity
