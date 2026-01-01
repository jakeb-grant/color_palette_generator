import json

from .styles import build_zed_style


def generate_zed_theme(palette, theme_name, is_dark, opacity=None):
    """Generate a Zed theme JSON file with a single theme variant.

    Args:
        palette: The theme palette
        theme_name: Base name for the theme
        is_dark: Whether this is a dark theme
        opacity: Optional opacity (0.0-1.0). If set, creates blur theme.

    Returns:
        JSON string of the theme data
    """
    is_blur_theme = opacity is not None
    name_suffix = " Blur" if is_blur_theme else ""
    appearance = "dark" if is_dark else "light"
    variant_name = "Dark" if is_dark else "Light"

    theme_data = {
        "$schema": "https://zed.dev/schema/themes/v0.2.0.json",
        "name": f"{theme_name}{name_suffix}",
        "author": "Palette Generator",
        "themes": [
            {
                "name": f"{theme_name} {variant_name}{name_suffix}",
                "appearance": appearance,
                "style": build_zed_style(palette, is_dark=is_dark, opacity=opacity),
            },
        ],
    }
    return json.dumps(theme_data, indent=2)


def generate_zed_themes(
    dark_palette, light_palette, theme_name, dark_opacity=None, light_opacity=None
):
    """Generate a Zed theme JSON file with both dark and light variants.

    Args:
        dark_palette: The dark theme palette
        light_palette: The light theme palette
        theme_name: Base name for the theme
        dark_opacity: Optional opacity for dark theme (0.0-1.0). If set, creates blur theme.
        light_opacity: Optional opacity for light theme (0.0-1.0). If set, creates blur theme.

    Returns:
        JSON string of the theme data
    """
    # Determine theme name suffix based on opacity
    is_blur_theme = dark_opacity is not None or light_opacity is not None
    name_suffix = " Blur" if is_blur_theme else ""

    theme_data = {
        "$schema": "https://zed.dev/schema/themes/v0.2.0.json",
        "name": f"{theme_name}{name_suffix}",
        "author": "Palette Generator",
        "themes": [
            {
                "name": f"{theme_name} Dark{name_suffix}",
                "appearance": "dark",
                "style": build_zed_style(
                    dark_palette, is_dark=True, opacity=dark_opacity
                ),
            },
            {
                "name": f"{theme_name} Light{name_suffix}",
                "appearance": "light",
                "style": build_zed_style(
                    light_palette, is_dark=False, opacity=light_opacity
                ),
            },
        ],
    }
    return json.dumps(theme_data, indent=2)
