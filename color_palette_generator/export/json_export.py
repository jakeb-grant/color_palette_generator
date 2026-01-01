import json

from ..opacity import opacity_to_hex


def export_json(
    palette,
    filepath,
    blur_opacity=None,
    source_file=None,
    theme_name=None,
    is_dark=True,
    has_both_variants=True,
):
    """Export palette as JSON with all 24 terminal colors and metadata.

    Args:
        palette: The color palette dict
        filepath: Output file path
        blur_opacity: Optional blur opacity value
        source_file: Source image/palette filename for metadata
        theme_name: Theme name for Zed theme references
        is_dark: Whether this is a dark theme
        has_both_variants: Whether both dark and light variants exist
    """
    data = {k: v.hex for k, v in palette.items()}

    if blur_opacity is not None:
        data["_blur_opacity"] = {
            "float": round(blur_opacity, 2),
            "hex": opacity_to_hex(blur_opacity),
        }

    data["_alpha_suggestion"] = {
        "background": "E6",
        "selection": "80",
    }

    data["_note"] = (
        "24 terminal colors: black/red/green/yellow/blue/magenta/cyan/white "
        "with _bright and _dim variants"
    )

    if source_file:
        data["_wallpaper"] = source_file

    if theme_name:
        if has_both_variants:
            data["_zed_theme_dark"] = f"{theme_name} Dark Blur"
            data["_zed_theme_light"] = f"{theme_name} Light Blur"
        else:
            # Single variant: both keys point to the same theme
            variant = "Dark" if is_dark else "Light"
            theme_value = f"{theme_name} {variant} Blur"
            data["_zed_theme_dark"] = theme_value
            data["_zed_theme_light"] = theme_value

    data["_gtk_color_scheme"] = "prefer-dark" if is_dark else "prefer-light"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
