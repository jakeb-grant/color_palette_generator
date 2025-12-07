#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "numpy",
#     "pillow",
#     "scikit-learn"
# ]
# ///

import colorsys
import json
from collections import namedtuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# """
# Functional Color Palette Generator v2
# Extracts colors from an image and assigns functional roles with strict readability enforcement.
# """

Color = namedtuple("Color", ["hex", "rgb", "hsl", "luminance"])

# Contrast requirements
MIN_TEXT_CONTRAST = 5.0  # Main text against bg AND bg_light
MIN_DIM_CONTRAST = 4.0  # Dim text against bg AND bg_light
MIN_TERMINAL_CONTRAST = 4.0  # All terminal colors 1-15
MIN_SEMANTIC_CONTRAST = 4.5  # Error, warning, success, info

# Saturation limits to avoid gaudy colors
MAX_BG_SATURATION = 35  # Backgrounds shouldn't be too colorful
MAX_FG_SATURATION = 25  # Foregrounds should be near-neutral
MAX_ACCENT_SATURATION = 75  # Accents can be vibrant but not neon


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hsl(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s * 100, l * 100)


def hsl_to_rgb(h, s, l):
    h, s, l = h / 360, s / 100, l / 100
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def relative_luminance(r, g, b):
    """Calculate relative luminance per WCAG 2.0"""

    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(lum1, lum2):
    """Calculate contrast ratio between two luminances"""
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def create_color(r, g, b):
    """Create a Color namedtuple with all representations"""
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    return Color(
        hex=rgb_to_hex(r, g, b),
        rgb=(r, g, b),
        hsl=rgb_to_hsl(r, g, b),
        luminance=relative_luminance(r, g, b),
    )


def adjust_color(color, lightness_delta=0, saturation_delta=0):
    """Adjust a color's HSL values"""
    h, s, l = color.hsl
    new_s = max(0, min(100, s + saturation_delta))
    new_l = max(0, min(100, l + lightness_delta))
    r, g, b = hsl_to_rgb(h, new_s, new_l)
    return create_color(r, g, b)


def set_color_lightness(color, target_lightness):
    """Set a color to a specific lightness"""
    h, s, _ = color.hsl
    r, g, b = hsl_to_rgb(h, s, target_lightness)
    return create_color(r, g, b)


def set_color_saturation(color, target_saturation):
    """Set a color to a specific saturation"""
    h, _, l = color.hsl
    r, g, b = hsl_to_rgb(h, target_saturation, l)
    return create_color(r, g, b)


def ensure_contrast(color, bg_color, bg_light_color, min_contrast, is_dark_theme):
    """
    Adjust color to ensure it meets minimum contrast against BOTH backgrounds.
    Returns adjusted color.
    """
    max_iterations = 50
    step = 3 if is_dark_theme else -3

    current = color
    for _ in range(max_iterations):
        contrast_bg = contrast_ratio(current.luminance, bg_color.luminance)
        contrast_bg_light = contrast_ratio(current.luminance, bg_light_color.luminance)
        min_achieved = min(contrast_bg, contrast_bg_light)

        if min_achieved >= min_contrast:
            return current

        # Adjust lightness in the appropriate direction
        current = adjust_color(current, lightness_delta=step)

        # Clamp to avoid going too far
        if current.hsl[2] > 95 or current.hsl[2] < 5:
            break

    return current


def ensure_terminal_contrast(
    color, bg_color, bg_light_color, min_contrast, is_dark_theme
):
    """
    Adjust terminal color for readability. More aggressive than text.
    """
    max_iterations = 60
    step = 4 if is_dark_theme else -4

    current = color
    for _ in range(max_iterations):
        contrast_bg = contrast_ratio(current.luminance, bg_color.luminance)
        contrast_bg_light = contrast_ratio(current.luminance, bg_light_color.luminance)
        min_achieved = min(contrast_bg, contrast_bg_light)

        if min_achieved >= min_contrast:
            return current

        current = adjust_color(current, lightness_delta=step)

        if current.hsl[2] > 90 or current.hsl[2] < 10:
            break

    return current


def clamp_saturation(color, max_sat):
    """Reduce saturation if it exceeds max"""
    h, s, l = color.hsl
    if s > max_sat:
        r, g, b = hsl_to_rgb(h, max_sat, l)
        return create_color(r, g, b)
    return color


def blend_colors(color1, color2, factor):
    """Blend two colors together. factor=0 returns color1, factor=1 returns color2."""
    r1, g1, b1 = color1.rgb
    r2, g2, b2 = color2.rgb
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return create_color(r, g, b)


def is_accent_compatible(bg_color, accent_color, max_hue_diff=60, max_sat_diff=40):
    """Check if an accent color is close enough to blend with background for borders."""
    bg_h, bg_s, bg_l = bg_color.hsl
    acc_h, acc_s, acc_l = accent_color.hsl

    # Calculate hue difference (accounting for wraparound)
    hue_diff = abs(bg_h - acc_h)
    if hue_diff > 180:
        hue_diff = 360 - hue_diff

    # Check saturation difference
    sat_diff = abs(bg_s - acc_s)

    # Check lightness difference (accent shouldn't be extremely different)
    light_diff = abs(bg_l - acc_l)

    return hue_diff <= max_hue_diff and sat_diff <= max_sat_diff and light_diff <= 50


def opacity_to_hex(opacity):
    """Convert 0.0-1.0 opacity to hex string (00-ff)."""
    clamped = max(0.0, min(1.0, opacity))
    return f"{int(clamped * 255):02x}"


def blend_color_with_opacity(bg_color, wallpaper_rgb, opacity):
    """
    Calculate the blended color when bg_color is shown at given opacity over wallpaper.
    Returns a new Color with the blended RGB values.
    """
    bg_r, bg_g, bg_b = bg_color.rgb
    wp_r, wp_g, wp_b = wallpaper_rgb

    blended_r = int(bg_r * opacity + wp_r * (1 - opacity))
    blended_g = int(bg_g * opacity + wp_g * (1 - opacity))
    blended_b = int(bg_b * opacity + wp_b * (1 - opacity))

    return create_color(blended_r, blended_g, blended_b)


def calculate_safe_opacity(bg_color, fg_color, min_contrast, is_dark_theme):
    """
    Calculate the minimum opacity that maintains contrast with worst-case wallpaper.

    For dark themes: worst case is white wallpaper (#ffffff)
    For light themes: worst case is black wallpaper (#000000)

    Uses binary search to find the minimum opacity where:
        contrast(blended_bg, fg) >= min_contrast

    Returns opacity value 0.0-1.0 (higher = more opaque)
    """
    # Worst case wallpaper
    wallpaper_rgb = (255, 255, 255) if is_dark_theme else (0, 0, 0)

    low, high = 0.0, 1.0
    result = 1.0  # Default to fully opaque if we can't find a solution

    for _ in range(50):  # Binary search iterations
        mid = (low + high) / 2

        blended = blend_color_with_opacity(bg_color, wallpaper_rgb, mid)
        achieved_contrast = contrast_ratio(blended.luminance, fg_color.luminance)

        if achieved_contrast >= min_contrast:
            result = mid
            high = mid  # Try for more transparency (lower opacity)
        else:
            low = mid  # Need more opacity

        if high - low < 0.001:
            break

    return result


def calculate_theme_opacity(palette, is_dark_theme):
    """
    Calculate optimal opacity for a theme based on contrast preservation.
    Uses the lowest-contrast text (foreground_dim) to ensure all text remains readable.

    Returns opacity value 0.0-1.0
    """
    bg = palette["background"]
    fg_dim = palette["foreground_dim"]

    # Use MIN_DIM_CONTRAST as the threshold since that's our lowest acceptable contrast
    opacity = calculate_safe_opacity(bg, fg_dim, MIN_DIM_CONTRAST, is_dark_theme)

    # Add a small safety margin (5% more opaque)
    opacity = min(1.0, opacity + 0.05)

    return opacity


def calculate_layered_opacities(editor_target):
    """
    Calculate opacity values for the layered transparency system.

    The hierarchy (dark theme, darkest to lightest):
    - Editor area = most opaque (editor_target)
    - Panels = medium (global background alone)
    - Title/status bars = lightest (5% more transparent than editor)

    Stacking: editor_target = global + editor_layer × (1 - global)

    Returns dict with opacity values (0.0-1.0) for each layer.
    """
    # Global background (panels) is 95% of editor target (5% more transparent)
    global_opacity = editor_target * 0.95

    # Editor layer combines with global to reach editor_target
    # editor_target = global + editor_layer × (1 - global)
    # editor_layer = (editor_target - global) / (1 - global)
    if global_opacity < 1.0:
        editor_layer = (editor_target - global_opacity) / (1 - global_opacity)
    else:
        editor_layer = 0.0

    # Title/status bars are 5% more transparent than editor target (standalone)
    title_status_opacity = editor_target * 0.95

    # Tab bar stacks on global, should match editor combined opacity
    # So tab_bar_layer uses same calculation as editor_layer
    tab_bar_layer = editor_layer

    return {
        "global": global_opacity,  # background (panels inherit this)
        "editor_layer": editor_layer,  # editor.background, editor.gutter.background
        "tab_bar_layer": tab_bar_layer,  # tab_bar.background
        "tab_layer": editor_layer
        * 0.5,  # tab.active/inactive (50% of editor layer for subtlety)
        "title_status": title_status_opacity,  # title_bar, status_bar (standalone)
        "transparent": 0.0,  # panel, terminal, surface (fully transparent)
    }


def extract_colors(image_path, n_colors=20):
    """Extract dominant colors using k-means clustering"""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((300, 300))
    pixels = np.array(img).reshape(-1, 3)

    # Remove extreme pixels
    mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 735)
    filtered_pixels = pixels[mask]

    if len(filtered_pixels) < n_colors:
        filtered_pixels = pixels

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)

    colors = []
    for center in kmeans.cluster_centers_:
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        colors.append(create_color(r, g, b))

    return colors


def find_average_color(image_path):
    """Get overall average color of image"""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    avg = pixels.mean(axis=0)
    return create_color(int(avg[0]), int(avg[1]), int(avg[2]))


def generate_functional_palette(image_path, force_theme=None):
    """Generate a functional color palette with strict readability

    Args:
        image_path: Path to the source image
        force_theme: "dark", "light", or None (auto-detect from image)
    """
    colors = extract_colors(image_path, n_colors=20)
    avg_color = find_average_color(image_path)

    colors_by_lum = sorted(colors, key=lambda c: c.luminance)
    colors_by_sat = sorted(colors, key=lambda c: c.hsl[1], reverse=True)

    palette = {}

    # Determine if this should be a dark or light theme
    if force_theme is not None:
        is_dark_theme = force_theme == "dark"
    else:
        is_dark_theme = avg_color.luminance < 0.5

    # === BACKGROUND ===
    # Target lightness ranges for themes
    DARK_BG_MAX_LIGHTNESS = 18  # Dark themes: 8-18% lightness
    DARK_BG_MIN_LIGHTNESS = 8
    LIGHT_BG_MIN_LIGHTNESS = 85  # Light themes: 85-95% lightness
    LIGHT_BG_MAX_LIGHTNESS = 95

    if is_dark_theme:
        # Dark theme: pick a color with good saturation, then force it dark
        # Prefer colors with moderate saturation for character
        bg_base = sorted(colors, key=lambda c: abs(c.hsl[1] - 25))[0]
        h, s, l = bg_base.hsl
        # Clamp lightness to dark range
        target_l = min(max(l, DARK_BG_MIN_LIGHTNESS), DARK_BG_MAX_LIGHTNESS)
        # If the source color was bright, bring it down to dark range
        if l > DARK_BG_MAX_LIGHTNESS:
            target_l = DARK_BG_MAX_LIGHTNESS - 3  # Aim for ~15% lightness
        bg = create_color(*hsl_to_rgb(h, min(s, MAX_BG_SATURATION), target_l))
    else:
        # Light theme: pick a color with subtle saturation, then force it light
        bg_base = sorted(colors, key=lambda c: abs(c.hsl[1] - 15))[0]
        h, s, l = bg_base.hsl
        # Clamp lightness to light range
        target_l = min(max(l, LIGHT_BG_MIN_LIGHTNESS), LIGHT_BG_MAX_LIGHTNESS)
        # If the source color was dark, bring it up to light range
        if l < LIGHT_BG_MIN_LIGHTNESS:
            target_l = LIGHT_BG_MIN_LIGHTNESS + 5  # Aim for ~90% lightness
        bg = create_color(
            *hsl_to_rgb(h, min(s, MAX_BG_SATURATION / 2), target_l)
        )  # Lower saturation for light themes

    # Clamp background saturation to avoid gaudy
    bg = clamp_saturation(bg, MAX_BG_SATURATION)
    palette["background"] = bg

    # === BACKGROUND MEDIUM ===
    if is_dark_theme:
        bg_medium = adjust_color(bg, lightness_delta=4)
    else:
        bg_medium = adjust_color(bg, lightness_delta=-3)
    bg_medium = clamp_saturation(bg_medium, MAX_BG_SATURATION)
    palette["background_medium"] = bg_medium

    # === BACKGROUND LIGHT ===
    if is_dark_theme:
        bg_light = adjust_color(bg, lightness_delta=8)
    else:
        bg_light = adjust_color(bg, lightness_delta=-6)
    bg_light = clamp_saturation(bg_light, MAX_BG_SATURATION)
    palette["background_light"] = bg_light

    # === BACKGROUND DISABLED ===
    # Desaturated, slightly different from bg for disabled surfaces
    if is_dark_theme:
        bg_disabled = adjust_color(bg, lightness_delta=-2, saturation_delta=-10)
    else:
        bg_disabled = adjust_color(bg, lightness_delta=2, saturation_delta=-10)
    bg_disabled = clamp_saturation(bg_disabled, MAX_BG_SATURATION)
    palette["background_disabled"] = bg_disabled

    # === ELEMENT BACKGROUNDS ===
    # For interactive elements (buttons, inputs, checkboxes, etc.)
    if is_dark_theme:
        # element.background - sits between bg and bg_medium
        element_bg = adjust_color(bg, lightness_delta=2)
        # element.hover - lighter than element
        element_hover = adjust_color(bg, lightness_delta=5)
        # element.active - lighter than hover (pressed state)
        element_active = adjust_color(bg, lightness_delta=8)
        # element.selected - same as active
        element_selected = element_active
        # element.disabled - darker and desaturated
        element_disabled = adjust_color(bg, lightness_delta=-1, saturation_delta=-10)
    else:
        element_bg = adjust_color(bg, lightness_delta=-2)
        element_hover = adjust_color(bg, lightness_delta=-4)
        element_active = adjust_color(bg, lightness_delta=-6)
        element_selected = element_active
        element_disabled = adjust_color(bg, lightness_delta=2, saturation_delta=-10)

    palette["element"] = clamp_saturation(element_bg, MAX_BG_SATURATION)
    palette["element_hover"] = clamp_saturation(element_hover, MAX_BG_SATURATION)
    palette["element_active"] = clamp_saturation(element_active, MAX_BG_SATURATION)
    palette["element_selected"] = clamp_saturation(element_selected, MAX_BG_SATURATION)
    palette["element_disabled"] = clamp_saturation(element_disabled, MAX_BG_SATURATION)

    # === POST-PROCESS: Ensure backgrounds allow for readable text ===
    # For dark themes, if bg_light is too bright, we can't get enough contrast even with white text
    # Calculate max possible contrast (white text on bg_light)
    if is_dark_theme:
        white_lum = 1.0
        max_possible_contrast = contrast_ratio(white_lum, bg_light.luminance)

        # If we can't even get MIN_TEXT_CONTRAST with white, darken the backgrounds
        if max_possible_contrast < MIN_TEXT_CONTRAST + 0.5:  # Add buffer
            # Calculate required bg_light luminance for 5.5:1 with white
            # contrast = (1 + 0.05) / (bg_lum + 0.05) >= 5.5
            # bg_lum <= (1.05 / 5.5) - 0.05 = 0.14
            target_lum = 0.12

            # Darken bg_light until we hit target luminance
            while bg_light.luminance > target_lum and bg_light.hsl[2] > 5:
                bg_light = adjust_color(bg_light, lightness_delta=-3)

            # Also darken main bg proportionally
            lum_diff = palette["background_light"].luminance - bg_light.luminance
            if lum_diff > 0:
                bg = adjust_color(bg, lightness_delta=-int(lum_diff * 80))
                bg = clamp_saturation(bg, MAX_BG_SATURATION)
                palette["background"] = bg

            palette["background_light"] = bg_light

    # === FOREGROUND ===
    # Must have good contrast with BOTH bg and bg_light
    if is_dark_theme:
        # Start with lightest color from palette
        fg_candidates = [c for c in colors if c.luminance > 0.5]
        if fg_candidates:
            fg_base = sorted(fg_candidates, key=lambda c: c.luminance, reverse=True)[0]
        else:
            fg_base = colors_by_lum[-1]
        fg_base = adjust_color(fg_base, lightness_delta=10, saturation_delta=-20)
    else:
        # Light theme: dark foreground
        fg_candidates = [c for c in colors if c.luminance < 0.3]
        if fg_candidates:
            fg_base = sorted(fg_candidates, key=lambda c: c.luminance)[0]
        else:
            fg_base = colors_by_lum[0]
        fg_base = adjust_color(fg_base, lightness_delta=-10, saturation_delta=-20)

    # Clamp saturation and ensure contrast
    fg_base = clamp_saturation(fg_base, MAX_FG_SATURATION)
    fg = ensure_contrast(fg_base, bg, bg_light, MIN_TEXT_CONTRAST, is_dark_theme)
    palette["foreground"] = fg

    # === FOREGROUND BRIGHT ===
    # Brighter than foreground for emphasis, headings, etc.
    if is_dark_theme:
        fg_bright_base = adjust_color(fg, lightness_delta=8)
    else:
        fg_bright_base = adjust_color(fg, lightness_delta=-8)
    fg_bright = clamp_saturation(fg_bright_base, MAX_FG_SATURATION)
    palette["foreground_bright"] = fg_bright

    # === FOREGROUND MEDIUM ===
    # Intermediate between foreground and foreground_dim
    if is_dark_theme:
        fg_medium_base = adjust_color(fg, lightness_delta=-6)
    else:
        fg_medium_base = adjust_color(fg, lightness_delta=6)
    fg_medium = ensure_contrast(
        fg_medium_base, bg, bg_light, MIN_TEXT_CONTRAST, is_dark_theme
    )
    palette["foreground_medium"] = fg_medium

    # === FOREGROUND DIM ===
    # Start closer to fg, then ensure contrast - don't go too far from readable
    if is_dark_theme:
        fg_dim_base = adjust_color(fg, lightness_delta=-12)
    else:
        fg_dim_base = adjust_color(fg, lightness_delta=12)
    fg_dim = ensure_contrast(fg_dim_base, bg, bg_light, MIN_DIM_CONTRAST, is_dark_theme)
    palette["foreground_dim"] = fg_dim

    # === PRIMARY ACCENT ===
    vibrant_colors = [c for c in colors if c.hsl[1] > 35 and 0.1 < c.luminance < 0.75]
    if vibrant_colors:
        primary = sorted(vibrant_colors, key=lambda c: c.hsl[1], reverse=True)[0]
    else:
        primary = colors_by_sat[0]
    primary = clamp_saturation(primary, MAX_ACCENT_SATURATION)
    palette["primary"] = primary

    # === SECONDARY ACCENT ===
    primary_hue = primary.hsl[0]
    secondary_candidates = [
        c
        for c in colors
        if c.hsl[1] > 25
        and abs(c.hsl[0] - primary_hue) > 40
        and 0.1 < c.luminance < 0.75
    ]
    if secondary_candidates:
        secondary = sorted(secondary_candidates, key=lambda c: c.hsl[1], reverse=True)[
            0
        ]
    else:
        comp_hue = (primary_hue + 150) % 360
        r, g, b = hsl_to_rgb(comp_hue, min(primary.hsl[1], 60), 50)
        secondary = create_color(r, g, b)
    secondary = clamp_saturation(secondary, MAX_ACCENT_SATURATION)
    palette["secondary"] = secondary

    # === PRIMARY VARIANT ===
    # Lighter/darker version of primary for hover states, borders, etc.
    if is_dark_theme:
        primary_variant = adjust_color(primary, lightness_delta=15, saturation_delta=-5)
    else:
        primary_variant = adjust_color(
            primary, lightness_delta=-15, saturation_delta=-5
        )
    primary_variant = clamp_saturation(primary_variant, MAX_ACCENT_SATURATION)
    palette["primary_variant"] = primary_variant

    # === SECONDARY VARIANT ===
    # Lighter/darker version of secondary
    if is_dark_theme:
        secondary_variant = adjust_color(
            secondary, lightness_delta=15, saturation_delta=-5
        )
    else:
        secondary_variant = adjust_color(
            secondary, lightness_delta=-15, saturation_delta=-5
        )
    secondary_variant = clamp_saturation(secondary_variant, MAX_ACCENT_SATURATION)
    palette["secondary_variant"] = secondary_variant

    # === TERTIARY ===
    # High-contrast highlight color for syntax, links, accents
    tertiary_hue = (primary_hue + 80) % 360
    tertiary_candidates = [
        c for c in colors if abs(c.hsl[0] - tertiary_hue) < 50 and c.hsl[1] > 20
    ]
    if tertiary_candidates:
        tertiary_base = tertiary_candidates[0]
    else:
        r, g, b = hsl_to_rgb(tertiary_hue, 50, 55)
        tertiary_base = create_color(r, g, b)
    tertiary_base = clamp_saturation(tertiary_base, MAX_ACCENT_SATURATION)
    # Enforce contrast for readability as highlight/accent text
    tertiary = ensure_terminal_contrast(
        tertiary_base, bg, bg_light, MIN_SEMANTIC_CONTRAST, is_dark_theme
    )
    palette["tertiary"] = tertiary

    # === MUTED ===
    muted = adjust_color(
        bg, lightness_delta=-5 if is_dark_theme else 5, saturation_delta=-5
    )
    palette["muted"] = muted

    # === SELECTION ===
    selection = adjust_color(primary, lightness_delta=-15, saturation_delta=-25)
    palette["selection"] = selection

    # === BORDER COLORS ===
    # Borders should be subtle - close to background but with a hint of accent if compatible
    BORDER_BLEND_COMPATIBLE = 0.50  # Blend toward accent if compatible
    BORDER_BLEND_INCOMPATIBLE = 0.05  # Very subtle blend even if incompatible

    if is_dark_theme:
        # Dark mode: borders slightly lighter than background_light
        border_base = adjust_color(bg_light, lightness_delta=5)
    else:
        # Light mode: borders slightly darker than background
        border_base = adjust_color(bg, lightness_delta=-5)

    # Check if primary accent is compatible for blending
    if is_accent_compatible(bg, primary):
        blend_factor = BORDER_BLEND_COMPATIBLE
    else:
        blend_factor = BORDER_BLEND_INCOMPATIBLE

    border = blend_colors(border_base, primary, blend_factor)
    border_variant = blend_colors(
        adjust_color(border_base, lightness_delta=3 if is_dark_theme else -3),
        primary,
        blend_factor * 0.7,
    )

    palette["border"] = border
    palette["border_variant"] = border_variant

    # Focused/selected borders can be more prominent - use secondary with some blending
    if is_accent_compatible(bg, secondary):
        focus_blend = BORDER_BLEND_COMPATIBLE
    else:
        focus_blend = BORDER_BLEND_INCOMPATIBLE

    border_focused = blend_colors(border_base, secondary, focus_blend * 1.2)
    border_selected = blend_colors(border_base, secondary_variant, focus_blend)

    palette["border_focused"] = border_focused
    palette["border_selected"] = border_selected

    # Disabled border - very muted
    palette["border_disabled"] = adjust_color(
        border_base, lightness_delta=-3 if is_dark_theme else 3, saturation_delta=-10
    )

    # === SEMANTIC COLORS (with enforced contrast) ===
    # Error - red
    error_base = create_color(*hsl_to_rgb(0, 65, 55))
    palette["error"] = ensure_terminal_contrast(
        error_base, bg, bg_light, MIN_SEMANTIC_CONTRAST, is_dark_theme
    )

    # Warning - yellow/orange
    warning_base = create_color(*hsl_to_rgb(38, 70, 55))
    palette["warning"] = ensure_terminal_contrast(
        warning_base, bg, bg_light, MIN_SEMANTIC_CONTRAST, is_dark_theme
    )

    # Success - green
    success_base = create_color(*hsl_to_rgb(120, 50, 45))
    palette["success"] = ensure_terminal_contrast(
        success_base, bg, bg_light, MIN_SEMANTIC_CONTRAST, is_dark_theme
    )

    # Info - cyan/blue
    info_base = create_color(*hsl_to_rgb(200, 60, 50))
    palette["info"] = ensure_terminal_contrast(
        info_base, bg, bg_light, MIN_SEMANTIC_CONTRAST, is_dark_theme
    )

    # === TERMINAL COLORS (24 total: base, bright, dim) ===
    # Using descriptive names: black, red, green, yellow, blue, magenta, cyan, white
    # Black should always be dark, white should always be light (regardless of theme)

    # Black - always a dark color
    if is_dark_theme:
        palette["black"] = bg  # Dark background
        black_bright_base = adjust_color(bg, lightness_delta=30)
        palette["black_dim"] = adjust_color(bg, lightness_delta=-10)
    else:
        palette["black"] = fg  # Dark foreground
        black_bright_base = adjust_color(fg, lightness_delta=10)
        palette["black_dim"] = adjust_color(fg, lightness_delta=15)
    palette["black_bright"] = ensure_terminal_contrast(
        black_bright_base, bg, bg_light, MIN_TERMINAL_CONTRAST, is_dark_theme
    )

    # Red - base=error, bright=lighter, dim=darker
    palette["red"] = palette["error"]
    palette["red_bright"] = ensure_terminal_contrast(
        adjust_color(palette["error"], lightness_delta=12),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["red_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["error"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # Green - base=success, bright=lighter, dim=darker
    palette["green"] = palette["success"]
    palette["green_bright"] = ensure_terminal_contrast(
        adjust_color(palette["success"], lightness_delta=15),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["green_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["success"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # Yellow - base=warning, bright=lighter, dim=darker
    palette["yellow"] = palette["warning"]
    palette["yellow_bright"] = ensure_terminal_contrast(
        adjust_color(palette["warning"], lightness_delta=12),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["yellow_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["warning"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # Blue
    blue_candidates = [c for c in colors if 190 < c.hsl[0] < 260 and c.hsl[1] > 25]
    if blue_candidates:
        blue_base = sorted(blue_candidates, key=lambda c: c.luminance, reverse=True)[0]
    else:
        blue_base = palette["info"]
    palette["blue"] = ensure_terminal_contrast(
        blue_base, bg, bg_light, MIN_TERMINAL_CONTRAST, is_dark_theme
    )
    palette["blue_bright"] = ensure_terminal_contrast(
        adjust_color(palette["blue"], lightness_delta=15),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["blue_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["blue"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # Magenta
    magenta_candidates = [
        c for c in colors if (c.hsl[0] > 280 or c.hsl[0] < 20) and c.hsl[1] > 30
    ]
    if magenta_candidates:
        magenta_base = sorted(magenta_candidates, key=lambda c: c.hsl[1], reverse=True)[
            0
        ]
    else:
        magenta_base = create_color(*hsl_to_rgb(300, 50, 55))
    palette["magenta"] = ensure_terminal_contrast(
        magenta_base, bg, bg_light, MIN_TERMINAL_CONTRAST, is_dark_theme
    )
    palette["magenta_bright"] = ensure_terminal_contrast(
        adjust_color(palette["magenta"], lightness_delta=15),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["magenta_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["magenta"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # Cyan
    cyan_candidates = [c for c in colors if 160 < c.hsl[0] < 200 and c.hsl[1] > 25]
    if cyan_candidates:
        cyan_base = cyan_candidates[0]
    else:
        cyan_base = create_color(*hsl_to_rgb(180, 50, 50))
    palette["cyan"] = ensure_terminal_contrast(
        cyan_base, bg, bg_light, MIN_TERMINAL_CONTRAST, is_dark_theme
    )
    palette["cyan_bright"] = ensure_terminal_contrast(
        adjust_color(palette["cyan"], lightness_delta=15),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )
    palette["cyan_dim"] = ensure_terminal_contrast(
        adjust_color(
            palette["cyan"],
            lightness_delta=-15 if is_dark_theme else 12,
            saturation_delta=-10,
        ),
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    # White - always a light color
    if is_dark_theme:
        palette["white"] = fg_dim  # Light foreground
        palette["white_bright"] = fg
        white_dim_base = adjust_color(fg_dim, lightness_delta=-12, saturation_delta=-5)
    else:
        palette["white"] = bg  # Light background
        palette["white_bright"] = adjust_color(bg, lightness_delta=5)
        white_dim_base = adjust_color(bg, lightness_delta=-8, saturation_delta=-5)
    palette["white_dim"] = ensure_terminal_contrast(
        white_dim_base,
        bg,
        bg_light,
        MIN_TERMINAL_CONTRAST,
        is_dark_theme,
    )

    return palette, colors, avg_color, is_dark_theme


def generate_readability_report(palette, is_dark_theme):
    """Generate a detailed readability report for inspection"""
    bg = palette["background"]
    bg_light = palette["background_light"]

    report = []
    report.append("=" * 70)
    report.append("READABILITY REPORT")
    report.append("=" * 70)
    report.append(f"Theme: {'DARK' if is_dark_theme else 'LIGHT'}")
    bg_medium = palette["background_medium"]
    report.append(
        f"Background:       {bg.hex} (L: {bg.hsl[2]:.1f}%, S: {bg.hsl[1]:.1f}%)"
    )
    report.append(
        f"Background Med:   {bg_medium.hex} (L: {bg_medium.hsl[2]:.1f}%, S: {bg_medium.hsl[1]:.1f}%)"
    )
    report.append(
        f"Background Light: {bg_light.hex} (L: {bg_light.hsl[2]:.1f}%, S: {bg_light.hsl[1]:.1f}%)"
    )
    report.append("")

    # Check categories
    categories = [
        ("FOREGROUND (bright)", ["foreground_bright"], MIN_TEXT_CONTRAST),
        ("FOREGROUND (main)", ["foreground", "foreground_medium"], MIN_TEXT_CONTRAST),
        ("FOREGROUND (dim)", ["foreground_dim"], MIN_DIM_CONTRAST),
        (
            "ACCENTS",
            ["primary", "primary_variant", "secondary", "secondary_variant"],
            MIN_TERMINAL_CONTRAST,
        ),
        ("HIGHLIGHT", ["tertiary"], MIN_SEMANTIC_CONTRAST),
        ("SEMANTIC", ["error", "warning", "success", "info"], MIN_SEMANTIC_CONTRAST),
        (
            "TERMINAL BASE",
            ["red", "green", "yellow", "blue", "magenta", "cyan", "white"],
            MIN_TERMINAL_CONTRAST,
        ),
        (
            "TERMINAL BRIGHT",
            [
                "red_bright",
                "green_bright",
                "yellow_bright",
                "blue_bright",
                "magenta_bright",
                "cyan_bright",
                "white_bright",
            ],
            MIN_TERMINAL_CONTRAST,
        ),
        (
            "TERMINAL DIM",
            [
                "red_dim",
                "green_dim",
                "yellow_dim",
                "blue_dim",
                "magenta_dim",
                "cyan_dim",
                "white_dim",
            ],
            MIN_TERMINAL_CONTRAST,
        ),
    ]

    issues = []

    for cat_name, keys, min_contrast in categories:
        report.append(f"\n{cat_name} (min: {min_contrast}:1)")
        report.append("-" * 50)
        for key in keys:
            if key not in palette:
                continue
            c = palette[key]
            cr_bg = contrast_ratio(c.luminance, bg.luminance)
            cr_bg_light = contrast_ratio(c.luminance, bg_light.luminance)
            min_cr = min(cr_bg, cr_bg_light)

            status = "✓" if min_cr >= min_contrast else "✗ FAIL"
            if min_cr < min_contrast:
                issues.append((key, c.hex, min_cr, min_contrast))

            report.append(
                f"  {key:14} {c.hex}  vs bg: {cr_bg:4.1f}:1  vs bg_light: {cr_bg_light:4.1f}:1  {status}"
            )

    report.append("\n" + "=" * 70)
    if issues:
        report.append(f"ISSUES FOUND: {len(issues)}")
        for key, hex_val, achieved, required in issues:
            report.append(
                f"  - {key}: {hex_val} has {achieved:.1f}:1, needs {required}:1"
            )
    else:
        report.append("ALL COLORS PASS CONTRAST REQUIREMENTS ✓")
    report.append("=" * 70)

    return "\n".join(report), issues


def print_palette(palette, is_dark_theme):
    """Print palette info"""
    bg = palette["background"]

    print("\n" + "=" * 60)
    print(f"FUNCTIONAL COLOR PALETTE ({'DARK' if is_dark_theme else 'LIGHT'} THEME)")
    print("=" * 60)

    categories = [
        ("BACKGROUNDS", ["background", "background_light"]),
        ("FOREGROUNDS", ["foreground", "foreground_dim"]),
        ("ACCENTS", ["primary", "secondary", "tertiary", "muted", "selection"]),
        ("SEMANTIC", ["error", "warning", "success", "info"]),
        (
            "TERMINAL (Base)",
            ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"],
        ),
        (
            "TERMINAL (Bright)",
            [
                "black_bright",
                "red_bright",
                "green_bright",
                "yellow_bright",
                "blue_bright",
                "magenta_bright",
                "cyan_bright",
                "white_bright",
            ],
        ),
        (
            "TERMINAL (Dim)",
            [
                "black_dim",
                "red_dim",
                "green_dim",
                "yellow_dim",
                "blue_dim",
                "magenta_dim",
                "cyan_dim",
                "white_dim",
            ],
        ),
    ]

    for cat_name, keys in categories:
        print(f"\n{cat_name}:")
        for key in keys:
            if key in palette:
                c = palette[key]
                contrast = contrast_ratio(c.luminance, bg.luminance)
                print(f"  {key:18} {c.hex}  (contrast: {contrast:.1f}:1)")


def export_json(palette, filepath, blur_opacity=None):
    """Export palette as JSON with all 24 terminal colors and blur opacity"""
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
        "24 terminal colors: black/red/green/yellow/blue/magenta/cyan/white with _bright and _dim variants"
    )
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def create_html_preview(palette, extracted_colors, output_path, is_dark_theme):
    """Create an HTML preview of the palette"""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Color Palette Preview</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: {bg};
            color: {fg};
            padding: 40px;
            min-height: 100vh;
        }
        h1 { margin-bottom: 10px; font-weight: 400; }
        .theme-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 30px;
            background: {bg_light};
            color: {fg_dim};
        }
        h2 {
            margin: 30px 0 15px 0;
            font-weight: 400;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: {fg_dim};
        }
        .palette-section {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        .color-card {
            width: 160px;
            border-radius: 8px;
            overflow: hidden;
            background: {bg_light};
        }
        .color-swatch {
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 500;
        }
        .color-info {
            padding: 12px;
            font-size: 11px;
        }
        .color-name {
            font-weight: 600;
            margin-bottom: 4px;
        }
        .color-hex {
            opacity: 0.7;
        }
        .terminal-grid {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 10px;
        }
        .terminal-color {
            aspect-ratio: 1;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
        }
        .preview-box {
            background: {bg_light};
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
        }
        .preview-box h3 {
            margin-bottom: 15px;
            font-weight: 400;
        }
        .sample-text { margin: 8px 0; }
        .dim { color: {fg_dim}; }
        .primary { color: {primary}; }
        .secondary { color: {secondary}; }
        .blur-demo {
            background: {bg}E6;
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid {muted};
        }
        .contrast-test {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .contrast-box {
            padding: 20px;
            border-radius: 8px;
        }
        .contrast-box h4 {
            margin-bottom: 10px;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .contrast-box p {
            margin: 5px 0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Functional Color Palette</h1>
    <div class="theme-badge">{theme_type} Theme</div>

    <h2>Backgrounds & Foregrounds</h2>
    <div class="palette-section">
        {bg_fg_cards}
    </div>

    <h2>Element Backgrounds</h2>
    <div class="palette-section">
        {element_cards}
    </div>

    <h2>Borders</h2>
    <div class="palette-section">
        {border_cards}
    </div>

    <h2>Accent Colors</h2>
    <div class="palette-section">
        {accent_cards}
    </div>

    <h2>Semantic Colors</h2>
    <div class="palette-section">
        {semantic_cards}
    </div>

    <h2>Terminal Colors (0-15)</h2>
    <div class="terminal-grid">
        {terminal_colors}
    </div>

    <h2>Readability Test</h2>
    <div class="contrast-test">
        <div class="contrast-box" style="background: {bg}">
            <h4>On Background</h4>
            <p style="color: {fg_bright}">Foreground bright (headings)</p>
            <p style="color: {fg}">Foreground text (primary)</p>
            <p style="color: {fg_medium}">Foreground medium (secondary)</p>
            <p style="color: {fg_dim}">Foreground dim (comments)</p>
            <p style="color: {error}">Error message</p>
            <p style="color: {warning}">Warning message</p>
            <p style="color: {success}">Success message</p>
            <p style="color: {info}">Info message</p>
        </div>
        <div class="contrast-box" style="background: {bg_light}">
            <h4>On Background Light</h4>
            <p style="color: {fg_bright}">Foreground bright (headings)</p>
            <p style="color: {fg}">Foreground text (primary)</p>
            <p style="color: {fg_medium}">Foreground medium (secondary)</p>
            <p style="color: {fg_dim}">Foreground dim (comments)</p>
            <p style="color: {error}">Error message</p>
            <p style="color: {warning}">Warning message</p>
            <p style="color: {success}">Success message</p>
            <p style="color: {info}">Info message</p>
        </div>
    </div>

    <div class="preview-box">
        <h3>Terminal Preview</h3>
        <p style="color: {red}">red: Error output</p>
        <p style="color: {green}">green: Success / git additions</p>
        <p style="color: {yellow}">yellow: Warnings / strings</p>
        <p style="color: {blue}">blue: Info / directories</p>
        <p style="color: {magenta}">magenta: Keywords / magenta</p>
        <p style="color: {cyan}">cyan: Cyan / special</p>
    </div>
</body>
</html>"""

    def make_card(name, color):
        text_color = "#ffffff" if color.luminance < 0.5 else "#000000"
        return f"""<div class="color-card">
            <div class="color-swatch" style="background: {color.hex}; color: {text_color}">Aa</div>
            <div class="color-info">
                <div class="color-name">{name}</div>
                <div class="color-hex">{color.hex}</div>
            </div>
        </div>"""

    def make_terminal_color(name, color):
        text_color = "#ffffff" if color.luminance < 0.5 else "#000000"
        return f'<div class="terminal-color" style="background: {color.hex}; color: {text_color}">{name}</div>'

    bg_fg_names = [
        "background",
        "background_medium",
        "background_light",
        "background_disabled",
        "foreground_bright",
        "foreground",
        "foreground_medium",
        "foreground_dim",
    ]
    element_names = [
        "element",
        "element_hover",
        "element_active",
        "element_selected",
        "element_disabled",
    ]
    border_names = [
        "border",
        "border_variant",
        "border_focused",
        "border_selected",
        "border_disabled",
    ]
    accent_names = [
        "primary",
        "primary_variant",
        "secondary",
        "secondary_variant",
        "tertiary",
        "muted",
        "selection",
    ]
    semantic_names = ["error", "warning", "success", "info"]
    terminal_color_names = [
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
    ]

    replacements = {
        "{bg}": palette["background"].hex,
        "{bg_light}": palette["background_light"].hex,
        "{fg}": palette["foreground"].hex,
        "{fg_bright}": palette["foreground_bright"].hex,
        "{fg_medium}": palette["foreground_medium"].hex,
        "{fg_dim}": palette["foreground_dim"].hex,
        "{primary}": palette["primary"].hex,
        "{secondary}": palette["secondary"].hex,
        "{muted}": palette["muted"].hex,
        "{error}": palette["error"].hex,
        "{warning}": palette["warning"].hex,
        "{success}": palette["success"].hex,
        "{info}": palette["info"].hex,
        "{theme_type}": "Dark" if is_dark_theme else "Light",
    }

    # Add terminal colors
    for name in terminal_color_names:
        if name in palette:
            replacements[f"{{{name}}}"] = palette[name].hex

    for old, new in replacements.items():
        html = html.replace(old, new)

    html = html.replace(
        "{bg_fg_cards}", "\n".join(make_card(n, palette[n]) for n in bg_fg_names)
    )
    html = html.replace(
        "{element_cards}", "\n".join(make_card(n, palette[n]) for n in element_names)
    )
    html = html.replace(
        "{border_cards}", "\n".join(make_card(n, palette[n]) for n in border_names)
    )
    html = html.replace(
        "{accent_cards}", "\n".join(make_card(n, palette[n]) for n in accent_names)
    )
    html = html.replace(
        "{semantic_cards}", "\n".join(make_card(n, palette[n]) for n in semantic_names)
    )

    # Generate terminal color grid with base + bright variants
    terminal_grid = []
    for name in terminal_color_names:
        if name in palette:
            terminal_grid.append(make_terminal_color(name, palette[name]))
    for name in terminal_color_names:
        bright_name = f"{name}_bright"
        if bright_name in palette:
            terminal_grid.append(make_terminal_color(bright_name, palette[bright_name]))
    for name in terminal_color_names:
        dim_name = f"{name}_dim"
        if dim_name in palette:
            terminal_grid.append(make_terminal_color(dim_name, palette[dim_name]))
    html = html.replace("{terminal_colors}", "\n".join(terminal_grid))

    with open(output_path, "w") as f:
        f.write(html)


def _build_zed_style(palette, is_dark, opacity=None):
    """Build the style dict for a Zed theme from a palette.

    Args:
        palette: The color palette dict
        is_dark: Whether this is a dark theme
        opacity: Optional opacity value (0.0-1.0) for transparent blur theme.
                 If None, creates opaque theme (ff alpha).
    """
    # Terminal foreground/dim_foreground are inverted (dim_fg shows low-contrast text)
    term_fg = palette["foreground_bright"]
    term_dim_fg = palette["background"]  # Inverted for low contrast

    # Calculate opacity hex values using layered transparency system
    if opacity is not None:
        layers = calculate_layered_opacities(opacity)
        target_alpha = opacity_to_hex(opacity)  # Full editor target
        global_alpha = opacity_to_hex(layers["global"])
        editor_alpha = opacity_to_hex(layers["editor_layer"])
        tab_bar_alpha = opacity_to_hex(layers["tab_bar_layer"])
        tab_alpha = opacity_to_hex(layers["tab_layer"])
        title_status_alpha = opacity_to_hex(layers["title_status"])
        transparent_alpha = "00"
    else:
        # Opaque theme - all fully opaque
        target_alpha = "ff"
        global_alpha = "ff"
        editor_alpha = "ff"
        tab_bar_alpha = "ff"
        tab_alpha = "ff"
        title_status_alpha = "ff"
        transparent_alpha = "ff"

    style = {
        "border": f"{palette['border'].hex}{'ff' if opacity is None else opacity_to_hex(0.4)}",
        "border.variant": f"{palette['border_variant'].hex}{'ff' if opacity is None else opacity_to_hex(0.4)}",
        "border.focused": f"{palette['border_focused'].hex}{'ff' if opacity is None else opacity_to_hex(0.4)}",
        "border.selected": f"{palette['border_selected'].hex}ff",
        "border.transparent": f"#00000000",
        "border.disabled": f"{palette['border_disabled'].hex}ff",
        # Elevated surfaces (popups) - use full target opacity
        "elevated_surface.background": f"{palette['background_medium'].hex}{target_alpha}",
        # Surface - transparent (inherits from global)
        "surface.background": f"{palette['background_medium'].hex}{transparent_alpha}",
        # Global background - base layer for panels
        "background": f"{palette['background_light' if opacity is None else 'background_medium'].hex}{global_alpha}",
        "element.background": f"{palette['element'].hex}{'ff' if opacity is None else transparent_alpha}",
        "element.hover": f"{palette['element_hover'].hex}{'ff' if opacity is None else title_status_alpha}",
        "element.active": f"{palette['element_active'].hex}{'ff' if opacity is None else title_status_alpha}",
        "element.selected": f"{palette['element_selected'].hex}{'ff' if opacity is None else title_status_alpha}",
        "element.disabled": f"{palette['element_disabled'].hex}{'ff' if opacity is None else title_status_alpha}",
        "drop_target.background": f"{palette['element_hover'].hex}80",
        "ghost_element.background": f"#00000000",
        "ghost_element.hover": f"{palette['element_hover'].hex}ff",
        "ghost_element.active": f"{palette['element_active'].hex}ff",
        "ghost_element.selected": f"{palette['element_selected'].hex}ff",
        "ghost_element.disabled": f"{palette['element_disabled'].hex}ff",
        "text": f"{palette['foreground_bright'].hex}ff",
        "text.muted": f"{palette['foreground'].hex}ff",
        "text.placeholder": f"{palette['foreground_dim'].hex}ff",
        "text.disabled": f"{palette['foreground_dim'].hex}ff",
        "text.accent": f"{palette['tertiary'].hex}ff",
        "icon": f"{palette['foreground_bright'].hex}ff",
        "icon.muted": f"{palette['foreground'].hex}ff",
        "icon.disabled": f"{palette['foreground_dim'].hex}ff",
        "icon.placeholder": f"{palette['foreground'].hex}ff",
        "icon.accent": f"{palette['tertiary'].hex}ff",
        # Title/status bars - standalone, no stacking
        "status_bar.background": f"{palette['background_light'].hex}{title_status_alpha}",
        "title_bar.background": f"{palette['background_light'].hex}{title_status_alpha}",
        "title_bar.inactive_background": f"{palette['background_light'].hex}{title_status_alpha}",
        "toolbar.background": f"{palette['background'].hex}{editor_alpha}",
        # Tab bar - stacks on global
        "tab_bar.background": f"{palette['background_medium'].hex}{tab_bar_alpha}",
        "tab.inactive_background": f"{palette['background_medium'].hex}{tab_alpha}",
        "tab.active_background": f"{palette['background'].hex}{target_alpha}",
        "search.match_background": f"{palette['tertiary'].hex}66",
        # Panel - transparent (inherits global)
        "panel.background": f"{palette['background_medium'].hex}{transparent_alpha}",
        "panel.focused_border": None,
        "pane.focused_border": None,
        "scrollbar.thumb.background": f"{palette['primary'].hex}4c",
        "scrollbar.thumb.hover_background": f"{palette['primary_variant'].hex}ff",
        "scrollbar.thumb.border": f"{palette['primary_variant'].hex}{'ff' if opacity is None else transparent_alpha}",
        "scrollbar.track.background": f"#00000000",
        "scrollbar.track.border": f"{palette['primary'].hex}{'ff' if opacity is None else transparent_alpha}",
        "editor.foreground": f"{palette['foreground'].hex}ff",
        # Editor - stacks on global to reach target opacity
        "editor.background": f"{palette['background'].hex}{editor_alpha}",
        "editor.gutter.background": f"{palette['background'].hex}{editor_alpha}",
        "editor.subheader.background": f"{palette['background_medium'].hex}{editor_alpha}",
        "editor.active_line.background": f"{palette['background_medium'].hex}bf",
        "editor.highlighted_line.background": f"{palette['background_medium'].hex}ff",
        "editor.line_number": f"{palette['foreground_dim'].hex}ff",
        "editor.active_line_number": f"{palette['foreground_bright'].hex}ff",
        "editor.hover_line_number": f"{palette['foreground_dim'].hex}ff",
        "editor.invisible": f"{palette['foreground_dim'].hex}ff",
        "editor.wrap_guide": f"{palette['primary'].hex}0d",
        "editor.active_wrap_guide": f"{palette['primary'].hex}1a",
        "editor.document_highlight.read_background": f"{palette['tertiary'].hex}1a",
        "editor.document_highlight.write_background": f"{palette['primary'].hex}66",
        # Terminal - transparent (inherits from editor which stacks on global)
        "terminal.background": f"{palette['background'].hex}{transparent_alpha}",
        "terminal.foreground": f"{term_fg.hex}ff",
        "terminal.bright_foreground": f"{term_fg.hex}ff",
        "terminal.dim_foreground": f"{term_dim_fg.hex}ff",
        "terminal.ansi.black": f"{palette['black'].hex}ff",
        "terminal.ansi.bright_black": f"{palette['black_bright'].hex}ff",
        "terminal.ansi.dim_black": f"{palette['black_dim'].hex}ff",
        "terminal.ansi.red": f"{palette['red'].hex}ff",
        "terminal.ansi.bright_red": f"{palette['red_bright'].hex}ff",
        "terminal.ansi.dim_red": f"{palette['red_dim'].hex}ff",
        "terminal.ansi.green": f"{palette['green'].hex}ff",
        "terminal.ansi.bright_green": f"{palette['green_bright'].hex}ff",
        "terminal.ansi.dim_green": f"{palette['green_dim'].hex}ff",
        "terminal.ansi.yellow": f"{palette['yellow'].hex}ff",
        "terminal.ansi.bright_yellow": f"{palette['yellow_bright'].hex}ff",
        "terminal.ansi.dim_yellow": f"{palette['yellow_dim'].hex}ff",
        "terminal.ansi.blue": f"{palette['blue'].hex}ff",
        "terminal.ansi.bright_blue": f"{palette['blue_bright'].hex}ff",
        "terminal.ansi.dim_blue": f"{palette['blue_dim'].hex}ff",
        "terminal.ansi.magenta": f"{palette['magenta'].hex}ff",
        "terminal.ansi.bright_magenta": f"{palette['magenta_bright'].hex}ff",
        "terminal.ansi.dim_magenta": f"{palette['magenta_dim'].hex}ff",
        "terminal.ansi.cyan": f"{palette['cyan'].hex}ff",
        "terminal.ansi.bright_cyan": f"{palette['cyan_bright'].hex}ff",
        "terminal.ansi.dim_cyan": f"{palette['cyan_dim'].hex}ff",
        "terminal.ansi.white": f"{palette['white'].hex}ff",
        "terminal.ansi.bright_white": f"{palette['white_bright'].hex}ff",
        "terminal.ansi.dim_white": f"{palette['white_dim'].hex}ff",
        "link_text.hover": f"{palette['info'].hex}ff",
        "version_control.added": f"{palette['green'].hex}ff",
        "version_control.modified": f"{palette['yellow'].hex}ff",
        "version_control.deleted": f"{palette['red'].hex}ff",
        "version_control.conflict_marker.ours": f"{palette['success'].hex}1a",
        "version_control.conflict_marker.theirs": f"{palette['tertiary'].hex}1a",
        "conflict": f"{palette['warning'].hex}ff",
        "conflict.background": f"{palette['warning'].hex}1a",
        "conflict.border": f"{palette['yellow_dim'].hex}c2",
        "created": f"{palette['success'].hex}ff",
        "created.background": f"{palette['success'].hex}1a",
        "created.border": f"{palette['green_dim'].hex}c2",
        "deleted": f"{palette['error'].hex}ff",
        "deleted.background": f"{palette['error'].hex}1a",
        "deleted.border": f"{palette['red_dim'].hex}c2",
        "error": f"{palette['error'].hex}ff",
        "error.background": f"{palette['error'].hex}1a",
        "error.border": f"{palette['red_dim'].hex}c2",
        "hidden": f"{palette['foreground_dim'].hex}ff",
        "hidden.background": f"{palette['background_disabled'].hex}1a",
        "hidden.border": f"{palette['muted'].hex}ff",
        "hint": f"{palette['blue_bright'].hex}ff",
        "hint.background": f"{palette['secondary_variant'].hex}1a",
        "hint.border": f"{palette['secondary_variant'].hex}ff",
        "ignored": f"{palette['foreground_dim'].hex}ff",
        "ignored.background": f"{palette['background_disabled'].hex}1a",
        "ignored.border": f"{palette['primary'].hex}ff",
        "info": f"{palette['info'].hex}ff",
        "info.background": f"{palette['info'].hex}1a",
        "info.border": f"{palette['blue_dim'].hex}ff",
        "modified": f"{palette['warning'].hex}ff",
        "modified.background": f"{palette['warning'].hex}1a",
        "modified.border": f"{palette['yellow_dim'].hex}c2",
        "predictive": f"{palette['cyan_bright'].hex}ff",
        "predictive.background": f"{palette['cyan_bright'].hex}1a",
        "predictive.border": f"{palette['green_dim'].hex}c2",
        "renamed": f"{palette['tertiary'].hex}ff",
        "renamed.background": f"{palette['tertiary'].hex}1a",
        "renamed.border": f"{palette['secondary_variant'].hex}ff",
        "success": f"{palette['success'].hex}ff",
        "success.background": f"{palette['success'].hex}1a",
        "success.border": f"{palette['green_dim'].hex}c2",
        "unreachable": f"{palette['foreground'].hex}ff",
        "unreachable.background": f"{palette['primary'].hex}1a",
        "unreachable.border": f"{palette['primary'].hex}ff",
        "warning": f"{palette['warning'].hex}ff",
        "warning.background": f"{palette['warning'].hex}1a",
        "warning.border": f"{palette['yellow_dim'].hex}c2",
        "players": [
            {
                "cursor": f"{palette['tertiary'].hex}ff",
                "background": f"{palette['tertiary'].hex}ff",
                "selection": f"{palette['tertiary'].hex}3d",
            },
            {
                "cursor": f"{palette['magenta'].hex}ff",
                "background": f"{palette['magenta'].hex}ff",
                "selection": f"{palette['magenta'].hex}3d",
            },
            {
                "cursor": f"{palette['cyan'].hex}ff",
                "background": f"{palette['cyan'].hex}ff",
                "selection": f"{palette['cyan'].hex}3d",
            },
            {
                "cursor": f"{palette['error'].hex}ff",
                "background": f"{palette['error'].hex}ff",
                "selection": f"{palette['error'].hex}3d",
            },
            {
                "cursor": f"{palette['warning'].hex}ff",
                "background": f"{palette['warning'].hex}ff",
                "selection": f"{palette['warning'].hex}3d",
            },
            {
                "cursor": f"{palette['success'].hex}ff",
                "background": f"{palette['success'].hex}ff",
                "selection": f"{palette['success'].hex}3d",
            },
        ],
        "syntax": {
            "attribute": {
                "color": f"{palette['tertiary'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "boolean": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "comment": {
                "color": f"{palette['foreground_dim'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "comment.doc": {
                "color": f"{palette['foreground_medium'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "constant": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "constructor": {
                "color": f"{palette['blue_dim'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "embedded": {
                "color": f"{palette['foreground_bright'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "emphasis": {
                "color": f"{palette['tertiary'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "emphasis.strong": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": 700,
            },
            "enum": {
                "color": f"{palette['error'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "function": {
                "color": f"{palette['cyan_dim'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "hint": {
                "color": f"{palette['blue_bright'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "keyword": {
                "color": f"{palette['magenta'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "label": {
                "color": f"{palette['tertiary'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "link_text": {
                "color": f"{palette['cyan_dim'].hex}ff",
                "font_style": "normal",
                "font_weight": None,
            },
            "link_uri": {
                "color": f"{palette['cyan'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "namespace": {
                "color": f"{palette['foreground_bright'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "number": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "operator": {
                "color": f"{palette['cyan'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "predictive": {
                "color": f"{palette['cyan_bright'].hex}ff",
                "font_style": f"italic",
                "font_weight": None,
            },
            "preproc": {
                "color": f"{palette['foreground_bright'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "primary": {
                "color": f"{palette['foreground'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "property": {
                "color": f"{palette['error'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation": {
                "color": f"{palette['foreground'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation.bracket": {
                "color": f"{palette['foreground_medium'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation.delimiter": {
                "color": f"{palette['foreground_medium'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation.list_marker": {
                "color": f"{palette['error'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation.markup": {
                "color": f"{palette['error'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "punctuation.special": {
                "color": f"{palette['red'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "selector": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "selector.pseudo": {
                "color": f"{palette['tertiary'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "string": {
                "color": f"{palette['success'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "string.escape": {
                "color": f"{palette['foreground_medium'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "string.regex": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "string.special": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "string.special.symbol": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "tag": {
                "color": f"{palette['tertiary'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "text.literal": {
                "color": f"{palette['success'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "title": {
                "color": f"{palette['error'].hex}ff",
                "font_style": None,
                "font_weight": 400,
            },
            "type": {
                "color": f"{palette['cyan'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "variable": {
                "color": f"{palette['foreground'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "variable.special": {
                "color": f"{palette['yellow'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
            "variant": {
                "color": f"{palette['blue_dim'].hex}ff",
                "font_style": None,
                "font_weight": None,
            },
        },
    }

    # Add blur appearance for transparent themes
    if opacity is not None:
        style["background.appearance"] = "blurred"

    return style


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
                "style": _build_zed_style(
                    dark_palette, is_dark=True, opacity=dark_opacity
                ),
            },
            {
                "name": f"{theme_name} Light{name_suffix}",
                "appearance": "light",
                "style": _build_zed_style(
                    light_palette, is_dark=False, opacity=light_opacity
                ),
            },
        ],
    }
    return json.dumps(theme_data, indent=2)


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate color palettes and Zed themes from images"
    )
    parser.add_argument("image_path", help="Path to the source image")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: same as image)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=None,
        help="Override blur theme opacity (0.0-1.0). If not set, auto-calculates optimal value.",
    )

    args = parser.parse_args()

    image_path = args.image_path
    output_dir = args.output_dir or os.path.dirname(image_path) or "."
    override_opacity = args.opacity

    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing: {image_path}")

    # Generate both dark and light palettes
    dark_palette, dark_extracted, _, _ = generate_functional_palette(
        image_path, force_theme="dark"
    )
    light_palette, light_extracted, _, _ = generate_functional_palette(
        image_path, force_theme="light"
    )

    # Print and export dark theme
    print_palette(dark_palette, is_dark_theme=True)
    dark_report, dark_issues = generate_readability_report(
        dark_palette, is_dark_theme=True
    )
    print("\n" + dark_report)

    # Print and export light theme
    print_palette(light_palette, is_dark_theme=False)
    light_report, light_issues = generate_readability_report(
        light_palette, is_dark_theme=False
    )
    print("\n" + light_report)

    # Export paths
    # Get theme name from image filename (without extension)
    theme_name = os.path.splitext(os.path.basename(image_path))[0]

    dark_json_path = os.path.join(output_dir, "palette-dark.json")
    dark_html_path = os.path.join(output_dir, "palette_preview-dark.html")
    dark_report_path = os.path.join(output_dir, "readability_report-dark.txt")

    light_json_path = os.path.join(output_dir, "palette-light.json")
    light_html_path = os.path.join(output_dir, "palette_preview-light.html")
    light_report_path = os.path.join(output_dir, "readability_report-light.txt")

    zed_path = os.path.join(output_dir, f"{theme_name}.json")
    zed_blur_path = os.path.join(output_dir, f"{theme_name}-blur.json")

    # Calculate opacity for blur theme (needed for palette export too)
    if override_opacity is not None:
        dark_opacity = override_opacity
        light_opacity = override_opacity
    else:
        dark_opacity = calculate_theme_opacity(dark_palette, is_dark_theme=True)
        light_opacity = calculate_theme_opacity(light_palette, is_dark_theme=False)

    # Export dark theme files
    export_json(dark_palette, dark_json_path, blur_opacity=dark_opacity)
    create_html_preview(
        dark_palette, dark_extracted, dark_html_path, is_dark_theme=True
    )
    with open(dark_report_path, "w") as f:
        f.write(dark_report)

    # Export light theme files
    export_json(light_palette, light_json_path, blur_opacity=light_opacity)
    create_html_preview(
        light_palette, light_extracted, light_html_path, is_dark_theme=False
    )
    with open(light_report_path, "w") as f:
        f.write(light_report)

    # Export opaque Zed theme
    zed_theme = generate_zed_themes(dark_palette, light_palette, theme_name)
    with open(zed_path, "w") as f:
        f.write(zed_theme)

    # Export blur Zed theme
    zed_blur_theme = generate_zed_themes(
        dark_palette,
        light_palette,
        theme_name,
        dark_opacity=dark_opacity,
        light_opacity=light_opacity,
    )
    with open(zed_blur_path, "w") as f:
        f.write(zed_blur_theme)

    print("\n" + "=" * 60)
    print("Exported:")
    print(f"  - {dark_json_path}")
    print(f"  - {dark_html_path}")
    print(f"  - {dark_report_path}")
    print(f"  - {light_json_path}")
    print(f"  - {light_html_path}")
    print(f"  - {light_report_path}")
    print(f"  - {zed_path} (contains '{theme_name} Dark' and '{theme_name} Light')")
    print(
        f"  - {zed_blur_path} (contains '{theme_name} Dark Blur' and '{theme_name} Light Blur')"
    )
    print(f"\nBlur opacity: dark={dark_opacity:.2f}, light={light_opacity:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
