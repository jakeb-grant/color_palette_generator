import argparse
import os

from .palette import generate_functional_palette, load_palette_from_json
from .opacity import calculate_theme_opacity
from .export import export_json, create_html_preview, generate_readability_report, print_palette
from .zed import generate_zed_theme, generate_zed_themes


def main():
    parser = argparse.ArgumentParser(
        description="Generate color palettes and Zed themes from images or palette JSON files"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the source image",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--from-palette",
        metavar="JSON",
        help="Load palette from existing JSON file instead of generating from image",
    )
    parser.add_argument(
        "--name",
        help="Theme name (required with --from-palette, otherwise derived from filename)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=None,
        help="Override blur theme opacity (0.0-1.0). If not set, auto-calculates optimal value.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.from_palette:
        if args.image_path:
            parser.error("Cannot use both image_path and --from-palette")
        if not args.name:
            parser.error("--name is required when using --from-palette")
        _run_from_palette(args)
    elif args.image_path:
        _run_from_image(args)
    else:
        parser.error("Either image_path or --from-palette is required")


def _run_from_palette(args):
    """Generate outputs from an existing palette JSON file."""
    palette_path = args.from_palette
    output_dir = args.output or os.path.dirname(palette_path) or "."
    theme_name = args.name
    override_opacity = args.opacity

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading palette: {palette_path}")

    # Load palette from JSON
    palette, is_dark_theme, json_opacity = load_palette_from_json(palette_path)
    variant = "dark" if is_dark_theme else "light"

    print(f"Detected theme type: {variant}")

    # Determine opacity: CLI override > JSON metadata > auto-calculate
    if override_opacity is not None:
        opacity = override_opacity
    elif json_opacity is not None:
        opacity = json_opacity
    else:
        opacity = calculate_theme_opacity(palette, is_dark_theme)

    # Print palette and generate report
    print_palette(palette, is_dark_theme)
    report, issues = generate_readability_report(palette, is_dark_theme)
    print("\n" + report)

    # Export paths
    palette_json_path = os.path.join(output_dir, f"palette-{variant}.json")
    html_path = os.path.join(output_dir, f"palette_preview-{variant}.html")
    report_path = os.path.join(output_dir, f"readability_report-{variant}.txt")
    zed_path = os.path.join(output_dir, f"{theme_name}.json")
    zed_blur_path = os.path.join(output_dir, f"{theme_name}-blur.json")

    # Export palette JSON (with updated opacity metadata)
    source_file = os.path.basename(palette_path)
    export_json(
        palette,
        palette_json_path,
        blur_opacity=opacity,
        source_file=source_file,
        theme_name=theme_name,
        is_dark=is_dark_theme,
        has_both_variants=False,
    )

    # Export HTML preview (pass empty list for extracted_colors since we don't have them)
    create_html_preview(palette, [], html_path, is_dark_theme)

    # Export readability report
    with open(report_path, "w") as f:
        f.write(report)

    # Export Zed themes (single variant)
    zed_theme = generate_zed_theme(palette, theme_name, is_dark_theme)
    with open(zed_path, "w") as f:
        f.write(zed_theme)

    zed_blur_theme = generate_zed_theme(palette, theme_name, is_dark_theme, opacity=opacity)
    with open(zed_blur_path, "w") as f:
        f.write(zed_blur_theme)

    # Print summary
    print("\n" + "=" * 60)
    print("Exported:")
    print(f"  - {palette_json_path}")
    print(f"  - {html_path}")
    print(f"  - {report_path}")
    print(f"  - {zed_path} (contains '{theme_name} {variant.title()}')")
    print(f"  - {zed_blur_path} (contains '{theme_name} {variant.title()} Blur')")
    print(f"\nBlur opacity: {opacity:.2f}")
    print("=" * 60)


def _run_from_image(args):
    """Generate palettes and outputs from an image file."""
    image_path = args.image_path
    output_dir = args.output or os.path.dirname(image_path) or "."
    override_opacity = args.opacity
    theme_name = args.name or os.path.splitext(os.path.basename(image_path))[0]

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
    source_file = os.path.basename(image_path)
    export_json(
        dark_palette,
        dark_json_path,
        blur_opacity=dark_opacity,
        source_file=source_file,
        theme_name=theme_name,
        is_dark=True,
        has_both_variants=True,
    )
    create_html_preview(
        dark_palette, dark_extracted, dark_html_path, is_dark_theme=True
    )
    with open(dark_report_path, "w") as f:
        f.write(dark_report)

    # Export light theme files
    export_json(
        light_palette,
        light_json_path,
        blur_opacity=light_opacity,
        source_file=source_file,
        theme_name=theme_name,
        is_dark=False,
        has_both_variants=True,
    )
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
