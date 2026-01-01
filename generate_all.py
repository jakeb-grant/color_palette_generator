#!/usr/bin/env python3
"""
Generate all themes from images and palette JSON files.
Consolidates themes into out/themes/ folder.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate all themes from images and palette JSON files"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=None,
        help="Override blur theme opacity (0.0-1.0) for all themes",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    images_dir = root / "images"
    palettes_dir = root / "palettes"
    out_dir = root / "out"
    themes_dir = out_dir / "themes"

    # Create themes directory
    themes_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg"}

    # Find all images
    images = []
    if images_dir.exists():
        images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]

    # Find all palette JSON files
    palettes = []
    if palettes_dir.exists():
        palettes = [f for f in palettes_dir.iterdir() if f.suffix.lower() == ".json"]

    if not images and not palettes:
        print(f"No images in {images_dir} or palettes in {palettes_dir}")
        return

    print(f"Found {len(images)} images and {len(palettes)} palettes to process\n")

    # Process each image
    for image_path in sorted(images):
        theme_name = image_path.stem
        theme_out_dir = out_dir / theme_name

        print(f"{'=' * 60}")
        print(f"Generating from image: {theme_name}")
        print(f"{'=' * 60}")

        cmd = [
            "uv",
            "run",
            "color-palette-generator",
            str(image_path),
            "-o",
            str(theme_out_dir),
        ]
        if args.opacity is not None:
            cmd.extend(["--opacity", str(args.opacity)])

        result = subprocess.run(cmd, cwd=root)

        if result.returncode != 0:
            print(f"Error generating {theme_name}")
            continue

        _copy_themes(theme_out_dir, theme_name, themes_dir)
        print()

    # Process each palette JSON
    for palette_path in sorted(palettes):
        theme_name = palette_path.stem
        theme_out_dir = out_dir / theme_name

        print(f"{'=' * 60}")
        print(f"Generating from palette: {theme_name}")
        print(f"{'=' * 60}")

        cmd = [
            "uv",
            "run",
            "color-palette-generator",
            "--from-palette",
            str(palette_path),
            "-o",
            str(theme_out_dir),
            "--name",
            theme_name,
        ]
        if args.opacity is not None:
            cmd.extend(["--opacity", str(args.opacity)])

        result = subprocess.run(cmd, cwd=root)

        if result.returncode != 0:
            print(f"Error generating {theme_name}")
            continue

        _copy_themes(theme_out_dir, theme_name, themes_dir)
        print()

    print(f"{'=' * 60}")
    print("Done! All themes consolidated in:")
    print(f"  {themes_dir}")
    print(f"{'=' * 60}")


def _copy_themes(theme_out_dir, theme_name, themes_dir):
    """Copy generated theme files to the consolidated themes directory."""
    # Copy blur theme
    blur_theme = theme_out_dir / f"{theme_name}-blur.json"
    if blur_theme.exists():
        shutil.copy(blur_theme, themes_dir / blur_theme.name)
        print(f"Copied {blur_theme.name} to {themes_dir}")

    # Copy opaque theme
    opaque_theme = theme_out_dir / f"{theme_name}.json"
    if opaque_theme.exists():
        shutil.copy(opaque_theme, themes_dir / opaque_theme.name)
        print(f"Copied {opaque_theme.name} to {themes_dir}")


if __name__ == "__main__":
    main()
