#!/usr/bin/env python3
"""
Generate all themes from images in the images folder.
Consolidates blur themes into out/themes/ folder.
"""

import shutil
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).parent
    images_dir = root / "images"
    out_dir = root / "out"
    themes_dir = out_dir / "themes"

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg"}

    # Find all images
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(images)} images to process\n")

    # Create themes directory
    themes_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in sorted(images):
        theme_name = image_path.stem
        theme_out_dir = out_dir / theme_name

        print(f"{'=' * 60}")
        print(f"Generating: {theme_name}")
        print(f"{'=' * 60}")

        # Run the generator
        result = subprocess.run(
            [
                "uv",
                "run",
                "color-palette-generator",
                str(image_path),
                str(theme_out_dir),
            ],
            cwd=root,
        )

        if result.returncode != 0:
            print(f"Error generating {theme_name}")
            continue

        # Copy blur theme to consolidated folder
        blur_theme = theme_out_dir / f"{theme_name}-blur.json"
        if blur_theme.exists():
            shutil.copy(blur_theme, themes_dir / blur_theme.name)
            print(f"Copied {blur_theme.name} to {themes_dir}")

        # Also copy opaque theme
        opaque_theme = theme_out_dir / f"{theme_name}.json"
        if opaque_theme.exists():
            shutil.copy(opaque_theme, themes_dir / opaque_theme.name)
            print(f"Copied {opaque_theme.name} to {themes_dir}")

        print()

    print(f"{'=' * 60}")
    print("Done! All themes consolidated in:")
    print(f"  {themes_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
