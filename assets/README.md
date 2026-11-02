# Method overview

- `method_overview.svg`: editable vector figure with live text and grouped icons.
- `method_overview.pdf`: vector export with embedded fonts.
- `method_overview.png`: 5016 x 2823 raster export for compatibility.

The figure reconstructs the paper's layout using explicit text, shapes, paths,
and connectors. Robot, document, and stacked-page icons are vector primitives;
neither the SVG nor the PDF embeds the original raster image.

Regenerate the three exports from the repository root:

```bash
python -m pip install reportlab pypdfium2
python scripts/render_method_overview.py
```

These are optional figure-rendering dependencies, separate from model training
and CPU metric replay. The renderer uses Times New Roman when installed, or
Liberation Serif on Linux. SVG viewers use available system fonts; the PDF
embeds the fonts used during export. `--output-dir` selects an export directory
and `--png-width` changes the raster width without changing the vector layout.
