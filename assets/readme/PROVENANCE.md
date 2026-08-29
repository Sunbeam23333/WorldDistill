# README figure provenance

The README figures use the WorldDistill paper visual system.  TeX sources and
the raster components required to rebuild them live in [`source/`](source/).

| Asset | Evidence status | Provenance |
|---|---|---|
| `worlddistill_overview.png` | implemented + illustrative | The labeled topology follows the catalog, trainer registry, cache, CUDA-stream, target-mask, and memory code paths. The bow-tie base and task icons are generated illustrative components; they are not benchmark evidence. |
| `cuda_stream_runtime.png` | implemented | Vector schematic of the independent teacher stream, event/wait dependency, student stream, and target-only supervision. Timeline spacing is explanatory and is not a measured profiler trace. |
| `hybrid_sparse_memory.png` | implemented | Vector schematic of the full-history sparse anchors plus contiguous recent tail selected under a fixed frame budget. |
| `action_conditioned_rollout.png` | implemented + illustrative | The lower computation path follows context-forcing code. The street sequence is a generated schematic visualization, not output from a released checkpoint. |

Colors and typography are presentation choices. “NVIDIA Green” refers only to
the palette inspiration in the private candidate history; the public figure is
not produced, sponsored, or endorsed by NVIDIA.

## Rebuild

The diagrams require XeLaTeX, TikZ, `fontspec`, and Comic Sans MS. From the
repository root:

```bash
mkdir -p /tmp/worlddistill-figures
for source in assets/readme/source/*.tex; do
  latexmk -xelatex -interaction=nonstopmode \
    -output-directory=/tmp/worlddistill-figures "$source"
done
```

Rasterize the resulting PDFs with Poppler at the desired publication DPI; keep
the PDFs as the vector source of truth. Comic Sans MS is intentionally requested
by the paper's visual brief and must be installed (or explicitly substituted in
the three shared style files) on Linux builders.

The overview base/icons and the illustrative street strip were generated during
the 2026-08-29 figure-design session with OpenAI GPT Image 2, then cropped and
composited under TeX labels/formulas. No generated text was retained. The source
component filenames are preserved under `source/overview_assets/` and
`source/rollout_assets/`; the exact generation conversation remains in the
project history rather than being represented as experimental provenance.
