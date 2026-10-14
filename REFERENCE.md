# ibutamoren reference

[Overview](README.md)

## How it works

**Tiled inference.** The image is cut into 224 px tiles, each read with 16 px
of surrounding context and edge-replicated to a multiple of 8. Each tile goes
through the model, the context margin is cropped off the output, and the
rest is placed into the result. GPU and WASM memory therefore stay bounded,
whatever the image size.

**Any target size.** Output can be a scale (1.5×–4×, or custom up to 8×) or a
resolution preset applied to the short side. The page chains 2× model passes
until it reaches or passes the target, then does one high-quality canvas
downsample to the exact size. A target at or below 1× is a plain resize with
no model pass. Any intermediate above 64 MP is refused with an error that
names the size it would have needed.

**Self-describing models.** When a model loads, a 16×16 probe tile measures its
real scale factor, so any 1×–4× model with dynamic input shapes drops in by
adding one entry to `MODELS`. fp16 models use `Float16Array` where the browser
has it and a bit-level conversion where it does not. The models are RGB-only,
so alpha is resampled separately and re-applied.

**Background tabs.** Between tiles the loop yields to the UI by racing
`requestAnimationFrame` against a 120 ms timeout. rAF alone stops firing in
background tabs and would freeze the run.

**Export.** AVIF (default), HEIC, JPEG, or PNG, at four quality tiers. Canvas
encoders vary by browser, so each format is feature-detected at startup with
`OffscreenCanvas.convertToBlob`. AVIF falls back to a vendored WebAssembly
encoder where the canvas cannot produce it, and the file extension comes from
the blob the encoder actually returned.

**One state object.** Every view on the page is rendered from a single `state`
object in `ibutamoren.js`: drop → preview → before/after comparison in one
card, a glass control bar, and a metrics panel with a progress ring, live
MP/s, elapsed time, tile count, and a per-tile timing sparkline.

**Accessibility.** The segmented pickers are radiogroups driven by arrow keys,
the comparison divider is a keyboard slider (arrows, Home, End), errors are
announced through `aria-live`, and the primary button is always enabled: with
no image loaded, it opens the file picker. Reduced motion, increased contrast,
and reduced transparency are all honored.

## Layout

```
index.html                  markup and styles
ibutamoren.js               all page logic (inference, tiling, rendering, export)
serve.py                    dev server with COOP/COEP headers
tools/patch-dynamic-shapes.py   ONNX output-shape patch for new models
models/                     ONNX models (own licenses)
vendor/                     ONNX Runtime Web 1.27.0, jSquash AVIF encoder
fonts/, fonts.css           self-hosted subset fonts
detect/                     tiny AVIF/HEIC files for decode detection
sample/                     the demo image
```
