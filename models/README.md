# Models

Three 2× anime upscalers, exported to ONNX by their authors. They are **not**
covered by this repository's MIT license: each keeps its own Creative Commons
license, and both licenses forbid commercial use.

| File | Model | Author | License |
|---|---|---|---|
| `2x_Ani4Kv2_G6i2_Compact_107500_fp32.onnx` | Ani4K v2 (Compact) | [Sirosky](https://github.com/Sirosky/Upscale-Hub) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `2x-AnimeSharpV2_MoSR_Sharp_fp16.onnx` | AnimeSharp V2 (MoSR, Sharp) | [Kim2091](https://github.com/Kim2091/Kim2091-Models) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `2x-AnimeSharpV4_Fast_RCAN_PU_fp16_opset17.onnx` | AnimeSharp V4 Fast (RCAN) | [Kim2091](https://github.com/Kim2091/Kim2091-Models) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

## Modification

Each file differs from the upstream export in one way: the output tensor's
dynamic dims are renamed (`height` → `out_2_height`, and so on), so ONNX Runtime
stops assuming the output is the same size as the input. The weights and graph
are untouched. `tools/patch-dynamic-shapes.py` performs exactly this change and
reproduces these files from the upstream exports. Under CC BY-NC-SA, the
modified AnimeSharp files are shared under that same license.

## Adding a model

Any 1×–4× ONNX model with an NCHW RGB input in `[0, 1]` and dynamic spatial dims
works: run the patch tool on it, drop it here, and add an entry to `MODELS` in
`ibutamoren.js`. The page probes the real scale factor with a 16×16 tile when the
session loads and refuses to run if it disagrees with the entry.
