# Tint Patches for Godot WebGPU

These patches modify the vendored Tint source for Godot's WebGPU backend.
They are applied on top of clean upstream Tint extracted via `extract_tint.sh`.

## Applying

From the repository root:

```bash
for p in thirdparty/tint/patches/*.patch; do
    patch -p1 < "$p"
done
```

## Patch Summary

| Patch | Files | Group | Description |
|-------|-------|-------|-------------|
| 0001 | validate.cc | UBO layout | `SetSkipBlockLayout(true)` + improved error messages |
| 0002 | validator.h, validator.cc, reader.cc | Spec constants | `kAllowStructMemberSizeMismatch` capability |
| 0003 | decompose_strided_array.cc | Spec constants | Skip padding when stride < element size |
| 0004 | shader_io.cc | Point size | Accept non-constant `point_size` stores |
| 0005 | ir_to_program.cc | Spec constants | `@size` emission guard + capability |
| 0006 | parse_num.cc | Vendoring | Replace `absl::from_chars` with `std::from_chars` |
| 0007 | validator.cc | UBO layout | Extend `kAllowStructMemberSizeMismatch` to the struct TOTAL-size check (spec-constant-sized arrays leave `Size()` at its unfolded value) |

## Logical Groups

**Group A — UBO Layout (0001)**: Godot uses C++ struct packing for uniform buffers,
not std140/std430. Always necessary.

**Group B — Specialization Constants (0002, 0003, 0005)**: Godot's specialization
constants can change struct/array sizes at runtime, creating size mismatches that
Tint's IR validator and lowering passes don't expect. These patches relax validation
and prevent invalid WGSL output.

**Group C — Point Size (0004)**: Godot shaders pass through `gl_PointSize` with
non-constant values. Tint strips point_size during lowering but validates the stored
value first. This patch relaxes that validation. Could potentially be moved to
`spirv_preprocess.cpp` or proposed upstream.

**Group D — Vendoring (0006)**: Replaces Abseil dependency with C++17 `std::from_chars`.
Always necessary when vendoring without Abseil.

## Upstream Source

Tint is extracted from [Dawn](https://dawn.googlesource.com/dawn) using
`extract_tint.sh`. The patches were generated against upstream `main` and verified
to apply cleanly and produce identical output via round-trip testing.
