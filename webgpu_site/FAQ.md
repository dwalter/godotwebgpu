# Godot WebGPU — Frequently Asked Questions

## General / Getting Started

### 1. What is godot-webgpu?

A complete WebGPU rendering backend for Godot Engine 4.6.2 that enables Godot games to run in the browser using the modern WebGPU API instead of the legacy WebGL. It implements Godot's `RenderingDeviceDriver` interface — the same abstraction used by the Vulkan, Metal, and D3D12 backends — targeting the browser's WebGPU API via Emscripten's emdawnwebgpu port.

### 2. Which Godot renderer does it use?

The **Forward Mobile** renderer. This is Godot's lighter-weight 3D renderer designed for mobile GPUs, which maps well to WebGPU's single-queue, limited-feature-set model. The Forward Clustered renderer (Godot's desktop-class renderer) is not supported and is saved for a future update.

### 3. What browsers are supported?

- **Chrome 113+** (and Chromium-based browsers: Edge, Opera, Brave)
- **Firefox 120+** (via wgpu backend)
- **Safari 18+** (Technology Preview; WebGPU support still maturing)

All major desktop browsers with WebGPU enabled. Mobile browser support is emerging (Chrome Android with WebGPU flag, Safari iOS 18+).

### 4. How do I build the WebGPU export template?

```bash
scons platform=web target=template_release dlink_enabled=yes webgpu=yes opengl3=no threads=no
```

The `dlink_enabled=yes` flag enables Emscripten dynamic linking, which produces a main module (`godot.wasm`) and a side module (`godot.side.wasm`). It is required only for GDExtension support on web — the WebGPU template builds and runs without it (verified: compile, desktop rendering, and immersive-vr entry all work on a non-dlink build), and the monolithic build is smaller and avoids a multi-second main-thread linking step after download. Use `dlink_enabled=yes` when the project ships GDExtensions; omit it otherwise.

Requirements:
- Emscripten 4.0.10+ (for the emdawnwebgpu port)
- No Rust toolchain needed (Tint C++ translator is compiled directly into the engine)
- Standard Godot build dependencies (SCons, Python, C++ compiler)

### 5. Do I need to modify my Godot project to use WebGPU?

Almost nothing. Existing Godot 4.6 projects targeting the Mobile renderer will generally work without modification — shaders, materials, and scenes carry over as-is.

The one exception is **GPU readback**. Any code that reads data back from the GPU — including `RenderingDevice.texture_get_data()`, `Image.get_image()`, or compute shader result retrieval — must be adapted. On native backends (Vulkan, Metal, D3D12), these calls are synchronous and return data immediately. On WebGPU, the underlying `buffer.mapAsync()` is asynchronous and cannot resolve while C++ holds the WASM call stack, so the first call returns `null`/empty data. You must poll across frames until the result is available:

```gdscript
# Native (Vulkan/Metal): works immediately
var data = rd.texture_get_data(texture_rid, 0)

# WebGPU: must poll — first call returns null
var data = null
while data == null or data.size() == 0:
    await get_tree().process_frame  # unwind the call stack so the browser event loop can tick
    data = rd.texture_get_data(texture_rid, 0)
```

This 1+ frame latency applies to any GPU→CPU data transfer. If your project never reads back from the GPU (the common case for most games), no changes are needed.

The only other user-visible difference is setting `rendering/renderer/rendering_method.web` to `mobile` or `forward_plus` (which maps to `mobile` on web) in project settings.

### 6. Why WebGPU instead of WebGL 2?

WebGPU provides an asynchronous GPU API, which prevents the frame-stalling pipeline bubbles that are impossible to avoid on WebGL — the browser can overlap CPU and GPU work instead of blocking on synchronous GL calls. Beyond that, WebGL 2 maps to OpenGL ES 3.0, which lacks compute shaders, storage buffers, and the programmable pipeline features that Godot 4's RenderingDevice architecture requires. WebGPU provides a modern GPU API comparable to Vulkan/Metal/D3D12, enabling Godot's full RD-based renderer to run in the browser for the first time.

---

## Performance

### 7. How does WebGPU performance compare to native?

On desktop browsers with the optimizations enabled, WebGPU achieves **close to parity with native Vulkan/Metal** for typical Forward Mobile scenes — around 1.4x native frame times. The benchmark journey:
- Initial unoptimized: 3.25x slower than native
- After all optimizations: ~1.4x native

The key insight: WebGPU is IPC-bound (not GPU-bound), and our optimizations reduce IPC message count by 99%+ for common cases.

### 8. Why was WebGPU initially 3.25x slower than native?

Every WebGPU API call on the web crosses an inter-process communication boundary: WASM → JS context switch → browser GPU process (via Mojo IPC on Chromium). Each call costs ~200-500ns vs ~5ns on native. Godot's renderer, designed assuming "commands are free," issued ~24,000 GPU calls per frame — accumulating 7ms+ of pure IPC overhead.

### 9. How did you achieve near parity?

A layered optimization stack that reduces IPC message count:
1. **Staging buffer fixes**: Eliminated redundant 32MB flushes (200x improvement in staging cost)
2. **Shadow pass merging**: 196 render passes → 4 per frame (-43% frame time)
3. **Instance batching**: 20,000 draws → 1 instanced draw (-25%)
4. **Skeleton atlas**: 4,000 bone uploads → 1 per frame (-17%)
5. **firstInstance encoding**: Eliminates per-draw push constant IPC (-3.6%)
6. **Color pass batching**: 32,190 draws → 14 per frame (+34.6% FPS)

### 10. What's the startup time?

Approximately 15 seconds for first-time shader compilation (~383 unique SPIR-V stages converted to WGSL via Tint). Subsequent loads from browser cache are near-instant because the WGSL cache persists for the page session.

This is a known limitation. Future work: pre-compile WGSL at export time, shipping directly without runtime conversion.

### 11. Are there any performance regressions vs native I should know about?

- **Omni light shadows**: Forced to dual-paraboloid mode (slightly lower quality than cubemap, but 43% faster)
- **Shader startup**: ~15s first-time compilation (native has no equivalent cost)
- **Texture readback**: 1-frame async delay (native is synchronous)
- **No multi-threaded command recording**: Single queue, main thread only
- **No GPU-driven rendering**: Indirect draw count always uses max (wastes some GPU cycles)

---

## Architecture / Design

### 12. How does the WebGPU driver fit into Godot's architecture?

It implements `RenderingDeviceDriver` — the same abstract interface that the Vulkan, Metal, and D3D12 backends implement. The driver is selected at startup based on project settings. From the renderer's perspective, it's just another backend — the Forward Mobile renderer doesn't know or care that it's running on WebGPU.

The driver advertises its capabilities via `ApiTrait` enums, and the rendering server conditionally activates optimized code paths based on those traits. This keeps platform-specific logic out of the core renderer.

### 13. How are push constants handled without native WebGPU support?

WebGPU has no push constants. We emulate them with a **256KB storage buffer ring**:

- 1024 slots x 256 bytes each (matching `minStorageBufferOffsetAlignment`)
- A bind group with `hasDynamicOffset=true` is bound once, and the offset advances per draw
- A CPU shadow buffer accumulates all push constant data during command recording
- A single `wgpuQueueWriteBuffer` flushes the dirty range at submit time (one IPC per frame)

This design means push constant updates cost zero additional IPC — they're batched into the one submit-time flush.

### 14. How does the shader pipeline work?

Godot compiles GLSL → SPIR-V at engine build time (via glslang). At runtime in the browser:

1. SPIR-V bytecode is loaded from the shader container
2. Twelve binary-level preprocessing passes in C++ (`spirv_preprocess.cpp`) transform the SPIR-V for WebGPU compatibility (spec constant folding, push constant rewrite, combined sampler split, Y-negation, readonly inference, and more)
3. Tint (Google's Dawn/Chrome SPIR-V→WGSL compiler, compiled directly into the engine WASM) parses the preprocessed SPIR-V and generates WGSL
4. Post-processing fixes browser-specific issues (f32::MAX literals, f16 stripping, derivative uniformity)
5. C++ applies format remapping
6. `wgpuDeviceCreateShaderModule` creates the GPU shader

Results are cached by 64-bit SPIR-V hash — identical stages are never converted twice.

### 15. Why Tint instead of Naga? Why vendor and patch it?

Tint is Google's SPIR-V→WGSL compiler from the Dawn/Chrome project — it's literally what Chrome uses for WebGPU shader compilation. We chose Tint over the previous Naga (Rust) approach because:
- **C++ only** — matches Godot's build system (no Rust/wasm-pack dependency)
- **Compiled directly into the engine WASM** — no separate sidecar binary to load
- **Far fewer patches** — 6 patches covering 8 files (3 logical groups) vs Naga's 42 patched files
- **BSD 3-Clause license** — compatible with Godot's licensing

Six small patches to vendored Tint handle Godot-specific requirements:
1. UBO layout relaxation (Godot uses C++ struct packing, not std140/std430)
2. Spec constant size mismatches (Godot's specialization constants change struct sizes at runtime) — 5 files
3. Point size tolerance (accept non-constant `gl_PointSize` stores)
4. Vendoring (replace `absl::from_chars` with `std::from_chars`)

### 16. How does the WebGPU device get created?

JavaScript creates the device **before** WASM loads (since device creation is async and Emscripten C++ can't await Promises without Asyncify). The `engine.js` shell:
1. Calls `navigator.gpu.requestAdapter()` with `powerPreference: 'high-performance'`
2. Requests a device with all supported optional features and maximum limits
3. Passes the device into the Emscripten module config
4. C++ imports it via `WebGPU.importJsDevice()`
5. Tint is already compiled into the engine WASM — no separate shader converter to load

### 17. Why a single 7733-line implementation file?

This matches the Vulkan driver's structure — `rendering_device_driver_vulkan.cpp` is also a single 6598-line file. Both backends implement the same `RenderingDeviceDriver` interface in one `.cpp`. The WebGPU driver is actually *more* factored than Vulkan, having extracted `webgpu_objects.h` (429 lines) and `pixel_formats_webgpu.h` (705 lines) into separate files. The file is navigable due to clear `/* SECTION */` markers and extensive comments.

---

## Compatibility

### 18. What rendering features work on WebGPU vs native?

**Fully working**: 3D Forward Mobile rendering, 2D Canvas, sky rendering, shadow mapping (dual-paraboloid), skeleton animation, tone mapping, SMAA, debanding, instance batching, texture compression (BC/ETC2/ASTC if GPU supports it).

**Limited/degraded**: Omni shadows (DP only, no cubemap), texture readback (async 1-frame delay), storage textures (format promoted), FSR (unavailable — storage bit removed from render targets).

**Not available**: Subpass-based post-processing, multiview/VR, VRS, tessellation, geometry shaders, subgroups, half-float math, texture component swizzle (emulated on CPU for L8/LA8).

### 19. Will my custom shaders work?

Yes, with caveats:
- Standard Godot shader language features work
- `modf`, `isinf`, `isnan` are automatically replaced with WGSL-compatible equivalents
- Array varyings are decomposed into individual variables
- Switch statements are converted to if/else chains
- Canvas shaders: binding 0 in set 3 is reserved for push constant emulation (bindings shifted to 1-4)

If you use very advanced GLSL features or rely on specific Vulkan behaviors, test on WebGPU early.

### 20. Does it work on mobile browsers?

Emerging support:
- **Chrome Android** (with WebGPU flag): Works, but Adreno GPUs may trigger float32→float16 texture downgrades
- **Safari iOS 18+**: WebGPU support is early; works with the binding_array and format workarounds
- **Firefox Android**: Not yet shipping WebGPU

Mobile WebGPU is the next frontier. The performance optimizations (especially IPC reduction) are even more impactful on mobile where the trampoline cost is 3-5x higher.

### 21. What about SharedArrayBuffer / COOP/COEP headers?

WebGPU does **not** require SharedArrayBuffer or cross-origin isolation headers. This means:
- Non-threaded WebGPU builds work on any hosting provider (GitHub Pages, Netlify, etc.)
- No special server configuration needed
- Only threaded builds (for audio/physics offloading) need COOP/COEP

This is a significant deployment advantage over threaded web builds.

### 22. How does it handle browsers that don't support WebGPU?

The HTML shell (`full-size.html`) checks `navigator.gpu` on load. If WebGPU is unavailable, it displays a clear message directing users to update their browser. The engine never starts if WebGPU is missing.

For dual-renderer builds (`webgpu=yes opengl3=yes`), the export plugin can be configured to fall back to WebGL if WebGPU is unavailable.

---

## Technical Deep-Dives

### 23. How does the bind group rebinding cache work?

WebGPU requires bind group layouts to match exactly when binding a uniform set to a pipeline. When a uniform set created with shader A's layout needs to work with shader B's pipeline (common in Godot where one material is used with multiple shaders), the driver:

1. Checks if layouts are pointer-equal (fast path — same shader)
2. Checks the per-uniform-set rebind cache
3. On cache miss: adapts entries (adjusts sampler types, texture sample types, view dimensions), filters to target layout's bindings, creates a new bind group, caches it

This avoids re-creating bind groups every frame while ensuring WebGPU validation passes.

### 24. How does format promotion work for storage textures?

WebGPU only supports a limited set of storage texture formats (see spec section 26.1.1). When Godot creates a texture with `StorageBinding` usage and a format like R8Unorm:

1. **At texture creation**: format promoted to R32Float (actual GPU allocation)
2. **In WGSL shaders**: format name `r8unorm` replaced with `r32float` via string substitution
3. **On upload**: data converted from R8 to R32Float by `texture_upload_convert`
4. **On readback**: data converted back from R32Float to R8 by `texture_readback_convert`

This is transparent to the engine — it thinks it's working with R8, and the driver handles all conversion.

### 25. What's the encoder splitting mechanism for sync isolation?

WebGPU has implicit synchronization within a command encoder, but it cannot handle a texture being used as both a read (TextureBinding) and write (RenderAttachment) target within the same encoder without an explicit scope break.

When the driver detects a render pass attachment that has both usage flags, it:
1. Flushes the push constant shadow buffer
2. Finishes and submits the current command encoder
3. Creates a fresh encoder

This ensures the prior read operations complete before the write begins. It's conservative (splits even when no actual conflict exists in the current frame) but prevents WebGPU validation errors that would invalidate the entire command buffer.

### 26. How does async buffer readback work without blocking?

WebGPU's `mapAsync` returns a Promise that cannot be awaited from synchronous C++ (without Asyncify, which was abandoned). The solution is a frame-deferred pattern:

**Frame N**:
- Engine calls `texture_get_data()` or `buffer_get_data_direct()`
- Driver copies GPU texture/buffer to a staging buffer
- Initiates `wgpuBufferMapAsync` on the staging buffer
- Returns empty data (not ready yet)

**Frame N+1**:
- Engine calls again
- Callback has fired between frames (during `requestAnimationFrame`)
- Driver returns the mapped data from the staging buffer
- Initiates a fresh copy for the next request

The 1-frame latency is acceptable for viewport capture, profiling, and compute result readback.

### 27. How is the combined-sampler splitting implemented?

At the SPIR-V binary level (before Tint parsing), the most complex preprocessing pass (~500 lines of C++):

1. Identifies all `OpVariable` with `OpTypeSampledImage` pointer type
2. For each combined variable at binding N:
   - Keeps original variable as the **image** (reuses original ID for reference stability)
   - Creates a new variable as the **sampler** at binding N*2
   - Image gets binding N*2+1
3. Rewrites all `OpLoad` of combined variables to emit separate image + sampler loads followed by `OpSampledImage`
4. Handles function parameters by rewriting call sites
5. Doubles ALL other bindings to avoid collision (except PC ring buffer at binding 120)
6. Updates `OpEntryPoint` interface lists

This approach is more robust than post-conversion text patching because it handles edge cases like multi-level function call chains and access chain expressions.

---

## For Godot Core Contributors

### 28. What changes were made to the base RenderingDeviceDriver interface?

10 new `ApiTrait` enum values and ~9 new virtual methods were added to `rendering_device_driver.h`. All new methods have safe default implementations (return invalid/false/empty), so non-WebGPU backends are completely unaffected.

New methods include: `buffer_create_with_data`, `buffer_initiate_async_map`, `buffer_write_direct`, `buffer_get_data_direct`, `buffer_flush`, `texture_get_gpu_pixel_size`, `texture_readback_convert`, `texture_upload_convert`, `texture_initialize_direct_layered`, `command_copy_buffer_to_texture_layered`.

### 29. What would need to change for upstream Godot acceptance?

1. **RFC for API traits**: The 10 new trait enums need formal review by Godot rendering team
2. **Upstream Tint patches**: Submit the 6 patches to the Dawn/Tint project (patches 5 & 6 are arguably Tint bugs)
4. **Reduce WGSL text manipulation**: Move remaining string-level transforms into SPIR-V preprocessing where possible
5. **CI infrastructure**: Headless Chrome/Firefox WebGPU testing
6. **Shared constants**: Single source of truth for magic numbers (binding 120, ring size, alignment)
7. **Feature flags documentation**: Clear docs on what each ApiTrait does and when it's safe to enable

### 30. How does the instance batching interact with Godot's existing render list?

The batching is a lookahead in `_render_list_template` (Forward Mobile). After processing a draw, it peeks ahead at the next N draws checking if they share:
- Same mesh surface (same variant for the current pass)
- Same material uniform set
- Same LOD index
- Same cull variant
- Same pipeline specialization (light counts, features)
- Not transparent, not skinned, not multimesh, not particle

If all match, they merge into a single `drawIndexed(indexCount, batchCount)` call. The shader uses `draw_call.instance_index + gl_InstanceIndex` to index into per-instance data.

This works because Godot's render list is already sorted by shader → material → geometry, making same-state draws naturally adjacent.

### 31. How does the skeleton atlas integrate with the existing skeleton system?

When `supports_buffer_direct_write()` returns true (WebGPU only), the mesh storage system uses a shared GPU buffer instead of per-skeleton buffers:

- `_update_dirty_skeletons()` has two paths: atlas (new) and legacy (original)
- Atlas path: memcpy all dirty skeleton data into CPU mirror, track dirty range, single `buffer_update_direct()`
- Shader: `bone_offset` push constant (repurposed padding field) indexes into the atlas
- Skeleton compute shader adds `bone_offset` to all bone lookups

The atlas grows with power-of-two strategy (min 64KB). The uniform set is rebuilt lazily on buffer reallocation.

### 32. Is there a test suite? How is correctness validated?

Currently, there is no automated test suite for the WebGPU backend. Correctness is validated through:
- Manual testing across Chrome, Firefox, and Safari
- Stress test scenes (20k instances, 32 lights, 60k shared-material objects)
- The browser's own WebGPU validation layer (catches spec violations)
- Tint's built-in SPIR-V and WGSL validation passes
- Performance counter monitoring for unexpected behavior

For upstream, we'd need:
- Headless browser CI with WebGPU (Chrome --headless + SwiftShader)
- Shader corpus tests (convert all built-in shaders, validate WGSL output)
- Screenshot comparison tests across browsers
- Resource lifecycle stress tests (rapid create/destroy cycles)
