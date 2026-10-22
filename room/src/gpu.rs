//! wgpu-based GPU rendering state for the Doom frame buffer.
//!
//! [`GpuState`] encapsulates all wgpu resources needed to upload and display
//! the 640 × 400 pixel Doom frame buffer produced by the engine's
//! `I_FinishUpdate` function.
//!
//! ## Rendering pipeline
//!
//! Each frame, the workflow is:
//!
//! 1. **Upload** – `DG_ScreenBuffer` (raw BGRA8 pixels) is written into a
//!    wgpu [`Texture`] via [`Queue::write_texture`].
//! 2. **Render** – A single fullscreen-quad draw call samples the texture and
//!    outputs to the swapchain surface.
//!
//! The WGSL shaders are embedded at compile time as string literals so the
//! binary is self-contained (no external shader files required).
//!
//! ## Pixel format
//!
//! The Doom engine's `I_FinishUpdate` function (in `i_video.c`) writes pixels
//! in BGRA byte order:
//!
//! | Byte offset | Channel |
//! |-------------|---------|
//! | 0 | Blue |
//! | 1 | Green |
//! | 2 | Red |
//! | 3 | Alpha (unused, always 0) |
//!
//! The intermediate [`Texture`] uses [`TextureFormat::Bgra8Unorm`] to match
//! this layout exactly, avoiding any CPU-side byte-swapping.
//!
//! [`Texture`]: wgpu::Texture
//! [`TextureFormat::Bgra8Unorm`]: wgpu::TextureFormat::Bgra8Unorm
//! [`Queue::write_texture`]: wgpu::Queue::write_texture

use std::sync::Arc;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::doom::doomgeneric::{DOOMGENERIC_RESX, DOOMGENERIC_RESY};

// ---------------------------------------------------------------------------
// WGSL shaders (embedded)
// ---------------------------------------------------------------------------

/// WGSL source for the fullscreen-quad vertex and fragment shaders.
///
/// The vertex shader generates a full-screen quad from the vertex index
/// without needing a vertex buffer.  The fragment shader samples the Doom
/// frame texture and writes it to the render target.
const SHADER_SOURCE: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

/// Generate a fullscreen quad (two triangles) from the vertex index.
///
/// Vertices are laid out as a counter-clockwise pair of triangles covering
/// clip space from (-1,-1) to (1,1).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // NDC positions for a fullscreen quad (two triangles, 6 vertices).
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    // UV coordinates: Doom's origin is top-left; wgpu NDC bottom-left.
    // Flip the V axis so the image is not upside-down.
    var tex_coords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    out.tex_coords = tex_coords[vi];
    return out;
}

@group(0) @binding(0) var doom_texture: texture_2d<f32>;
@group(0) @binding(1) var doom_sampler: sampler;

/// Sample the Doom frame texture and write the result to the render target with full opacity.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(doom_texture, doom_sampler, in.tex_coords);
    return vec4<f32>(color.rgb, 1.0);
}
"#;

/// World-space mesh. `uv` tiles inside the atlas rectangle carried on the vertex.
const WORLD_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) atlas: vec4<f32>,
    @location(3) light: f32,
    @location(4) tint: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) atlas: vec4<f32>,
    @location(2) light: f32,
    @location(3) tint: vec3<f32>,
};

@vertex
fn vs_world(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    out.atlas = in.atlas;
    out.light = in.light;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs_world(in: VertexOutput) -> @location(0) vec4<f32> {
    let tile = fract(in.uv);
    let atlas_uv = vec2<f32>(
        mix(in.atlas.x, in.atlas.z, tile.x),
        mix(in.atlas.y, in.atlas.w, tile.y),
    );
    // Level 0 avoids the huge derivative fract() creates at a tile edge,
    // which otherwise shimmers along walls and floors.
    let color = textureSampleLevel(atlas_tex, atlas_samp, atlas_uv, 0.0);
    if (color.a < 0.5) {
        discard;
    }
    return vec4<f32>(color.rgb * in.tint * in.light, 1.0);
}
"#;

/// Software frame with the 3D viewport punched out, so the status bar and
/// border stay while the triangle scene shows through.
const HUD_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct HudRect {
    rect: vec4<f32>,
};

@group(0) @binding(0) var doom_texture: texture_2d<f32>;
@group(0) @binding(1) var doom_sampler: sampler;
@group(1) @binding(0) var<uniform> hud: HudRect;

@vertex
fn vs_hud(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var tex_coords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    out.tex_coords = tex_coords[vi];
    return out;
}

@fragment
fn fs_hud(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = hud.rect;
    if (in.tex_coords.x >= r.x && in.tex_coords.x <= r.x + r.z
        && in.tex_coords.y >= r.y && in.tex_coords.y <= r.y + r.w) {
        discard;
    }
    let color = textureSample(doom_texture, doom_sampler, in.tex_coords);
    return vec4<f32>(color.rgb, 1.0);
}
"#;

/// Path-traced level mesh. The sky and Brightmaps Plus masks are the lights,
/// carried through six diffuse bounces. Sprites and the weapon stay rasterized.
const RAY_SHADER: &str = r#"
enable wgpu_ray_query;

struct RayCam {
    view_proj: mat4x4<f32>,
    origin_aspect: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    viewport: vec4<f32>,
    // x: emissive scale, y: environment scale, z: bounce count.
    light: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

struct Surf {
    pos: vec3<f32>,
    n: vec3<f32>,
    albedo: vec3<f32>,
    light: f32,
    emit: f32,
    sky: bool,
    alpha: f32,
    hit: bool,
};

const VERTEX_FLOATS: u32 = 14u;
const FORCE_OPAQUE: u32 = 1u;
const SAMPLES: u32 = 4u;
const MAX_BOUNCES: i32 = 8;
const RAY_BIAS: f32 = 0.08;
// Same sun as the octree tracer: direction toward the light, and its radiance.
const SUN_TO = vec3<f32>(0.798, 0.469, -0.376);
const SUN_RGB = vec3<f32>(2.4, 2.25, 2.0);

@group(0) @binding(0) var<uniform> cam: RayCam;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var scene_tlas: acceleration_structure;
@group(0) @binding(4) var<storage, read> verts: array<f32>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var emit_tex: texture_2d<f32>;
@group(0) @binding(7) var emit_samp: sampler;

@vertex
fn vs_ray(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    return out;
}

fn attr(index: u32, comp: u32) -> f32 {
    return verts[index * VERTEX_FLOATS + comp];
}

fn bary_mix(i0: u32, i1: u32, i2: u32, comp: u32, b: vec3<f32>) -> f32 {
    return attr(i0, comp) * b.x + attr(i1, comp) * b.y + attr(i2, comp) * b.z;
}

fn hash_u(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = hash_u(*state);
    return f32(*state) / 4294967296.0;
}

fn sky_light(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.45 + 0.35, 0.0, 1.0);
    return mix(vec3<f32>(0.66, 0.74, 0.80), vec3<f32>(0.32, 0.48, 0.74), t) * 1.5;
}

fn hemisphere(n: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand(state);
    let r2 = rand(state);
    let phi = 6.2831853 * r1;
    let r = sqrt(r2);
    let x = r * cos(phi);
    let z = r * sin(phi);
    let y = sqrt(max(0.0, 1.0 - r2));
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(helper, n));
    let b = cross(n, t);
    return normalize(t * x + n * y + b * z);
}

fn trace(origin: vec3<f32>, dir: vec3<f32>) -> Surf {
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene_tlas,
        RayDesc(FORCE_OPAQUE, 0xFFu, 0.02, 20000.0, origin, dir),
    );
    while (rayQueryProceed(&rq)) {
        rayQueryConfirmIntersection(&rq);
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    var surf: Surf;
    surf.pos = origin;
    surf.n = vec3<f32>(0.0, 1.0, 0.0);
    surf.albedo = vec3<f32>(0.0);
    surf.light = 0.0;
    surf.emit = 0.0;
    surf.sky = false;
    surf.alpha = 0.0;
    surf.hit = false;
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return surf;
    }
    let prim = hit.primitive_index * 3u;
    let i0 = indices[prim];
    let i1 = indices[prim + 1u];
    let i2 = indices[prim + 2u];
    let b = vec3<f32>(
        1.0 - hit.barycentrics.x - hit.barycentrics.y,
        hit.barycentrics.x,
        hit.barycentrics.y,
    );
    let p0 = vec3<f32>(attr(i0, 0u), attr(i0, 1u), attr(i0, 2u));
    let p1 = vec3<f32>(attr(i1, 0u), attr(i1, 1u), attr(i1, 2u));
    let p2 = vec3<f32>(attr(i2, 0u), attr(i2, 1u), attr(i2, 2u));
    var n = normalize(cross(p1 - p0, p2 - p0));
    if (dot(n, dir) > 0.0) {
        n = -n;
    }
    let uv = vec2<f32>(bary_mix(i0, i1, i2, 3u, b), bary_mix(i0, i1, i2, 4u, b));
    let atlas = vec4<f32>(attr(i0, 5u), attr(i0, 6u), attr(i0, 7u), attr(i0, 8u));
    let tile = fract(uv);
    let atlas_uv = vec2<f32>(mix(atlas.x, atlas.z, tile.x), mix(atlas.y, atlas.w, tile.y));
    let tex = textureSampleLevel(atlas_tex, atlas_samp, atlas_uv, 0.0);
    let tint = vec3<f32>(
        bary_mix(i0, i1, i2, 10u, b),
        bary_mix(i0, i1, i2, 11u, b),
        bary_mix(i0, i1, i2, 12u, b),
    );
    surf.pos = origin + dir * hit.t;
    surf.n = n;
    surf.albedo = tex.rgb * tint;
    surf.light = bary_mix(i0, i1, i2, 9u, b);
    surf.emit = textureSampleLevel(emit_tex, emit_samp, atlas_uv, 0.0).r;
    surf.sky = bary_mix(i0, i1, i2, 13u, b) > 0.5;
    surf.alpha = tex.a;
    surf.hit = true;
    return surf;
}

// One shadow ray. The sky sphere is the only thing that counts as the sun.
fn sun_ndotl(pos: vec3<f32>, n: vec3<f32>) -> f32 {
    let ndl = dot(n, SUN_TO);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let blocker = trace(pos + n * RAY_BIAS, SUN_TO);
    if (blocker.sky) {
        return ndl;
    }
    return 0.0;
}

@fragment
fn fs_ray(@builtin(position) frag: vec4<f32>) -> FragOut {
    let vp = cam.viewport;
    let ndc_x = ((frag.x - vp.x) / vp.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((frag.y - vp.y) / vp.w) * 2.0;
    let aspect = max(cam.origin_aspect.w, 0.01);
    let view_dir = normalize(
        cam.right.xyz * ndc_x + cam.up.xyz * (ndc_y / aspect) + cam.forward.xyz
    );
    let eye = cam.origin_aspect.xyz;
    let primary = trace(eye, view_dir);
    if (!primary.hit) {
        return FragOut(1.0, vec4<f32>(0.02, 0.02, 0.03, 1.0));
    }
    if (primary.alpha < 0.5) {
        discard;
    }
    let clip = cam.view_proj * vec4<f32>(primary.pos, 1.0);
    let depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
    if (primary.sky) {
        return FragOut(depth, vec4<f32>(primary.albedo, 1.0));
    }

    // The sun and the sky light every surface that can see them. Brightmap
    // masks add the authored lamps. Sector brightness is not a light.
    var radiance = vec3<f32>(0.0);
    let frame = u32(cam.right.w);
    let px = u32(frag.x) + u32(frag.y) * 4096u;
    let emit_scale = cam.light.x;
    let env_scale = cam.light.y;
    let bounce_limit = clamp(i32(cam.light.z), 1, MAX_BOUNCES);
    for (var s = 0u; s < SAMPLES; s = s + 1u) {
        var rng = hash_u(px + frame * 9781u + s * 101u);
        var throughput = vec3<f32>(1.0);
        var surf = primary;
        var dir = view_dir;
        for (var bounce = 0; bounce < MAX_BOUNCES; bounce = bounce + 1) {
            if (bounce >= bounce_limit) {
                break;
            }
            if (surf.sky) {
                radiance += throughput * sky_light(dir) * env_scale;
                break;
            }
            if (surf.emit > 0.0) {
                radiance += throughput * surf.albedo * surf.emit * emit_scale;
            }
            let ndl = sun_ndotl(surf.pos, surf.n);
            if (ndl > 0.0) {
                radiance += throughput * surf.albedo * SUN_RGB * ndl * env_scale;
            }
            // Open sky above this point. Floors take most of it; walls take a little.
            let above = trace(surf.pos + surf.n * RAY_BIAS, vec3<f32>(0.0, 1.0, 0.0));
            if (above.sky) {
                let facing = max(surf.n.y, 0.0) * 0.5 + 0.25;
                radiance += throughput * surf.albedo * sky_light(vec3<f32>(0.0, 1.0, 0.0)) * facing * env_scale;
            }
            if (bounce + 1 >= bounce_limit) {
                break;
            }
            throughput = throughput * surf.albedo;
            dir = hemisphere(surf.n, &rng);
            let next = trace(surf.pos + surf.n * RAY_BIAS, dir);
            if (!next.hit || next.alpha < 0.5) {
                break;
            }
            surf = next;
        }
    }
    radiance = radiance / f32(SAMPLES);
    return FragOut(depth, vec4<f32>(radiance, 1.0));
}
"#;

struct WorldGpu {
    atlas: wgpu::Texture,
    emission: wgpu::Texture,
    atlas_size: u32,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    weapon_pipeline: wgpu::RenderPipeline,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    vertex_cap: u64,
    index_cap: u64,
    /// Hardware ray queries are available on this device.
    ray_query: bool,
    ray: Option<RayGpu>,
}

/// Sliders for the path tracer.
pub(crate) struct TraceSettings {
    /// Photons per masked texel. 4 matches the octree tracer's `texture_emission`.
    pub(crate) emission: f32,
    /// Scale on the sun and the sky.
    pub(crate) environment: f32,
    pub(crate) bounces: i32,
    /// Tallest path-traced image, in pixels. 720 is 720p for a 16:9 view.
    pub(crate) height: u32,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self {
            emission: 40.0,
            environment: 3.2,
            bounces: 2,
            height: 720,
        }
    }
}

struct RayImage {
    /// This frame's path-traced sample, before it is averaged.
    _color: wgpu::Texture,
    view: wgpu::TextureView,
    /// Running average of samples taken while the camera is still.
    _history: wgpu::Texture,
    /// Average including this frame. Sprites are drawn here after the copy
    /// into history, so they do not smear across frames.
    _combined: wgpu::Texture,
    combined: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    bind: wgpu::BindGroup,
    accum_bind: wgpu::BindGroup,
    size: (u32, u32),
}

struct AccGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
}

struct BlitGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

struct RayGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    blas: wgpu::Blas,
    tlas: wgpu::Tlas,
    /// Vertex and index capacities the acceleration structure was sized for.
    sized_for: (u32, u32),
}

// ---------------------------------------------------------------------------
// GpuState
// ---------------------------------------------------------------------------

/// All wgpu resources required to render the Doom frame buffer to a window.
pub struct GpuState {
    /// The wgpu surface tied to the OS window.
    surface: wgpu::Surface<'static>,
    /// The logical GPU device.
    device: wgpu::Device,
    /// The command queue used to submit GPU work.
    queue: wgpu::Queue,
    /// Current swapchain configuration (updated on resize).
    config: wgpu::SurfaceConfiguration,
    /// Intermediate texture that receives the Doom screen pixels each frame.
    doom_texture: wgpu::Texture,
    /// Bind group binding `doom_texture` and its sampler to the shader.
    bind_group: wgpu::BindGroup,
    /// The render pipeline executing the fullscreen-quad blit.
    render_pipeline: wgpu::RenderPipeline,
    /// Software frame with the 3D view discarded, drawn on top of the mesh.
    hud_pipeline: wgpu::RenderPipeline,
    hud_bind_group: wgpu::BindGroup,
    hud_rect_group: wgpu::BindGroup,
    hud_uniform: wgpu::Buffer,
    /// Triangle scene resources, created on the first level frame.
    world: Option<WorldGpu>,
    /// The device was created with [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`].
    ray_query: bool,
    /// Emissive scale, environment scale, bounce count, and ray-target height.
    trace: TraceSettings,
    /// Pixel size of the path-traced image last frame.
    traced: (u32, u32),
    /// Samples already averaged into the history image.
    accum_count: u32,
    accum_key: Option<AccumKey>,
    /// Offscreen path-traced view, capped by [`TraceSettings::height`].
    ray_image: Option<RayImage>,
    blit: BlitGpu,
    accum: AccGpu,
    #[cfg(not(target_arch = "wasm32"))]
    ui: crate::trace_ui::TraceUi,
}

/// Constructors and frame-time operations for [`GpuState`].
impl GpuState {
    /// Create a new [`GpuState`] for the given window.
    ///
    /// This performs the following wgpu setup:
    ///
    /// 1. Creates a [`wgpu::Instance`].
    /// 2. Creates a [`wgpu::Surface`] from the window.
    /// 3. Requests a [`wgpu::Adapter`] and [`wgpu::Device`] / [`wgpu::Queue`].
    /// 4. Configures the surface.
    /// 5. Creates the intermediate Doom-frame [`wgpu::Texture`].
    /// 6. Creates the render pipeline with the embedded WGSL shaders.
    ///
    /// # Errors
    ///
    /// Returns an error string if any wgpu initialisation step fails.
    pub fn new(window: Arc<Window>) -> Result<Self, String> {
        pollster::block_on(Self::new_async(window))
    }

    /// Asynchronous wgpu initialization, required for WebAssembly targets where blocking is forbidden.
    pub async fn new_async(window: Arc<Window>) -> Result<Self, String> {
        let mut size = window.inner_size();
        // winit's web backend reports 0×0 until its resize observer runs.
        // Configuring a 1×1 swapchain from that leaves a black canvas.
        #[cfg(target_arch = "wasm32")]
        if size.width < 2 || size.height < 2 {
            size = web_canvas_size();
        }
        // Use the default backends (Vulkan, Metal, DX12, OpenGL, WebGPU, WebGL2).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // `Surface<'static>` requires the window to outlive the surface.
        // `Arc<Window>` is `'static`, so this is sound.
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create_surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let adapter_info = adapter.get_info();
        log::info!(
            "GPU Adapter: backend={:?}, name={}, device_type={:?}",
            adapter_info.backend,
            adapter_info.name,
            adapter_info.device_type
        );

        #[cfg(target_arch = "wasm32")]
        let limits = wgpu::Limits::downlevel_webgl2_defaults();
        #[cfg(not(target_arch = "wasm32"))]
        let limits = wgpu::Limits::default();

        #[cfg(target_arch = "wasm32")]
        let want_rays = false;
        #[cfg(not(target_arch = "wasm32"))]
        let want_rays = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

        let ray_limits = if want_rays {
            limits
                .clone()
                .using_acceleration_structure_values(adapter.limits())
        } else {
            limits.clone()
        };
        // Ray queries are still an experimental wgpu feature. The Metal backend
        // on this machine implements them; agreeing to the token is required
        // before the device will enable `EXPERIMENTAL_RAY_QUERY`.
        let experimental = if want_rays {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::default()
        };
        let ray_features = if want_rays {
            wgpu::Features::EXPERIMENTAL_RAY_QUERY
        } else {
            wgpu::Features::empty()
        };

        let requested = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("doom_device"),
                required_features: ray_features,
                required_limits: ray_limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: experimental,
                trace: wgpu::Trace::Off,
            })
            .await;
        let (device, queue, ray_query) = match requested {
            Ok(pair) => (pair.0, pair.1, want_rays),
            Err(e) if want_rays => {
                log::warn!("hardware ray tracing unavailable, using the raster path: {e}");
                let pair = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("doom_device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: limits,
                        memory_hints: wgpu::MemoryHints::default(),
                        experimental_features: wgpu::ExperimentalFeatures::default(),
                        trace: wgpu::Trace::Off,
                    })
                    .await
                    .map_err(|err| format!("request_device: {err}"))?;
                (pair.0, pair.1, false)
            }
            Err(e) => return Err(format!("request_device: {e}")),
        };
        log::info!("hardware ray queries: {ray_query}");
        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB format so colours match the palette exactly.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        // Created with RGBA8Unorm to match the standard RGBA pixel format written by
        // the engine's I_FinishUpdate function. Supported universally across WebGPU and WebGL2.
        let doom_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("doom_frame_texture"),
            size: wgpu::Extent3d {
                width: DOOMGENERIC_RESX as u32,
                height: DOOMGENERIC_RESY as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let doom_texture_view = doom_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let doom_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("doom_sampler"),
            // Use nearest-neighbour filtering to preserve pixel sharpness.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("doom_bind_group_layout"),
            entries: &[
                // Binding 0: the Doom frame texture.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 1: the texture sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&doom_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&doom_sampler),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("doom_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("doom_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("doom_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffer; positions come from vertex index.
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("doom_hud_shader"),
            source: wgpu::ShaderSource::Wgsl(HUD_SHADER.into()),
        });
        let hud_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_hud_uniform"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hud_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("doom_hud_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let hud_rect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("doom_hud_rect_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let hud_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_hud_bind_group"),
            layout: &hud_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&doom_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&doom_sampler),
                },
            ],
        });
        let hud_rect_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_hud_rect_group"),
            layout: &hud_rect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: hud_uniform.as_entire_binding(),
            }],
        });
        let hud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("doom_hud_pipeline_layout"),
            bind_group_layouts: &[Some(&hud_layout), Some(&hud_rect_layout)],
            immediate_size: 0,
        });
        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("doom_hud_pipeline"),
            layout: Some(&hud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: Some("vs_hud"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: Some("fs_hud"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let blit = create_blit(&device, surface_format);
        let accum = create_accum(&device, surface_format);
        #[cfg(not(target_arch = "wasm32"))]
        let ui = crate::trace_ui::TraceUi::new(&device, surface_format, window);
        Ok(Self {
            surface,
            device,
            queue,
            config,
            doom_texture,
            bind_group,
            render_pipeline,
            hud_pipeline,
            hud_bind_group,
            hud_rect_group,
            hud_uniform,
            world: None,
            ray_query,
            trace: TraceSettings::default(),
            traced: (0, 0),
            accum_count: 0,
            accum_key: None,
            ray_image: None,
            blit,
            accum,
            #[cfg(not(target_arch = "wasm32"))]
            ui,
        })
    }

    /// Reconfigure the surface after a window resize.
    ///
    /// This must be called whenever the window's inner size changes, before
    /// the next call to [`render`].
    ///
    /// [`render`]: Self::render
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Upload `pixels` to the Doom frame texture and blit it to the surface.
    ///
    /// `pixels` must be exactly `DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4`
    /// bytes in BGRA8 format.
    ///
    /// # Errors
    ///
    /// Returns an error string if the swapchain texture cannot be acquired
    /// (e.g. the surface is lost or the window is minimised).
    pub fn render(&mut self, pixels: &[u8]) -> Result<(), String> {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.doom_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((DOOMGENERIC_RESX * 4) as u32),
                rows_per_image: Some(DOOMGENERIC_RESY as u32),
            },
            wgpu::Extent3d {
                width: DOOMGENERIC_RESX as u32,
                height: DOOMGENERIC_RESY as u32,
                depth_or_array_layers: 1,
            },
        );
        let mut reconfigured = false;
        let output = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => break t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    // Surface is suboptimal but still usable; present this frame.
                    break t;
                }
                wgpu::CurrentSurfaceTexture::Timeout => {
                    return Err("surface timeout".to_owned());
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    // Window is minimised or hidden; skip this frame.
                    return Ok(());
                }
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    // The canvas size often changes after the first configure on
                    // the web. Rebuild the swapchain once and present this frame.
                    if reconfigured || self.config.width == 0 || self.config.height == 0 {
                        return Err("surface lost or outdated".to_owned());
                    }
                    reconfigured = true;
                    self.surface.configure(&self.device, &self.config);
                }
                // Handle future variants without breaking the build.
                _ => {
                    return Err("surface error: unknown status".to_owned());
                }
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("doom_frame_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            // Draw 6 vertices (2 triangles) without a vertex buffer.
            render_pass.draw(0..6, 0..1);
        }
        self.paint_ui(&mut encoder, &view);
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Draw a triangle scene, then composite the software HUD on top.
    pub fn render_world(&mut self, pixels: &[u8], scene: &crate::hw::Scene) -> Result<(), String> {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.doom_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((DOOMGENERIC_RESX * 4) as u32),
                rows_per_image: Some(DOOMGENERIC_RESY as u32),
            },
            wgpu::Extent3d {
                width: DOOMGENERIC_RESX as u32,
                height: DOOMGENERIC_RESY as u32,
                depth_or_array_layers: 1,
            },
        );
        self.queue
            .write_buffer(&self.hud_uniform, 0, bytemuck::cast_slice(&scene.view_rect));

        let atlas = scene
            .atlas
            .as_ref()
            .map(|(size, rgba, emit)| (*size, rgba.as_slice(), emit.as_slice()));
        self.ensure_world(atlas);
        let rect = scene.view_rect;
        let surf_w = self.config.width as f32;
        let surf_h = self.config.height as f32;
        let vx = (rect[0] * surf_w).round().clamp(0.0, surf_w - 1.0);
        let vy = (rect[1] * surf_h).round().clamp(0.0, surf_h - 1.0);
        let vw = (rect[2] * surf_w).round().clamp(1.0, surf_w - vx);
        let vh = (rect[3] * surf_h).round().clamp(1.0, surf_h - vy);
        if let Some(world) = self.world.as_mut() {
            upload_mesh(world, &self.device, &self.queue, scene);
            self.queue
                .write_buffer(&world.uniform, 0, bytemuck::cast_slice(&scene.view_proj));
        }
        let (rw, rh) = ray_extent(vw, vh, self.trace.height);
        self.traced = (rw, rh);
        let trace = self.prepare_rays(scene, 0.0, 0.0, rw as f32, rh as f32);
        if trace {
            self.ensure_ray_image(rw, rh);
        } else if let Some(world) = self.world.as_mut() {
            if self.config.width != world.depth_size.0 || self.config.height != world.depth_size.1 {
                let (depth, depth_view) =
                    depth_target(&self.device, self.config.width, self.config.height);
                world.depth = depth;
                world.depth_view = depth_view;
                world.depth_size = (self.config.width, self.config.height);
            }
        }

        let mut reconfigured = false;
        let output = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => break t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => break t,
                wgpu::CurrentSurfaceTexture::Timeout => return Err("surface timeout".to_owned()),
                wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    if reconfigured || self.config.width == 0 || self.config.height == 0 {
                        return Err("surface lost or outdated".to_owned());
                    }
                    reconfigured = true;
                    self.surface.configure(&self.device, &self.config);
                }
                _ => return Err("surface error: unknown status".to_owned()),
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        if trace {
            let weight = self.begin_accumulation(scene, rw, rh);
            self.queue.write_buffer(
                &self.accum.uniform,
                0,
                bytemuck::cast_slice(&[weight, 0.0, 0.0, 0.0]),
            );
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("doom_world_encoder"),
            });

        if trace {
            if let Some(world) = self.world.as_mut() {
                build_rays(world, scene, &mut encoder);
            }
        }

        if trace {
            {
            let image = self.ray_image.as_ref().expect("ray image");
            let world = self.world.as_ref().expect("world");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &image.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, rw as f32, rh as f32, 0.0, 1.0);
            draw_scene(&mut pass, world, scene, true);
            }
            {
            let image = self.ray_image.as_ref().expect("ray image");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_accum"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image.combined,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.accum.pipeline);
            pass.set_bind_group(0, &image.accum_bind, &[]);
            pass.draw(0..6, 0..1);
            }
            let image = self.ray_image.as_ref().expect("ray image");
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &image._combined,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &image._history,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: rw,
                    height: rh,
                    depth_or_array_layers: 1,
                },
            );
            {
            let image = self.ray_image.as_ref().expect("ray image");
            let world = self.world.as_ref().expect("world");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_sprites"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image.combined,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &image.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, rw as f32, rh as f32, 0.0, 1.0);
            draw_overlays(&mut pass, world, scene);
            }
            let image = self.ray_image.as_ref().expect("ray image");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
            pass.set_scissor_rect(vx as u32, vy as u32, vw as u32, vh as u32);
            pass.set_pipeline(&self.blit.pipeline);
            pass.set_bind_group(0, &image.bind, &[]);
            pass.draw(0..6, 0..1);
        } else if let Some(world) = self.world.as_ref() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_world_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &world.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
            pass.set_scissor_rect(vx as u32, vy as u32, vw as u32, vh as u32);
            draw_scene(&mut pass, world, scene, false);
        }

        {
            let mut hud = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_hud_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            hud.set_pipeline(&self.hud_pipeline);
            hud.set_bind_group(0, &self.hud_bind_group, &[]);
            hud.set_bind_group(1, &self.hud_rect_group, &[]);
            hud.draw(0..6, 0..1);
        }
        self.paint_ui(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    fn ensure_ray_image(&mut self, width: u32, height: u32) {
        if self
            .ray_image
            .as_ref()
            .is_some_and(|image| image.size == (width, height))
        {
            return;
        }
        let format = self.config.format;
        let color = ray_target(&self.device, width, height, format, false);
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let history = ray_target(&self.device, width, height, format, true);
        let history_view = history.create_view(&wgpu::TextureViewDescriptor::default());
        let combined = ray_target(&self.device, width, height, format, true);
        let combined_view = combined.create_view(&wgpu::TextureViewDescriptor::default());
        let (depth, depth_view) = depth_target(&self.device, width, height);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_ray_blit_group"),
            layout: &self.blit.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&combined_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit.sampler),
                },
            ],
        });
        let accum_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_ray_accum_group"),
            layout: &self.accum.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.accum.uniform.as_entire_binding(),
                },
            ],
        });
        self.accum_count = 0;
        self.accum_key = None;
        self.ray_image = Some(RayImage {
            _color: color,
            view,
            _history: history,
            _combined: combined,
            combined: combined_view,
            _depth: depth,
            depth_view,
            bind,
            accum_bind,
            size: (width, height),
        });
    }

    /// `true` when this frame continues the running average. A camera or
    /// lighting change starts the average over.
    fn begin_accumulation(&mut self, scene: &crate::hw::Scene, width: u32, height: u32) -> f32 {
        let key = AccumKey::now(scene, &self.trace, width, height);
        let still = self.accum_key.as_ref().is_some_and(|prev| prev.same(&key));
        if !still {
            self.accum_count = 0;
            self.accum_key = Some(key);
        }
        let weight = 1.0 / (self.accum_count + 1) as f32;
        self.accum_count += 1;
        weight
    }

    fn paint_ui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let GpuState {
                ui,
                device,
                queue,
                config,
                trace,
                traced,
                accum_count,
                ..
            } = self;
            ui.paint(
                device,
                queue,
                encoder,
                view,
                config.width,
                config.height,
                trace,
                *traced,
                *accum_count,
            );
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (encoder, view);
        }
    }

    /// Feed a window event to the path-tracer panel. `true` means egui used it.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ui.on_event(event)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = event;
            false
        }
    }

    /// Upload the ray-query camera and make sure the acceleration structure
    /// covers the current mesh buffers. Returns whether this frame should trace.
    fn prepare_rays(
        &mut self,
        scene: &crate::hw::Scene,
        vx: f32,
        vy: f32,
        vw: f32,
        vh: f32,
    ) -> bool {
        let light = [
            self.trace.emission,
            self.trace.environment,
            self.trace.bounces as f32,
            0.0,
        ];
        let GpuState {
            world: Some(world),
            device,
            queue,
            config,
            ..
        } = self
        else {
            return false;
        };
        if !world.ray_query || scene.opaque_indices < 3 || scene.verts.is_empty() {
            return false;
        }
        if let Err(err) = sync_ray(world, device, config.format) {
            log::warn!("ray tracing disabled: {err}");
            world.ray_query = false;
            world.ray = None;
            return false;
        }
        let Some(ray) = world.ray.as_ref() else {
            return false;
        };
        let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as f32;
        let uniform = RayUniform {
            view_proj: scene.view_proj,
            origin_aspect: [scene.eye[0], scene.eye[1], scene.eye[2], scene.aspect],
            right: [scene.right[0], scene.right[1], scene.right[2], frame],
            up: [scene.up[0], scene.up[1], scene.up[2], 0.0],
            forward: [scene.forward[0], scene.forward[1], scene.forward[2], 0.0],
            viewport: [vx, vy, vw, vh],
            light,
        };
        queue.write_buffer(&ray.uniform, 0, bytemuck::bytes_of(&uniform));
        true
    }

    fn ensure_world(&mut self, atlas: Option<(u32, &[u8], &[u8])>) {
        let Some((size, rgba, emit)) = atlas else {
            return;
        };
        if self.world.as_ref().is_some_and(|w| w.atlas_size == size) {
            let world = self.world.as_ref().unwrap();
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &world.atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &world.emission,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                emit,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
            return;
        }

        let atlas_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("doom_atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        let emission_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("doom_emission"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &emission_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            emit,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("doom_atlas_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("doom_world_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_camera"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_world_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("doom_world_shader"),
                source: wgpu::ShaderSource::Wgsl(WORLD_SHADER.into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("doom_world_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let buffers = [vertex_layout()];
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("doom_world_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_world"),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_world"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let weapon_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("doom_weapon_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_world"),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_world"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Always)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let (depth, depth_view) = depth_target(&self.device, self.config.width, self.config.height);
        let (vertex_usage, index_usage) = mesh_usage(self.ray_query);
        let vertices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_vertices"),
            size: 64 * 1024,
            usage: vertex_usage,
            mapped_at_creation: false,
        });
        let indices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_indices"),
            size: 64 * 1024,
            usage: index_usage,
            mapped_at_creation: false,
        });
        self.world = Some(WorldGpu {
            atlas: atlas_tex,
            emission: emission_tex,
            atlas_size: size,
            bind_group,
            pipeline,
            weapon_pipeline,
            depth,
            depth_view,
            depth_size: (self.config.width, self.config.height),
            uniform,
            vertices,
            indices,
            vertex_cap: 64 * 1024,
            index_cap: 64 * 1024,
            ray_query: self.ray_query,
            ray: None,
        });
    }
}

const VERTEX_ATTRS: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 20,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 36,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 40,
        shader_location: 4,
    },
];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::hw::Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRS,
    }
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

const BLIT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@fragment
fn fs_blit(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, in.uv);
}
"#;

fn ray_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    copy: bool,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if copy {
        usage |= wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
    }
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("doom_ray_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumKey {
    eye: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    light: [f32; 4],
    size: [f32; 2],
}

impl AccumKey {
    fn now(scene: &crate::hw::Scene, trace: &TraceSettings, width: u32, height: u32) -> Self {
        Self {
            eye: scene.eye,
            right: scene.right,
            up: scene.up,
            forward: scene.forward,
            light: [
                trace.emission,
                trace.environment,
                trace.bounces as f32,
                trace.height as f32,
            ],
            size: [width as f32, height as f32],
        }
    }

    fn same(&self, other: &Self) -> bool {
        let a = bytemuck::bytes_of(self);
        let b = bytemuck::bytes_of(other);
        a.chunks(4).zip(b.chunks(4)).all(|(x, y)| {
            let x = f32::from_le_bytes(x.try_into().unwrap());
            let y = f32::from_le_bytes(y.try_into().unwrap());
            (x - y).abs() <= 1e-4
        })
    }
}

const ACC_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

struct AccWeight {
    weight: f32,
};

@group(0) @binding(0) var sample_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> acc: AccWeight;

@vertex
fn vs_acc(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_acc(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(frag.xy);
    let fresh = textureLoad(sample_tex, p, 0);
    let old = textureLoad(history_tex, p, 0);
    return vec4<f32>(mix(old.rgb, fresh.rgb, acc.weight), 1.0);
}
"#;

fn create_accum(device: &wgpu::Device, format: wgpu::TextureFormat) -> AccGpu {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doom_ray_accum_shader"),
        source: wgpu::ShaderSource::Wgsl(ACC_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("doom_ray_accum_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("doom_ray_accum_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("doom_ray_accum_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_acc"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_acc"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_ray_accum_weight"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    AccGpu {
        pipeline,
        layout,
        uniform,
    }
}

fn ray_extent(view_w: f32, view_h: f32, max_h: u32) -> (u32, u32) {
    let view_w = view_w.max(1.0);
    let view_h = view_h.max(1.0);
    let max_h = max_h.clamp(120, 1080);
    let max_w = ((max_h as f32) * (16.0 / 9.0)).round().max(1.0) as u32;
    let mut h = (view_h.round() as u32).clamp(1, max_h);
    let mut w = ((h as f32) * (view_w / view_h)).round().max(1.0) as u32;
    if w > max_w {
        w = max_w;
        h = ((w as f32) * (view_h / view_w)).round().max(1.0) as u32;
    }
    (w.max(1), h.max(1))
}

fn draw_scene<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    world: &'a WorldGpu,
    scene: &crate::hw::Scene,
    trace: bool,
) {
    pass.set_vertex_buffer(0, world.vertices.slice(..));
    pass.set_index_buffer(world.indices.slice(..), wgpu::IndexFormat::Uint32);
    let world_n = scene.world_indices.min(scene.indices.len() as u32);
    let total = scene.indices.len() as u32;
    if trace {
        let ray = world.ray.as_ref().expect("ray resources");
        pass.set_pipeline(&ray.pipeline);
        pass.set_bind_group(0, &ray.bind_group, &[]);
        pass.draw(0..6, 0..1);
    } else if world_n > 0 {
        pass.set_pipeline(&world.pipeline);
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(0..world_n, 0, 0..1);
    }
    if !trace && total > world_n {
        pass.set_pipeline(&world.weapon_pipeline);
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(world_n..total, 0, 0..1);
    }
}

fn draw_overlays<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    world: &'a WorldGpu,
    scene: &crate::hw::Scene,
) {
    pass.set_vertex_buffer(0, world.vertices.slice(..));
    pass.set_index_buffer(world.indices.slice(..), wgpu::IndexFormat::Uint32);
    let world_n = scene.world_indices.min(scene.indices.len() as u32);
    let total = scene.indices.len() as u32;
    let sprites = scene.opaque_indices.min(world_n);
    if world_n > sprites {
        pass.set_pipeline(&world.pipeline);
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(sprites..world_n, 0, 0..1);
    }
    if total > world_n {
        pass.set_pipeline(&world.weapon_pipeline);
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(world_n..total, 0, 0..1);
    }
}

fn create_blit(device: &wgpu::Device, format: wgpu::TextureFormat) -> BlitGpu {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doom_ray_blit_shader"),
        source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("doom_ray_blit_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("doom_ray_blit_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("doom_ray_blit_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_blit"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_blit"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("doom_ray_blit_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
    BlitGpu {
        pipeline,
        layout,
        sampler,
    }
}

fn depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("doom_depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    (depth, view)
}

fn upload_mesh(
    world: &mut WorldGpu,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &crate::hw::Scene,
) {
    let v_bytes = bytemuck::cast_slice(&scene.verts);
    let i_bytes = bytemuck::cast_slice(&scene.indices);
    let (vertex_usage, index_usage) = mesh_usage(world.ray_query);
    if v_bytes.len() as u64 > world.vertex_cap {
        let size = (v_bytes.len() as u64).next_power_of_two().max(64 * 1024);
        world.vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_vertices"),
            size,
            usage: vertex_usage,
            mapped_at_creation: false,
        });
        world.vertex_cap = size;
    }
    if i_bytes.len() as u64 > world.index_cap {
        let size = (i_bytes.len() as u64).next_power_of_two().max(64 * 1024);
        world.indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_indices"),
            size,
            usage: index_usage,
            mapped_at_creation: false,
        });
        world.index_cap = size;
    }
    if !v_bytes.is_empty() {
        queue.write_buffer(&world.vertices, 0, v_bytes);
    }
    if !i_bytes.is_empty() {
        queue.write_buffer(&world.indices, 0, i_bytes);
    }
}

/// Packed [`crate::hw::Vertex`] size. The ray shader reads 14 floats per vertex.
const VERTEX_STRIDE: u64 = 56;
const _: () = assert!(std::mem::size_of::<crate::hw::Vertex>() as u64 == VERTEX_STRIDE);

const IDENTITY_3X4: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// Changes the bounce seed each frame so the path-traced noise moves.
static FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    view_proj: [[f32; 4]; 4],
    origin_aspect: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    viewport: [f32; 4],
    /// x: emissive scale, y: environment scale, z: bounce count.
    light: [f32; 4],
}

fn mesh_usage(ray: bool) -> (wgpu::BufferUsages, wgpu::BufferUsages) {
    let mut vertex = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
    let mut index = wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST;
    if ray {
        vertex |= wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE;
        index |= wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE;
    }
    (vertex, index)
}

fn tri_size(vertex_count: u32, index_count: u32) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_count.max(1),
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(index_count.max(3)),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    }
}

fn make_blas(device: &wgpu::Device, max_verts: u32, max_indices: u32) -> wgpu::Blas {
    device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("doom_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![tri_size(max_verts, max_indices)],
        },
    )
}

fn sync_ray(
    world: &mut WorldGpu,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> Result<(), String> {
    let max_verts = (world.vertex_cap / VERTEX_STRIDE).max(1) as u32;
    let max_indices = (world.index_cap / 4).max(3) as u32;
    if world
        .ray
        .as_ref()
        .is_some_and(|ray| ray.sized_for == (max_verts, max_indices))
    {
        return Ok(());
    }

    let kept = world.ray.take();
    let (layout, pipeline, uniform) = if let Some(ray) = kept {
        (ray.layout, ray.pipeline, ray.uniform)
    } else {
        create_ray_pipeline(device, format)?
    };
    let blas = make_blas(device, max_verts, max_indices);
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("doom_tlas"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let bind_group = ray_bind_group(
        device,
        &layout,
        &uniform,
        &world.atlas,
        &world.emission,
        &world.vertices,
        &world.indices,
        &tlas,
    );
    world.ray = Some(RayGpu {
        pipeline,
        layout,
        uniform,
        bind_group,
        blas,
        tlas,
        sized_for: (max_verts, max_indices),
    });
    Ok(())
}

fn create_ray_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> Result<(wgpu::BindGroupLayout, wgpu::RenderPipeline, wgpu::Buffer), String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doom_ray_shader"),
        source: wgpu::ShaderSource::Wgsl(RAY_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("doom_ray_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            storage_entry(4),
            storage_entry(5),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_ray_camera"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("doom_ray_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("doom_ray_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_ray"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_ray"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Always)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(err.to_string());
    }
    Ok((layout, pipeline, uniform))
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn ray_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    atlas: &wgpu::Texture,
    emission: &wgpu::Texture,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    tlas: &wgpu::Tlas,
) -> wgpu::BindGroup {
    let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
    let emit_view = emission.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("doom_ray_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
    let emit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("doom_emit_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("doom_ray_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&emit_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&emit_sampler),
            },
        ],
    })
}

fn build_rays(world: &mut WorldGpu, scene: &crate::hw::Scene, encoder: &mut wgpu::CommandEncoder) {
    let vert_count = scene.verts.len() as u32;
    let index_count = scene.opaque_indices;
    if vert_count == 0 || index_count < 3 {
        return;
    }
    let WorldGpu {
        ray,
        vertices,
        indices,
        ..
    } = world;
    let Some(ray) = ray.as_mut() else {
        return;
    };
    ray.tlas[0] = Some(wgpu::TlasInstance::new(
        &ray.blas,
        IDENTITY_3X4,
        0,
        0xff,
    ));
    let size = tri_size(vert_count, index_count);
    let entry = wgpu::BlasBuildEntry {
        blas: &ray.blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &size,
            vertex_buffer: vertices,
            first_vertex: 0,
            vertex_stride: VERTEX_STRIDE,
            index_buffer: Some(indices),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };
    encoder.build_acceleration_structures(std::iter::once(&entry), std::iter::once(&ray.tlas));
}

/// CSS size of the page canvas, used when winit has not reported a real size yet.
#[cfg(target_arch = "wasm32")]
fn web_canvas_size() -> PhysicalSize<u32> {
    let fallback = PhysicalSize::new(640, 400);
    let Some(window) = web_sys::window() else {
        return fallback;
    };
    let Some(document) = window.document() else {
        return fallback;
    };
    let Some(canvas) = document.get_element_by_id("doom-canvas") else {
        return fallback;
    };
    let width = canvas.client_width();
    let height = canvas.client_height();
    if width < 2 || height < 2 {
        fallback
    } else {
        PhysicalSize::new(width as u32, height as u32)
    }
}

#[cfg(test)]
mod ray_test {
    use super::*;

    /// One opaque triangle in front of the camera. The center pixel must come
    /// back white, which only happens if the ray query hits and samples the atlas.
    #[test]
    fn primary_ray_hits_a_triangle() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Some(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()
            else {
                eprintln!("no adapter; skipping ray test");
                return;
            };
            if !adapter
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            {
                eprintln!("adapter has no ray queries; skipping");
                return;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ray_test"),
                    required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    required_limits: wgpu::Limits::default()
                        .using_acceleration_structure_values(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("device");

            let format = wgpu::TextureFormat::Rgba8Unorm;
            let (layout, pipeline, uniform) =
                create_ray_pipeline(&device, format).expect("ray pipeline");

            // pos, uv, atlas, light, tint, pad. A white texel covers the whole triangle.
            // Pad is 1 so the triangle is sky emission. A non-emissive triangle
            // with nothing else in the scene would correctly stay black.
            let verts: [f32; 42] = [
                -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ];
            let indices: [u32; 3] = [0, 1, 2];
            let (vertex_usage, index_usage) = mesh_usage(true);
            let vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&verts) as u64,
                usage: vertex_usage,
                mapped_at_creation: true,
            });
            vertices
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&verts));
            vertices.unmap();
            let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&indices) as u64,
                usage: index_usage,
                mapped_at_creation: true,
            });
            index_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&indices));
            index_buf.unmap();

            let atlas = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255, 255, 255, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            let blas = make_blas(&device, 3, 3);
            let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: None,
                max_instances: 1,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            tlas[0] = Some(wgpu::TlasInstance::new(&blas, IDENTITY_3X4, 0, 0xff));
            let emission = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &emission,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8; 256],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &vertices, &index_buf, &tlas,
            );
            let cam = RayUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                origin_aspect: [0.0, 0.0, -2.0, 1.0],
                right: [1.0, 0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0, 0.0],
                forward: [0.0, 0.0, 1.0, 0.0],
                viewport: [0.0, 0.0, 1.0, 1.0],
                light: [4.0, 1.0, 6.0, 0.0],
            };
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&cam));

            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ray_test"),
            });
            let size = tri_size(3, 3);
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: VERTEX_STRIDE,
                    index_buffer: Some(&index_buf),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            };
            encoder.build_acceleration_structures([&entry], [&tlas]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            queue.submit([encoder.finish()]);
            let slice = read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            rx.recv().unwrap().expect("map");
            if let Some(err) = scope.pop().await {
                panic!("validation: {err}");
            }
            let px = slice.get_mapped_range()[0..4].to_vec();
            assert_eq!(px, [255, 255, 255, 255], "primary ray missed the triangle");
        });
    }

    #[test]
    fn emissive_texel_is_white() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Some(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()
            else {
                eprintln!("no adapter; skipping ray test");
                return;
            };
            if !adapter
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            {
                eprintln!("adapter has no ray queries; skipping");
                return;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ray_emit_test"),
                    required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    required_limits: wgpu::Limits::default()
                        .using_acceleration_structure_values(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("device");

            let format = wgpu::TextureFormat::Rgba8Unorm;
            let (layout, pipeline, uniform) =
                create_ray_pipeline(&device, format).expect("ray pipeline");

            // Not sky. White albedo, emission texel 255. The lamp term is
            // albedo * 1 * 4, then divided by the sample count, so the pixel
            // stays white. A missed emission sample stays black.
            let verts: [f32; 42] = [
                -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ];
            let indices: [u32; 3] = [0, 1, 2];
            let (vertex_usage, index_usage) = mesh_usage(true);
            let vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&verts) as u64,
                usage: vertex_usage,
                mapped_at_creation: true,
            });
            vertices
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&verts));
            vertices.unmap();
            let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&indices) as u64,
                usage: index_usage,
                mapped_at_creation: true,
            });
            index_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&indices));
            index_buf.unmap();

            let atlas = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255, 255, 255, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let emission = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut emit_row = [0u8; 256];
            emit_row[0] = 255;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &emission,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &emit_row,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            let blas = make_blas(&device, 3, 3);
            let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: None,
                max_instances: 1,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            tlas[0] = Some(wgpu::TlasInstance::new(&blas, IDENTITY_3X4, 0, 0xff));
            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &vertices, &index_buf, &tlas,
            );
            let cam = RayUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                origin_aspect: [0.0, 0.0, -2.0, 1.0],
                right: [1.0, 0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0, 0.0],
                forward: [0.0, 0.0, 1.0, 0.0],
                viewport: [0.0, 0.0, 1.0, 1.0],
                light: [4.0, 1.0, 6.0, 0.0],
            };
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&cam));

            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ray_emit_test"),
            });
            let size = tri_size(3, 3);
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: VERTEX_STRIDE,
                    index_buffer: Some(&index_buf),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            };
            encoder.build_acceleration_structures([&entry], [&tlas]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            queue.submit([encoder.finish()]);
            let slice = read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            rx.recv().unwrap().expect("map");
            if let Some(err) = scope.pop().await {
                panic!("validation: {err}");
            }
            let px = slice.get_mapped_range()[0..4].to_vec();
            assert!(
                px[0] > 200 && px[1] > 200 && px[2] > 200,
                "emissive texel did not light the surface: {px:?}"
            );
        });
    }

    /// The camera looks at the back of the triangle, which is the face Doom
    /// walls present. The hit sits on the lit texel of a 2x2 mask. A swapped
    /// back-face barycentric lands on an unlit texel and the pixel stays black.
    #[test]
    fn backface_hit_samples_the_lit_texel() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Some(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()
            else {
                eprintln!("no adapter; skipping ray test");
                return;
            };
            if !adapter
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            {
                eprintln!("adapter has no ray queries; skipping");
                return;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ray_backface_uv"),
                    required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    required_limits: wgpu::Limits::default()
                        .using_acceleration_structure_values(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("device");

            let format = wgpu::TextureFormat::Rgba8Unorm;
            let (layout, pipeline, uniform) =
                create_ray_pipeline(&device, format).expect("ray pipeline");

            // Center ray hits (0, 0, 0). Correct barycentrics land on UV
            // ~(0.08, 0.83), texel (0, 1). Swapped barycentrics land on
            // texel (1, 0), which is dark.
            let verts: [f32; 42] = [
                -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                2.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ];
            let indices: [u32; 3] = [0, 1, 2];
            let (vertex_usage, index_usage) = mesh_usage(true);
            let vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&verts) as u64,
                usage: vertex_usage,
                mapped_at_creation: true,
            });
            vertices
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&verts));
            vertices.unmap();
            let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of_val(&indices) as u64,
                usage: index_usage,
                mapped_at_creation: true,
            });
            index_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&indices));
            index_buf.unmap();

            let atlas = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8, 255, 255, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let mut mask = vec![0u8; 256 * 2];
            mask[256] = 255;
            let emission = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &emission,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mask,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
                wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            );

            let blas = make_blas(&device, 3, 3);
            let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: None,
                max_instances: 1,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            tlas[0] = Some(wgpu::TlasInstance::new(&blas, IDENTITY_3X4, 0, 0xff));

            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &vertices, &index_buf, &tlas,
            );
            let cam = RayUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                origin_aspect: [0.0, 0.0, -2.0, 1.0],
                right: [1.0, 0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0, 0.0],
                forward: [0.0, 0.0, 1.0, 0.0],
                viewport: [0.0, 0.0, 1.0, 1.0],
                light: [4.0, 0.0, 1.0, 0.0],
            };
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&cam));

            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ray_backface_uv"),
            });
            let size = tri_size(3, 3);
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: VERTEX_STRIDE,
                    index_buffer: Some(&index_buf),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            };
            encoder.build_acceleration_structures([&entry], [&tlas]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            queue.submit([encoder.finish()]);
            let slice = read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
            rx.recv().unwrap().expect("map");
            if let Some(err) = scope.pop().await {
                panic!("validation: {err}");
            }
            let px = slice.get_mapped_range()[0..4].to_vec();
            assert!(
                px[0] > 200 && px[1] > 200 && px[2] > 200,
                "backface hit missed the lit texel: {px:?}"
            );
        });
    }
}
