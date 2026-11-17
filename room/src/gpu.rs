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

/// Rasterized level mesh. Play mode draws this with the light map; bake mode
/// uses the ray pass instead when hardware ray queries are available.
const WORLD_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    // x: minimum incoming light; y: bake blend factor (0.0 = 100% OG Doom, 1.0 = 100% Baked); z: display ceiling; w: bypass bake (1.0 = bypass, 0.0 = use bake).
    fill: vec4<f32>,
    // x: normal strength; y: roughness scale; z: specular sheen; w: OG Doom distance light diminishing.
    pbr: vec4<f32>,
    // x: object/voxel bake blend factor (0.0 = 100% OG Doom, 1.0 = 100% Baked).
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var light_tex: texture_2d<f32>;
@group(0) @binding(4) var emit_tex: texture_2d<f32>;
@group(0) @binding(5) var pbr_tex: texture_2d<f32>;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) atlas: vec4<f32>,
    @location(3) light: f32,
    @location(4) tint: vec3<f32>,
    @location(5) light_uv: vec2<f32>,
    @location(6) surface: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) atlas: vec4<f32>,
    @location(2) light: f32,
    @location(3) tint: vec3<f32>,
    @location(4) light_uv: vec2<f32>,
    @location(5) @interpolate(flat) surface: f32,
    @location(6) world_pos: vec3<f32>,
    @location(7) view_dist: f32,
};

@vertex
fn vs_world(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    out.atlas = in.atlas;
    out.light = in.light;
    out.tint = in.tint;
    out.light_uv = in.light_uv;
    out.surface = in.surface;
    out.world_pos = in.pos;
    out.view_dist = out.clip_position.w;
    return out;
}

fn doom_diminish(light_in: vec3<f32>, dist: f32, strength: f32, min_floor: f32) -> vec3<f32> {
    if (strength <= 0.001) {
        return light_in;
    }
    let lum = dot(light_in, vec3<f32>(0.299, 0.587, 0.114));
    if (lum >= 1.0) {
        return light_in;
    }
    let startmap = (1.0 - clamp(lum, 0.0, 1.0)) * 60.0;
    let dist_scale = (1280.0 * strength) / max(dist, 16.0);
    let colormap_index = clamp(startmap - dist_scale, 0.0, 31.0);
    let factor = 1.0 - colormap_index / 31.0;
    let diminished = light_in * (factor / max(lum, 0.04));
    let result = max(diminished, vec3<f32>(min_floor * 0.5));
    return mix(light_in, result, clamp(strength, 0.0, 1.0));
}

fn cotangent_frame(N: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, N);
    let dp1perp = cross(N, dp1);
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(max(dot(T, T), dot(B, B)), 1e-10));
    return mat3x3<f32>(T * invmax, B * invmax, N);
}

@fragment
fn fs_world(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // Wall quads face their Doom sector clockwise. The opposite sidedef
    // can occupy this exact plane, so depth testing alone cannot select it.
    if (in.surface < -0.5 && front) {
        discard;
    }
    // Sprites and actor billboards (surface > 1.5) must never tile or wrap around.
    if (in.surface > 1.5) {
        if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) {
            discard;
        }
    }
    var tile = in.uv;
    if (in.surface <= 1.5) {
        tile = fract(in.uv);
    }
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
    var doom_light = vec3<f32>(in.light);
    doom_light = doom_diminish(doom_light, in.view_dist, camera.pbr.w, camera.fill.x);
    var incoming = doom_light;
    if (camera.fill.w < 0.5 && in.light_uv.x >= 0.0) {
        let dims = textureDimensions(light_tex);
        let coord = clamp(
            vec2<i32>(in.light_uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - vec2<i32>(1),
        );
        // Alpha stores the sample count. An incomplete or loading bake has
        // empty texels; keep sector lighting there instead of painting black.
        let baked = textureLoad(light_tex, coord, 0);
        if (baked.a > 0.0) {
            let baked_rgb = max(baked.rgb, vec3<f32>(in.light * camera.fill.x * 2.0));
            let blend = select(clamp(camera.fill.y, 0.0, 1.0), clamp(camera.extra.x, 0.0, 1.0), in.surface > 1.5);
            let baked_final = select(
                baked_rgb,
                doom_diminish(clamp(baked_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), in.view_dist, camera.pbr.w, camera.fill.x),
                in.surface > 1.5 && in.light < 0.99,
            );
            incoming = mix(doom_light, baked_final, blend);
        }
    }
    let is_drop = in.surface > 1.5 && in.light > 2.0;
    let is_explosion = in.surface > 1.5 && in.light > 1.8 && !is_drop;
    let actual_light = select(in.light, in.light - 2.0, is_drop);
    let ceiling = max(camera.fill.x, select(1.0, camera.fill.z, camera.fill.z > 0.0));
    let obj_ceiling = select(1.0, max(ceiling, actual_light), in.surface <= 1.5 || is_explosion);
    incoming = clamp(incoming, vec3<f32>(camera.fill.x), vec3<f32>(obj_ceiling));
    let albedo = color.rgb * in.tint;

    let pbr = textureSampleLevel(pbr_tex, atlas_samp, atlas_uv, 0.0);
    var bump = 1.0;
    if (in.surface <= 1.5) {
        let n_xy = (pbr.rg * 2.0 - 1.0) * camera.pbr.x;
        bump = clamp(1.0 + dot(n_xy, vec2<f32>(0.25, 0.35)), 0.2, 2.0);
    }
    let roughness = clamp(pbr.b * camera.pbr.y, 0.02, 1.0);
    let metallic = pbr.a;
    var final_incoming = incoming;
    if (is_drop) {
        final_incoming = max(final_incoming, vec3<f32>(0.65));
    }
    let shaded = albedo * final_incoming * bump;
    let emission = max(
        textureSampleLevel(emit_tex, atlas_samp, atlas_uv, 0.0).r,
        select(0.0, in.light, is_explosion),
    );
    var radiance = max(shaded, albedo * emission);
    let sheen = (1.0 - roughness) * (metallic * 0.4 + 0.1) * camera.pbr.z;
    radiance = mix(radiance, radiance * (1.0 + sheen * 0.4) + final_incoming * sheen * 0.15, metallic * 0.5 + 0.1);
    return vec4<f32>(radiance, select(1.0, 0.40, is_drop));
}

@fragment
fn fs_weapon(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) {
        discard;
    }
    let atlas_uv = vec2<f32>(
        mix(in.atlas.x, in.atlas.z, in.uv.x),
        mix(in.atlas.y, in.atlas.w, in.uv.y),
    );
    let color = textureSampleLevel(atlas_tex, atlas_samp, atlas_uv, 0.0);
    if (color.a < 0.5) {
        discard;
    }
    let ceiling = max(camera.fill.x, select(1.0, camera.fill.z, camera.fill.z > 0.0));
    let incoming = clamp(vec3<f32>(in.light), vec3<f32>(camera.fill.x), vec3<f32>(ceiling));
    let albedo = color.rgb * in.tint;
    let shaded = albedo * incoming;
    let emission = textureSampleLevel(emit_tex, atlas_samp, atlas_uv, 0.0).r;
    let radiance = max(shaded, albedo * emission);
    return vec4<f32>(radiance, 1.0);
}

struct VoxelVertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) tint: vec3<f32>,
};

struct VoxelInstanceInput {
    @location(2) origin: vec3<f32>,
    @location(3) sin_yaw: f32,
    @location(4) cos_yaw: f32,
    @location(5) light: f32,
    @location(6) light_uv: vec2<f32>,
};

struct VoxelVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) @interpolate(flat) light_uv: vec2<f32>,
    @location(2) light: f32,
    @location(3) view_dist: f32,
    @location(4) world_pos: vec3<f32>,
};

@vertex
fn vs_voxel(v: VoxelVertexInput, inst: VoxelInstanceInput) -> VoxelVertexOutput {
    var out: VoxelVertexOutput;
    let world_pos = vec3<f32>(
        inst.origin.x + inst.cos_yaw * v.pos.y + inst.sin_yaw * v.pos.x,
        inst.origin.y + v.pos.z,
        inst.origin.z - inst.sin_yaw * v.pos.y + inst.cos_yaw * v.pos.x,
    );
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = v.tint;
    out.light_uv = inst.light_uv;
    out.light = inst.light;
    out.view_dist = out.clip_position.w;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_voxel(in: VoxelVertexOutput) -> @location(0) vec4<f32> {
    let dp1 = dpdx(in.world_pos);
    let dp2 = dpdy(in.world_pos);
    let cross_p = cross(dp1, dp2);
    let len2 = dot(cross_p, cross_p);
    let geom_n = select(vec3<f32>(0.0, 1.0, 0.0), cross_p * inverseSqrt(len2), len2 > 1e-8);
    // Directional shading for solid objects; self-luminous explosions emit uniformly.
    let is_drop = in.light > 2.0;
    let is_explosion = in.light > 1.8 && !is_drop;
    let actual_light = select(in.light, in.light - 2.0, is_drop);
    let shade = select(clamp(0.80 + 0.20 * geom_n.y + 0.05 * geom_n.x, 0.65, 1.0), 1.0, is_explosion);

    let ceiling = max(camera.fill.x, select(1.0, camera.fill.z, camera.fill.z > 0.0));
    let obj_ceiling = select(min(ceiling, 1.0), max(ceiling, actual_light), is_explosion);

    var doom_light = select(
        doom_diminish(vec3<f32>(actual_light), in.view_dist, camera.pbr.w, camera.fill.x),
        vec3<f32>(actual_light),
        is_explosion,
    );
    doom_light = clamp(doom_light, vec3<f32>(camera.fill.x), vec3<f32>(obj_ceiling));

    var incoming = doom_light;
    if (camera.fill.w < 0.5 && in.light_uv.x >= 0.0) {
        let dims = textureDimensions(light_tex);
        let coord = clamp(
            vec2<i32>(in.light_uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - vec2<i32>(1),
        );
        let baked = textureLoad(light_tex, coord, 0);
        if (baked.a > 0.0) {
            var baked_rgb = clamp(baked.rgb, vec3<f32>(camera.fill.x), vec3<f32>(obj_ceiling));
            baked_rgb = doom_diminish(baked_rgb, in.view_dist, camera.pbr.w, camera.fill.x);
            let blend = clamp(camera.extra.x, 0.0, 1.0);
            incoming = mix(doom_light, baked_rgb, blend);
        }
    }
    if (is_explosion) {
        incoming = max(incoming, vec3<f32>(in.light));
    }
    if (is_drop) {
        incoming = max(incoming, vec3<f32>(0.65));
    }
    incoming = clamp(incoming * shade, vec3<f32>(camera.fill.x), vec3<f32>(obj_ceiling));
    let lit_color = in.color * incoming;
    return vec4<f32>(lit_color, select(1.0, 0.40, is_drop));
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
const RAY_SHADER: &str = concat!(r#"
enable wgpu_ray_query;

override LINEAR_OUTPUT: bool = false;

struct FixtureLight {
    position_radius: vec4<f32>,
    color_power: vec4<f32>,
    triangle_c: vec4<f32>,
    atlas: vec4<f32>,
    uv_ab: vec4<f32>,
    uv_c: vec4<f32>,
};
struct FixtureSources {
    counts: vec4<u32>,
    lights: array<FixtureLight, 128>,
};
struct RayCam {
    view_proj: mat4x4<f32>,
    origin_aspect: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    viewport: vec4<f32>,
    // x: emissive scale, y: environment scale, z: bounce count.
    // w: 1 stores light on the surface, 0 averages the screen instead.
    light: vec4<f32>,
    // x: minimum incoming light so dark rooms stay readable.
    // y: 1 = play mode (sample light map only). z: rows reserved under the level.
    fill: vec4<f32>,
    // x: normal strength; y: roughness scale; z: specular sheen.
    pbr: vec4<f32>,
    fixtures: FixtureSources,
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
    geom_n: vec3<f32>,
    albedo: vec3<f32>,
    light: f32,
    emit: f32,
    sky: bool,
    sampled_emitter: bool,
    alpha: f32,
    hit: bool,
    // Albedo atlas coordinate. `light_uv` is this surface's own light texel,
    // unique to that placed wall or floor. Negative means it is not stored.
    atlas_uv: vec2<f32>,
    light_uv: vec2<f32>,
    roughness: f32,
    metallic: f32,
};

const VERTEX_FLOATS: u32 = 16u;
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
// Settled incoming light per surface texel. The pass reads it and shows it.
@group(0) @binding(8) var light_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var light_read: texture_2d<f32>;
// One fresh sample per screen pixel, binned into the texel it hit.
struct LightHit {
    r: f32,
    g: f32,
    b: f32,
    packed: u32,
}
@group(0) @binding(10) var<storage, read_write> light_hits: array<LightHit>;
@group(0) @binding(11) var pbr_tex: texture_2d<f32>;

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

fn bounce_albedo(alb: vec3<f32>, boost: f32) -> vec3<f32> {
    let luminance = dot(alb, vec3<f32>(0.299, 0.587, 0.114));
    let norm_color = alb / max(luminance, 0.001);
    let scale = select(1.8, boost, boost > 0.01);
    let boosted_lum = clamp(pow(luminance, 0.55) * scale, 0.0, 0.85);
    return clamp(norm_color * boosted_lum, vec3<f32>(0.0), vec3<f32>(0.85));
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
    surf.geom_n = vec3<f32>(0.0, 1.0, 0.0);
    surf.albedo = vec3<f32>(0.0);
    surf.light = 0.0;
    surf.emit = 0.0;
    surf.sky = false;
    surf.sampled_emitter = false;
    surf.alpha = 0.0;
    surf.hit = false;
    surf.atlas_uv = vec2<f32>(0.0);
    surf.light_uv = vec2<f32>(-1.0);
    surf.roughness = 0.8;
    surf.metallic = 0.0;
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return surf;
    }
    let prim = hit.primitive_index * 3u;
    var i0 = indices[prim];
    var i1 = indices[prim + 1u];
    var i2 = indices[prim + 2u];
    var b = vec3<f32>(
        1.0 - hit.barycentrics.x - hit.barycentrics.y,
        hit.barycentrics.x,
        hit.barycentrics.y,
    );
    let p0 = vec3<f32>(attr(i0, 0u), attr(i0, 1u), attr(i0, 2u));
    let p1 = vec3<f32>(attr(i1, 0u), attr(i1, 1u), attr(i1, 2u));
    let p2 = vec3<f32>(attr(i2, 0u), attr(i2, 1u), attr(i2, 2u));
    var n = normalize(cross(p1 - p0, p2 - p0));
    let surface = attr(i0, 13u);
    if (surface < -1.5 && dot(n, dir) < 0.0) {
        // Metal's committed ray query may select either coincident triangle.
        // The mesh records the opposite quad so all material and light UVs
        // can be resolved to the sidedef facing this ray, on every backend.
        let opposite = u32(-surface - 2.0);
        let q0 = vec3<f32>(attr(opposite, 0u), attr(opposite, 1u), attr(opposite, 2u));
        let q1 = vec3<f32>(attr(opposite + 1u, 0u), attr(opposite + 1u, 1u), attr(opposite + 1u, 2u));
        let q3 = vec3<f32>(attr(opposite + 3u, 0u), attr(opposite + 3u, 1u), attr(opposite + 3u, 2u));
        let local = origin + dir * hit.t - q0;
        let ds = q3 - q0;
        let dt = q1 - q0;
        let st = clamp(vec2<f32>(dot(local, ds) / dot(ds, ds), dot(local, dt) / dot(dt, dt)), vec2<f32>(0.0), vec2<f32>(1.0));
        i0 = opposite;
        if (st.y >= st.x) {
            i1 = opposite + 1u;
            i2 = opposite + 2u;
            b = vec3<f32>(1.0 - st.y, st.y - st.x, st.x);
        } else {
            i1 = opposite + 2u;
            i2 = opposite + 3u;
            b = vec3<f32>(1.0 - st.x, st.y, st.x - st.y);
        }
    }
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
    let uv0 = vec2<f32>(attr(i0, 3u), attr(i0, 4u));
    let uv1 = vec2<f32>(attr(i1, 3u), attr(i1, 4u));
    let uv2 = vec2<f32>(attr(i2, 3u), attr(i2, 4u));
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let delta_uv1 = uv1 - uv0;
    let delta_uv2 = uv2 - uv0;
    let det = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
    var tangent = select(vec3<f32>(1.0, 0.0, 0.0), normalize((edge1 * delta_uv2.y - edge2 * delta_uv1.y) * sign(det)), abs(det) > 1e-6);
    tangent = normalize(tangent - n * dot(n, tangent));
    let bitangent = cross(n, tangent);

    let pbr = textureSampleLevel(pbr_tex, atlas_samp, atlas_uv, 0.0);
    let n_raw = pbr.rg * 2.0 - 1.0;
    let n_z = sqrt(max(0.0, 1.0 - dot(n_raw, n_raw)));
    let normal_ts = normalize(vec3<f32>(n_raw * cam.pbr.x, n_z));
    let tbn = mat3x3<f32>(tangent, bitangent, n);
    var perturbed_n = normalize(tbn * normal_ts);
    if (dot(perturbed_n, n) < 0.2) {
        perturbed_n = normalize(mix(perturbed_n, n, 0.7));
    }
    if (dot(perturbed_n, dir) > 0.0) {
        perturbed_n = normalize(perturbed_n - dir * (dot(perturbed_n, dir) + 0.01));
    }
    let is_sky = bary_mix(i0, i1, i2, 13u, b) > 0.5;
    if (!is_sky && dot(perturbed_n, perturbed_n) > 0.001) {
        surf.n = perturbed_n;
    } else {
        surf.n = n;
    }
    surf.geom_n = n;
    surf.roughness = clamp(pbr.b * cam.pbr.y, 0.02, 1.0);
    surf.metallic = pbr.a;
    surf.albedo = tex.rgb * tint;
    surf.light = bary_mix(i0, i1, i2, 9u, b);
    surf.emit = textureSampleLevel(emit_tex, emit_samp, atlas_uv, 0.0).r;
    surf.sky = bary_mix(i0, i1, i2, 13u, b) > 0.5;
    surf.sampled_emitter = attr(i0, 13u) == -0.25 && cam.fixtures.counts.y != 0u;
    surf.alpha = tex.a;
    surf.hit = true;
    surf.atlas_uv = atlas_uv;
    surf.light_uv = vec2<f32>(
        bary_mix(i0, i1, i2, 14u, b),
        bary_mix(i0, i1, i2, 15u, b),
    );
    return surf;
}

// A soft beam from the camera. `cam.up.w` is its strength. The shadow ray is
// skipped for the surface we can already see.
fn flashlight(pos: vec3<f32>, n: vec3<f32>, check_shadow: bool) -> vec3<f32> {
    let amount = cam.up.w;
    if (amount <= 0.0) {
        return vec3<f32>(0.0);
    }
    let to_eye = cam.origin_aspect.xyz - pos;
    let dist2 = dot(to_eye, to_eye);
    if (dist2 < 1.0) {
        return vec3<f32>(0.0);
    }
    let dist = sqrt(dist2);
    let l = to_eye / dist;
    let ndl = max(dot(n, l), 0.0);
    let spot = max(dot(-l, cam.forward.xyz), 0.0);
    let cone = spot * spot * spot * spot;
    if (ndl <= 0.0 || cone <= 0.001) {
        return vec3<f32>(0.0);
    }
    if (check_shadow) {
        let blocker = trace(pos + n * RAY_BIAS, l);
        if (blocker.hit && !blocker.sky && distance(blocker.pos, pos) + 0.5 < dist) {
            return vec3<f32>(0.0);
        }
    }
    let falloff = 1.0 / (1.0 + dist2 * 0.00002);
    return vec3<f32>(1.0, 0.96, 0.88) * amount * ndl * cone * falloff;
}

// Sample one dedicated source: work is independent of the fixture count.
// Intensity is inverse-square, radius softens both shadows and near-field light.
fn fixture_light(pos: vec3<f32>, n: vec3<f32>, rng: ptr<function, u32>, first: u32, count: u32) -> vec3<f32> {
    if (count == 0u) { return vec3<f32>(0.0); }
    let index = first + min(u32(rand(rng) * f32(count)), count - 1u);
    let source = cam.fixtures.lights[index];
    var light_pos: vec3<f32>;
    var radiance = source.color_power.rgb * source.color_power.w * (cam.light.x / 40.0);
    let area_source = source.position_radius.w < 0.0;
    if (area_source) {
        let q = sqrt(rand(rng));
        let t = rand(rng);
        let bary = vec3<f32>(1.0 - q, q * (1.0 - t), q * t);
        light_pos = source.position_radius.xyz * bary.x + source.color_power.xyz * bary.y + source.triangle_c.xyz * bary.z;
        let uv = source.uv_ab.xy * bary.x + source.uv_ab.zw * bary.y + source.uv_c.xy * bary.z;
        let tile = fract(uv);
        let atlas_uv = mix(source.atlas.xy, source.atlas.zw, tile);
        let emission = textureSampleLevel(emit_tex, emit_samp, atlas_uv, 0.0).r;
        if (emission <= 0.0) { return vec3<f32>(0.0); }
        let color = textureSampleLevel(atlas_tex, atlas_samp, atlas_uv, 0.0).rgb;
        radiance = color * emission * cam.light.x * source.color_power.w / 3.14159265;
        // Ceilings emit into the room. Start just below the original plane.
        light_pos.y -= RAY_BIAS;
    } else {
        let z = rand(rng) * 2.0 - 1.0;
        let phi = rand(rng) * 6.2831853;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let offset = vec3<f32>(r * cos(phi), z, r * sin(phi)) * source.position_radius.w;
        light_pos = source.position_radius.xyz + offset;
    }
    let delta = light_pos - pos;
    let dist2 = dot(delta, delta);
    if (dist2 < 0.01) { return vec3<f32>(0.0); }
    let dist = sqrt(dist2);
    let direction = delta / dist;
    let ndl = max(dot(n, direction), 0.0);
    if (ndl <= 0.0) { return vec3<f32>(0.0); }
    let blocker = trace(pos + n * RAY_BIAS, direction);
    if (blocker.hit && !blocker.sky && distance(blocker.pos, pos) < dist - 2.0 * RAY_BIAS) {
        return vec3<f32>(0.0);
    }
    let radius2 = source.position_radius.w * source.position_radius.w;
    let emitter_cos = select(1.0, max(direction.y, 0.0), area_source);
    let denominator = select(max(dist2, radius2), dist2, area_source);
    return radiance * (ndl * emitter_cos * f32(count) / denominator);
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

fn load_light_uv(uv: vec2<f32>) -> vec3<f32> {
    if (uv.x < 0.0) {
        return vec3<f32>(0.0);
    }
    let dims = textureDimensions(light_read);
    let coord = clamp(
        vec2<i32>(uv * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    );
    return textureLoad(light_read, coord, 0).rgb;
}

// Play mode: walls and floors use their baked texel. Moving things have no
// texel, so they take the light of the floor (or wall) under them.
fn sample_surface_light(surf: Surf) -> vec3<f32> {
    var baked = vec3<f32>(0.0);
    if (surf.light_uv.x >= 0.0) {
        baked = load_light_uv(surf.light_uv);
    } else {
        let below = trace(surf.pos + vec3<f32>(0.0, 2.0, 0.0), vec3<f32>(0.0, -1.0, 0.0));
        if (below.hit && !below.sky && below.alpha >= 0.5 && below.light_uv.x >= 0.0) {
            baked = load_light_uv(below.light_uv);
        }
    }
    return max(baked, vec3<f32>(surf.light * cam.fill.x * 2.0));
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

    let emit_scale = cam.light.x;
    let play = cam.fill.y > 0.5;
    var incoming = vec3<f32>(0.0);
    if (play) {
        // One primary ray. Static surfaces read their texel; actors borrow
        // the light under their feet so they stay stable while moving.
        incoming = sample_surface_light(primary);
    } else {
        // Incoming light only. The texture stays at full resolution and is
        // applied after the light map is read.
        let frame = u32(cam.right.w);
        let px = u32(frag.x) + u32(frag.y) * 4096u;
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
                    incoming += throughput * sky_light(dir) * env_scale;
                    break;
                }
                // The surface we see keeps its own texel. Later hits contribute
                // their color into the light arriving here.
                let boost = cam.pbr.w;
                let bounce_alb = bounce_albedo(surf.albedo, boost);
                let shade = select(vec3<f32>(1.0), bounce_alb, bounce > 0);
                if (bounce > 0 && surf.emit > 0.0 && !surf.sampled_emitter) {
                    incoming += throughput * surf.albedo * surf.emit * emit_scale;
                }
                // Only one of the four paths samples fixtures. Compensate its
                // probability to retain energy. Bulbs and ceilings each get
                // one sample so large ceilings cannot starve small fixtures.
                // At most two shadow rays per bounce, never one per lamp.
                if (s == 0u) {
                    let points = cam.fixtures.counts.z;
                    let bulbs = fixture_light(surf.pos, surf.geom_n, &rng, 0u, points);
                    let panels = fixture_light(surf.pos, surf.geom_n, &rng, points, cam.fixtures.counts.x - points);
                    incoming += throughput * shade * (bulbs + panels) * f32(SAMPLES);
                }
                let ndl = sun_ndotl(surf.pos, surf.geom_n);
                if (ndl > 0.0) {
                    incoming += throughput * shade * SUN_RGB * ndl * env_scale;
                }
                // Open sky above this point. Floors take most of it; walls take a little.
                let above = trace(surf.pos + surf.geom_n * RAY_BIAS, vec3<f32>(0.0, 1.0, 0.0));
                if (above.sky) {
                    let facing = max(surf.geom_n.y, 0.0) * 0.5 + 0.25;
                    incoming += throughput * shade * sky_light(vec3<f32>(0.0, 1.0, 0.0)) * facing * env_scale;
                }
                let beam = flashlight(surf.pos, surf.geom_n, bounce > 0);
                if (beam.x > 0.0) {
                    incoming += throughput * shade * beam;
                }
                if (bounce + 1 >= bounce_limit) {
                    break;
                }
                if (bounce > 0) {
                    throughput = throughput * bounce_alb;
                }
                dir = hemisphere(surf.geom_n, &rng);
                let next = trace(surf.pos + surf.geom_n * RAY_BIAS, dir);
                if (!next.hit || next.alpha < 0.5) {
                    break;
                }
                surf = next;
            }
        }
        incoming = incoming / f32(SAMPLES);
        // Every screen pixel that lands on a texel is recorded. A later pass
        // averages those pixels into the one texel. The picture is always that
        // texel (black until the first store), so new surfaces do not wait on a
        // noisy "allocate" period before they look like lightmap light.
        if (cam.light.w > 0.5 && primary.light_uv.x >= 0.0) {
            let dims = textureDimensions(light_read);
            let coord = clamp(
                vec2<i32>(primary.light_uv * vec2<f32>(dims)),
                vec2<i32>(0),
                vec2<i32>(dims) - vec2<i32>(1),
            );
            let texel = u32(coord.y) * u32(dims.x) + u32(coord.x);
            let pix = u32(frag.x) + u32(frag.y) * u32(cam.viewport.z);
            if (pix < arrayLength(&light_hits)) {
                light_hits[pix].r = incoming.r;
                light_hits[pix].g = incoming.g;
                light_hits[pix].b = incoming.b;
                light_hits[pix].packed = texel + 1u;
            }
            let old = textureLoad(light_read, coord, 0);
            incoming = old.rgb;
        }
    }
    if (LINEAR_OUTPUT) {
        return FragOut(depth, vec4<f32>(incoming, 1.0));
    }
    var radiance = display_light(incoming, primary.albedo, primary.emit, cam.fill.x, cam.fill.w, emit_scale);
    let pbr = textureSampleLevel(pbr_tex, atlas_samp, primary.atlas_uv, 0.0);
    let n_xy = (pbr.rg * 2.0 - 1.0) * cam.pbr.x;
    let bump = clamp(1.0 + dot(n_xy, vec2<f32>(0.25, 0.35)), 0.2, 2.0);
    radiance = radiance * bump;

    let V = -view_dir;
    let NdotV = clamp(dot(primary.n, V), 0.0, 1.0);
    let F0 = mix(vec3<f32>(0.04), primary.albedo, primary.metallic);
    let fresnel = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - NdotV, 5.0);
    let sheen = fresnel * (1.0 - primary.roughness) * cam.pbr.z;
    radiance = mix(radiance, radiance * (vec3<f32>(1.0) + sheen * 0.35) + incoming * sheen * 0.15, primary.metallic * 0.5 + 0.15);
    return FragOut(depth, vec4<f32>(radiance, 1.0));
}

// Material/display pass after the screen HDR average (or the world-map read).
// Binding 9 is the screen's averaged linear lighting in this bind group.
@fragment
fn fs_present(@builtin(position) frag: vec4<f32>) -> FragOut {
    let vp = cam.viewport;
    let ndc_x = ((frag.x - vp.x) / vp.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((frag.y - vp.y) / vp.w) * 2.0;
    let dir = normalize(cam.right.xyz * ndc_x + cam.up.xyz * (ndc_y / max(cam.origin_aspect.w, 0.01)) + cam.forward.xyz);
    let primary = trace(cam.origin_aspect.xyz, dir);
    if (!primary.hit) { return FragOut(1.0, vec4<f32>(0.02, 0.02, 0.03, 1.0)); }
    if (primary.alpha < 0.5) { discard; }
    let clip = cam.view_proj * vec4<f32>(primary.pos, 1.0);
    let depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
    if (primary.sky) { return FragOut(depth, vec4<f32>(primary.albedo, 1.0)); }
    let incoming = textureLoad(light_read, vec2<i32>(frag.xy), 0).rgb;
    let base_incoming = max(incoming, vec3<f32>(primary.light * cam.fill.x * 2.0));
    var radiance = display_light(base_incoming, primary.albedo, primary.emit, cam.fill.x, cam.fill.w, cam.light.x);

    let pbr = textureSampleLevel(pbr_tex, atlas_samp, primary.atlas_uv, 0.0);
    let n_xy = (pbr.rg * 2.0 - 1.0) * cam.pbr.x;
    let bump = clamp(1.0 + dot(n_xy, vec2<f32>(0.25, 0.35)), 0.2, 2.0);
    radiance = radiance * bump;

    // Subtle PBR view-dependent Fresnel sheen and metallic luster (never overexposed).
    let V = -dir;
    let NdotV = clamp(dot(primary.n, V), 0.0, 1.0);
    let F0 = mix(vec3<f32>(0.04), primary.albedo, primary.metallic);
    let fresnel = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - NdotV, 5.0);
    let sheen = fresnel * (1.0 - primary.roughness) * cam.pbr.z;
    radiance = mix(radiance, radiance * (vec3<f32>(1.0) + sheen * 0.35) + base_incoming * sheen * 0.15, primary.metallic * 0.5 + 0.15);

    return FragOut(depth, vec4<f32>(radiance, 1.0));
}
"#, include_str!("gpu/light_display.wgsl"));

struct WorldGpu {
    atlas: wgpu::Texture,
    emission: wgpu::Texture,
    pbr: wgpu::Texture,
    /// Settled radiance on the atlas texels, sampled by the ray pass.
    light: wgpu::Texture,
    /// This frame's updated copy of `light`. Copied back after the pass.
    light_write: wgpu::Texture,
    /// Fresh incoming light, one record per ray-target pixel.
    light_hits: wgpu::Buffer,
    /// Adds every pixel on a texel into that texel, then writes the average.
    light_bins: Option<LightBins>,
    atlas_size: u32,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    weapon_pipeline: wgpu::RenderPipeline,
    voxel_pipeline: wgpu::RenderPipeline,
    pipeline_msaa: wgpu::RenderPipeline,
    weapon_pipeline_msaa: wgpu::RenderPipeline,
    voxel_pipeline_msaa: wgpu::RenderPipeline,
    _msaa_color: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    depth_samples: u32,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    vertex_cap: u64,
    index_cap: u64,
    voxel_mesh_cache: std::collections::HashMap<usize, (u32, u32)>,
    voxel_vertices_cpu: Vec<crate::voxel::VoxelVertex>,
    voxel_indices_cpu: Vec<u32>,
    voxel_mesh_dirty: bool,
    voxel_vertices: wgpu::Buffer,
    voxel_indices: wgpu::Buffer,
    voxel_instances: wgpu::Buffer,
    voxel_vert_cap: u64,
    voxel_index_cap: u64,
    voxel_inst_cap: u64,
    voxel_instances_len: usize,
    /// Hardware ray queries are available on this device.
    ray_query: bool,
    ray: Option<RayGpu>,
}

/// Walk the empty map and write light, or play with a cheap light-map sample.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TraceMode {
    /// No actors. Path-trace and store light into the textures.
    Bake,
    /// Actors back. Plain raster of albedo × light map — no hardware RT.
    Play,
}

/// Sliders for the path tracer.
pub(crate) struct TraceSettings {
    pub(crate) mode: TraceMode,
    /// Photons per masked texel. 4 matches the octree tracer's `texture_emission`.
    pub(crate) emission: f32,
    /// Scale on the sun and the sky.
    pub(crate) environment: f32,
    pub(crate) bounces: i32,
    /// Tallest path-traced image, in pixels. 720 is 720p for a 16:9 view.
    pub(crate) height: u32,
    /// Light is stored on the surfaces. Off, it only builds up on the screen.
    pub(crate) world_space: bool,
    /// Beam from the camera. World mode leaves it on the surfaces it hits.
    pub(crate) flashlight: f32,
    /// Floor under incoming light when drawing, so pitch-black rooms stay visible.
    pub(crate) min_light: f32,
    /// Display-only ceiling; HDR bake samples retain their original values.
    pub(crate) max_light: f32,
    /// Play mode renders at the classic Doom view size and nearest-upsamples.
    pub(crate) doom_res: bool,
    /// Smooth silhouettes during full-resolution raster Play.
    pub(crate) msaa: bool,
    /// Post-process edges before drawing the HUD.
    pub(crate) smaa: bool,
    pub(crate) neural: bool,
    pub(crate) neural_strength: f32,
    pub(crate) neural_style: u32,
    pub(crate) neural_mode: u32,
    pub(crate) neural_split_x: f32,
    pub(crate) neural_fsr: bool,
    pub(crate) neural_sharpness: f32,
    /// Normal map perturbation multiplier (0.0 = flat, 1.0 = default, up to 2.5).
    pub(crate) normal_strength: f32,
    /// Roughness multiplier (0.1 = high gloss/mirror, 1.0 = authored, 2.0 = matte).
    pub(crate) roughness_scale: f32,
    /// Specular sheen and metallic luster intensity (0.0 = off, 1.0 = default).
    pub(crate) specular_sheen: f32,
    /// Screen-Space Radiance Cascades for real-time secondary diffuse bounce.
    pub(crate) ssrc: bool,
    /// Multiplier for SSRC diffuse bounce irradiance (0.0 to 3.0).
    pub(crate) ssrc_strength: f32,
    /// Scale on diffuse indirect bounce albedo (1.0 = physical, 1.8 = real-world interior wall lift).
    pub(crate) bounce_boost: f32,
    /// Use raytraced baked lightmap when rendering Play mode.
    pub(crate) baked_light: bool,
    /// OG Doom distance light diminishing (0.0 = flat/disabled, 1.0 = classic hyperbolic falloff).
    pub(crate) diminish: f32,
    /// Blend between normal Doom lighting (0.0) and raytraced baked lighting (1.0).
    pub(crate) light_blend: f32,
    /// Blend between normal Doom lighting (0.0) and raytraced baked lighting (1.0) for voxel/actor objects.
    pub(crate) voxel_blend: f32,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self {
            mode: TraceMode::Play,
            emission: 40.0,
            environment: 1.0,
            bounces: 4,
            height: 720,
            world_space: true,
            flashlight: 0.0,
            min_light: 0.0,
            max_light: 2.0,
            doom_res: true,
            msaa: false,
            smaa: false,
            neural: false,
            neural_strength: 1.0,
            neural_style: 2,
            neural_mode: 1,
            neural_split_x: 0.5,
            neural_fsr: true,
            neural_sharpness: 0.8,
            normal_strength: 1.0,
            roughness_scale: 1.0,
            specular_sheen: 1.0,
            ssrc: true,
            ssrc_strength: 0.7,
            bounce_boost: 1.8,
            baked_light: true,
            diminish: 1.5,
            light_blend: 0.5,
            voxel_blend: 0.4,
        }
    }
}

impl TraceSettings {
    pub(crate) fn baking(&self) -> bool {
        self.mode == TraceMode::Bake
    }

    pub(crate) fn playing(&self) -> bool {
        self.mode == TraceMode::Play
    }

    /// Bake hides things so light fills the empty map; play draws them again.
    pub(crate) fn include_actors(&self) -> bool {
        self.playing()
    }

    fn stores_light(&self) -> bool {
        self.baking() && self.world_space
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
    _display: wgpu::Texture,
    display: wgpu::TextureView,
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
    /// Chunkier upscale for classic Doom resolution play.
    nearest: wgpu::Sampler,
}

/// Low-res raster target for play mode at native Doom view size.
struct PlayImage {
    _color: wgpu::Texture,
    view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    _msaa_color: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    bind: wgpu::BindGroup,
    size: (u32, u32),
    samples: u32,
}

struct RayGpu {
    pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    present_bind: Option<wgpu::BindGroup>,
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
    /// Offscreen raster play view when [`TraceSettings::doom_res`] is on.
    play_image: Option<PlayImage>,
    #[cfg(not(target_arch = "wasm32"))]
    nr_live: Option<neural::Live>,
    smaa_target: Option<(smaa::SmaaTarget, (u32, u32))>,
    ssrc_pass: Option<ssrc::SsrcPass>,
    blit: BlitGpu,
    accum: AccGpu,
    #[cfg(not(target_arch = "wasm32"))]
    ui: crate::trace_ui::TraceUi,
    /// Level packing currently stored in the light texture.
    light_bake_key: Option<crate::hw::LightBakeKey>,
    /// True when the GPU light map has samples that are not on disk yet.
    light_bake_dirty: bool,
    pending_clear_visible: bool,
    /// Last time a bake copy was started. Periodic saves stay off the hot path.
    light_bake_saved: Option<std::time::Instant>,
    /// Non-blocking readback / background write of a light bake.
    #[cfg(not(target_arch = "wasm32"))]
    light_bake_job: Option<LightBakeJob>,
}

impl Drop for GpuState {
    fn drop(&mut self) {
        self.begin_light_bake_save();
        self.finish_light_bake_save();
    }
}

/// GPU copy of the light map, then a background thread writes the sparse file.
#[cfg(not(target_arch = "wasm32"))]
enum LightBakeJob {
    Mapping {
        key: crate::hw::LightBakeKey,
        buffer: wgpu::Buffer,
        bytes_per_row: u32,
        /// Full atlas edge length (for sparse texel indices).
        size: u32,
        /// Copied band: `row_y` and `row_count` full-width rows.
        row_y: u32,
        row_count: u32,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
    Writing(std::thread::JoinHandle<()>),
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
        #[allow(unused_mut)]
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

        #[cfg(not(target_arch = "wasm32"))]
        let want_nr = std::env::var_os("WDOOM_NR_MODEL").is_some()
            && adapter_info.backend == wgpu::Backend::Metal
            && adapter.features().contains(wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        #[cfg(target_arch = "wasm32")]
        let want_nr = false;
        let mut ray_limits = if want_rays {
            limits
                .clone()
                .using_acceleration_structure_values(adapter.limits())
        } else {
            limits.clone()
        };
        // Ray queries are still an experimental wgpu feature. The Metal backend
        // on this machine implements them; agreeing to the token is required
        // before the device will enable `EXPERIMENTAL_RAY_QUERY`.
        if want_nr {ray_limits.max_compute_workgroup_storage_size=32768;}
        let experimental = if want_rays || want_nr {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::default()
        };
        let mut ray_features = if want_rays {
            wgpu::Features::EXPERIMENTAL_RAY_QUERY
        } else {
            wgpu::Features::empty()
        };

        if want_nr {ray_features |= wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;}
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
        let accum = create_accum(&device, if ray_query { wgpu::TextureFormat::Rgba16Float } else { surface_format });
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
            play_image: None,
            smaa_target: None,
            ssrc_pass: None,
            #[cfg(not(target_arch = "wasm32"))]
            nr_live: None,
            blit,
            accum,
            #[cfg(not(target_arch = "wasm32"))]
            ui,
            light_bake_key: None,
            light_bake_dirty: false,
            pending_clear_visible: false,
            light_bake_saved: None,
            #[cfg(not(target_arch = "wasm32"))]
            light_bake_job: None,
        })
    }

    pub(crate) fn toggle_neural(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {self.trace.neural = !self.trace.neural;log::info!("Neural rendering {}",if self.trace.neural {"on"}else{"off"});}
    }

    #[allow(dead_code)]
    pub(crate) fn cycle_neural_mode(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.trace.neural_mode = (self.trace.neural_mode + 1) % 3;
            log::info!(
                "Neural mode: {}",
                match self.trace.neural_mode {
                    0 => "Split Comparison",
                    1 => "Full DLSS-NR",
                    _ => "Full Native 3D",
                }
            );
        }
    }

    pub(crate) fn toggle_smaa(&mut self) -> bool {
        self.trace.smaa = !self.trace.smaa;
        log::info!("SMAA {}", if self.trace.smaa { "on" } else { "off" });
        self.trace.smaa
    }

    pub(crate) fn smaa(&self) -> bool {
        self.trace.smaa
    }

    pub(crate) fn set_smaa(&mut self, enabled: bool) -> bool {
        self.trace.smaa = enabled;
        log::info!("SMAA {}", if self.trace.smaa { "on" } else { "off" });
        self.trace.smaa
    }

    pub(crate) fn toggle_msaa(&mut self) -> bool {
        self.trace.msaa = !self.trace.msaa;
        log::info!("4× MSAA {}", if self.trace.msaa { "on" } else { "off" });
        self.trace.msaa
    }

    pub(crate) fn msaa(&self) -> bool {
        self.trace.msaa
    }

    pub(crate) fn set_msaa(&mut self, enabled: bool) -> bool {
        self.trace.msaa = enabled;
        log::info!("4× MSAA {}", if self.trace.msaa { "on" } else { "off" });
        self.trace.msaa
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
            .map(|(size, rgba, emit, pbr)| (*size, rgba.as_slice(), emit.as_slice(), pbr.as_slice()));
        self.ensure_world(atlas);
        if scene.light_reset {
            self.reset_light_bake(scene.light_bake);
        } else if scene.light_bake.is_some() {
            self.light_bake_key = scene.light_bake;
        }
        let rect = scene.view_rect;
        let surf_w = self.config.width as f32;
        let surf_h = self.config.height as f32;
        let vx = (rect[0] * surf_w).round().clamp(0.0, surf_w - 1.0);
        let vy = (rect[1] * surf_h).round().clamp(0.0, surf_h - 1.0);
        let vw = (rect[2] * surf_w).round().clamp(1.0, surf_w - vx);
        let vh = (rect[3] * surf_h).round().clamp(1.0, surf_h - vy);
        if let Some(world) = self.world.as_mut() {
            upload_mesh(world, &self.device, &self.queue, scene);
            let cam = WorldUniform {
                view_proj: scene.view_proj,
                fill: [
                    self.trace.min_light,
                    self.trace.light_blend,
                    self.trace.max_light,
                    if self.trace.baked_light { 0.0 } else { 1.0 },
                ],
                pbr: [
                    self.trace.normal_strength,
                    self.trace.roughness_scale,
                    self.trace.specular_sheen,
                    self.trace.diminish,
                ],
                extra: [
                    self.trace.voxel_blend,
                    0.0,
                    0.0,
                    0.0,
                ],
            };
            self.queue
                .write_buffer(&world.uniform, 0, bytemuck::bytes_of(&cam));
        }
        // Play is plain raster × light map. Only bake builds rays / BLAS.
        let baking = self.trace.baking();
        let doom_res = self.trace.playing() && self.trace.doom_res;
        let msaa = self.trace.playing() && self.trace.msaa;
        let (rw, rh) = if baking {
            ray_extent(vw, vh, self.trace.height)
        } else if doom_res {
            // Classic Doom 3D view size (usually 320×168 with the status bar).
            unsafe {
                (
                    crate::doom::r_draw::viewwidth.max(1) as u32,
                    crate::doom::r_draw::viewheight.max(1) as u32,
                )
            }
        } else {
            (vw as u32, vh as u32)
        };
        self.traced = (rw, rh);
        let screen_weight = if baking {
            self.begin_accumulation(scene, rw, rh)
        } else {
            1.0
        };
        let trace = baking
            && self.prepare_rays(scene, 0.0, 0.0, rw as f32, rh as f32, screen_weight);
        if trace {
            self.ensure_ray_image(rw, rh);
            self.ensure_ray_present_bind();
        } else if doom_res {
            let samples = if msaa { 4 } else { 1 };
            self.ensure_play_image(rw, rh, samples);
        } else if let Some(world) = self.world.as_mut() {
            let samples = if msaa { 4 } else { 1 };
            let resized = world.depth_size != (self.config.width, self.config.height);
            if resized || world.depth_samples != samples {
                let (w, h) = if msaa { (self.config.width, self.config.height) } else { (1, 1) };
                let (color, view) = msaa_color_target(&self.device, w, h, self.config.format);
                world._msaa_color = color;
                world.msaa_view = view;
            }
            if resized || world.depth_samples != samples {
                let (depth, depth_view) = depth_target_samples(&self.device, self.config.width, self.config.height, samples);
                world.depth = depth;
                world.depth_view = depth_view;
                world.depth_samples = samples;
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
        #[cfg(not(target_arch = "wasm32"))]
        if self.trace.neural && self.trace.playing() {
            if let Some(nr) = self.nr_live.as_mut() {
                nr.resize(self.config.format,(self.config.width,self.config.height));
            } else {
                match neural::Live::new(&self.device,&self.queue,self.config.format,(self.config.width,self.config.height),vw/vh) {
                    Ok(nr)=>self.nr_live=Some(nr),
                    Err(e)=>{log::error!("Neural rendering disabled: {e}");self.trace.neural=false;}
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let nr_active=self.trace.neural && self.trace.playing() && self.nr_live.is_some();
        #[cfg(target_arch = "wasm32")]
        let nr_active = false;
        // Keep classic nearest-neighbor pixels intact by default.
        // When SMAA is enabled, post-pixelization morphological antialiasing
        // smooths chunky pixel edges in Doom resolution, leaving HUD crisp.
        let smaa_enabled = self.trace.smaa && !nr_active;
        if smaa_enabled {
            let size = (self.config.width, self.config.height);
            match self.smaa_target.as_mut() {
                Some((target, old_size)) if *old_size != size => {
                    target.resize(&self.device, size.0, size.1);
                    *old_size = size;
                }
                None => {
                    self.smaa_target = Some((smaa::SmaaTarget::new(
                        &self.device, &self.queue, size.0, size.1,
                        self.config.format, smaa::SmaaMode::Smaa1X,
                    ), size));
                }
                _ => {}
            }
        }
        let mut smaa_frame = if smaa_enabled {
            Some(self.smaa_target.as_mut().expect("SMAA target").0
                .start_frame(&self.device, &self.queue, &view))
        } else { None };
        let post_view = smaa_frame.as_deref().unwrap_or(&view);
        #[cfg(not(target_arch = "wasm32"))]
        let scene_view = if nr_active && !doom_res { &self.nr_live.as_ref().expect("NR ready").view } else {post_view};
        #[cfg(target_arch = "wasm32")]
        let scene_view = post_view;
        let store_light = self.trace.stores_light();
        if trace {
            // World / bake shows the texel. The screen average would keep a
            // separate value in every pixel of that texel. Play is one sample.
            let image_weight = if store_light || self.trace.playing() {
                1.0
            } else {
                screen_weight
            };
            self.queue.write_buffer(
                &self.accum.uniform,
                0,
                bytemuck::cast_slice(&[image_weight, 0.0, 0.0, 0.0]),
            );
            if store_light {
                if let Some(world) = self.world.as_ref() {
                    let params = BinParams {
                        pixels: rw * rh,
                        light_size: crate::hw::LIGHTMAP_SIZE,
                        slots: LIGHT_SLOTS,
                        voxel_rows: (crate::hw::VOXEL_LIGHT_ROWS.max(0) as u32 & 0xffff)
                            | (FRAME.load(std::sync::atomic::Ordering::Relaxed) << 16),
                    };
                    self.queue.write_buffer(
                        &world.light_bins.as_ref().expect("native light bins").uniform,
                        0,
                        bytemuck::bytes_of(&params),
                    );
                }
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("doom_world_encoder"),
            });
        #[cfg(not(target_arch = "wasm32"))]
        let mut nr_cmd = None;

        if trace {
            if let Some(world) = self.world.as_mut() {
                if store_light {
                    copy_light(
                        &mut encoder,
                        &world.light,
                        &world.light_write,
                        crate::hw::LIGHTMAP_SIZE,
                    );
                    encoder.clear_buffer(&world.light_hits, 0, None);
                    encoder.clear_buffer(&world.light_bins.as_ref().expect("native light bins").bins, 0, None);
                }
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
            draw_scene(&mut pass, world, scene, true, false);
            }
            if store_light {
                if let Some(world) = self.world.as_ref() {
                    if self.pending_clear_visible {
                        self.pending_clear_visible = false;
                        {
                            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("doom_light_clear_visible"),
                                timestamp_writes: None,
                            });
                            pass.set_pipeline(&world.light_bins.as_ref().expect("native light bins").clear_visible_pipeline);
                            pass.set_bind_group(0, &world.light_bins.as_ref().expect("native light bins").bind, &[]);
                            pass.dispatch_workgroups((rw * rh + 63) / 64, 1, 1);
                        }
                        copy_light(
                            &mut encoder,
                            &world.light_write,
                            &world.light,
                            crate::hw::LIGHTMAP_SIZE,
                        );
                        self.accum_count = 0;
                        self.accum_key = None;
                        self.light_bake_dirty = true;
                    }
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("doom_light_bin"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&world.light_bins.as_ref().expect("native light bins").bin_pipeline);
                        pass.set_bind_group(0, &world.light_bins.as_ref().expect("native light bins").bind, &[]);
                        pass.dispatch_workgroups((rw * rh + 63) / 64, 1, 1);
                    }
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("doom_light_save"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&world.light_bins.as_ref().expect("native light bins").save_pipeline);
                        pass.set_bind_group(0, &world.light_bins.as_ref().expect("native light bins").bind, &[]);
                        pass.dispatch_workgroups(LIGHT_SLOTS / 256, 1, 1);
                    }
                    copy_light(
                        &mut encoder,
                        &world.light_write,
                        &world.light,
                        crate::hw::LIGHTMAP_SIZE,
                    );
                    self.light_bake_dirty = true;
                }
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
                let ray = self.world.as_ref().and_then(|w| w.ray.as_ref()).expect("rays");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("doom_ray_display"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &image.display, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &image.depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: None,
                    }),
                    occlusion_query_set: None, timestamp_writes: None, multiview_mask: None,
                });
                pass.set_pipeline(&ray.present_pipeline);
                pass.set_bind_group(0, ray.present_bind.as_ref().expect("display bindings"), &[]);
                pass.draw(0..6, 0..1);
            }
            {
            let image = self.ray_image.as_ref().expect("ray image");
            let world = self.world.as_ref().expect("world");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_sprites"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image.display,
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
            if self.trace.ssrc {
                let image = self.ray_image.as_ref().expect("ray image");
                if self.ssrc_pass.is_none() {
                    self.ssrc_pass = Some(ssrc::SsrcPass::new(&self.device, self.config.format, rw, rh));
                }
                if let Some(ssrc) = self.ssrc_pass.as_mut() {
                    ssrc.render(
                        &mut encoder,
                        &self.device,
                        &self.queue,
                        &image.display,
                        &image.depth_view,
                        &image.display,
                        rw,
                        rh,
                        self.trace.ssrc_strength,
                    );
                }
            }
            {
            let image = self.ray_image.as_ref().expect("ray image");
            let world = self.world.as_ref().expect("world");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_weapon"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image.display,
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
            draw_weapon(&mut pass, world, scene, false);
            }
            let image = self.ray_image.as_ref().expect("ray image");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_ray_blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
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
        } else if doom_res {
            {
                let image = self.play_image.as_ref().expect("play image");
                let world = self.world.as_ref().expect("world");
                let color_view = if msaa {
                    image.msaa_view.as_ref().expect("play msaa view")
                } else {
                    &image.view
                };
                let resolve_target = if msaa { Some(&image.view) } else { None };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("doom_play_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target,
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
                draw_scene(&mut pass, world, scene, false, msaa);
            }
            if self.trace.ssrc && !msaa {
                let image = self.play_image.as_ref().expect("play image");
                if self.ssrc_pass.is_none() {
                    self.ssrc_pass = Some(ssrc::SsrcPass::new(&self.device, self.config.format, rw, rh));
                }
                if let Some(ssrc) = self.ssrc_pass.as_mut() {
                    ssrc.render(
                        &mut encoder,
                        &self.device,
                        &self.queue,
                        &image.view,
                        &image.depth_view,
                        &image.view,
                        rw,
                        rh,
                        self.trace.ssrc_strength,
                    );
                }
            }
            {
                let image = self.play_image.as_ref().expect("play image");
                let world = self.world.as_ref().expect("world");
                let color_view = if msaa {
                    image.msaa_view.as_ref().expect("play msaa view")
                } else {
                    &image.view
                };
                let resolve_target = if msaa { Some(&image.view) } else { None };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("doom_play_weapon"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target,
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
                draw_weapon(&mut pass, world, scene, msaa);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let handled_by_nr = if nr_active {
                let image = self.play_image.as_ref().expect("play image");
                match self.nr_live.as_mut().expect("NR ready").encode_custom(
                    &mut encoder,
                    post_view,
                    &image.view,
                    (rw, rh),
                    [0.0, 0.0, rw as f32, rh as f32],
                    [vx, vy, vw, vh],
                    self.trace.neural_strength,
                    self.trace.neural_style as f32,
                    self.trace.neural_mode,
                    self.trace.neural_split_x,
                    self.trace.neural_fsr,
                    self.trace.neural_sharpness,
                    true,
                ) {
                    Ok(cmd) => {
                        nr_cmd = cmd;
                        true
                    }
                    Err(error) => {
                        log::error!("Neural preview stopped: {error}");
                        self.trace.neural = false;
                        false
                    }
                }
            } else {
                false
            };
            #[cfg(target_arch = "wasm32")]
            let handled_by_nr = false;

            if !handled_by_nr {
                let image = self.play_image.as_ref().expect("play image");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("doom_play_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: post_view,
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
            }
        } else if let Some(world) = self.world.as_ref() {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("doom_world_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: if msaa { &world.msaa_view } else { scene_view },
                        resolve_target: if msaa { Some(scene_view) } else { None },
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
                draw_scene(&mut pass, world, scene, false, msaa);
                draw_weapon(&mut pass, world, scene, msaa);
            }

            #[cfg(not(target_arch = "wasm32"))]
            if nr_active {
                match self.nr_live.as_mut().expect("NR ready").encode(
                    &mut encoder,
                    post_view,
                    [vx, vy, vw, vh],
                    self.trace.neural_strength,
                    self.trace.neural_style as f32,
                    self.trace.neural_mode,
                    self.trace.neural_split_x,
                    self.trace.neural_fsr,
                    self.trace.neural_sharpness,
                ) {
                    Ok(cmd) => nr_cmd = cmd,
                    Err(error) => {
                        log::error!("Neural preview stopped: {error}");
                        self.trace.neural = false;
                    }
                }
            }
        }

        // SMAA owns its resolve submission. Submit the scene first, then
        // resolve, then draw the HUD and egui onto the final surface.
        if let Some(frame) = smaa_frame.take() {
            self.queue.submit([encoder.finish()]);
            frame.resolve();
            encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("doom_hud_encoder"),
            });
        }
        drop(smaa_frame);

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

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(cmd) = nr_cmd {
            self.queue.submit([encoder.finish(), cmd]);
        } else {
            self.queue.submit(std::iter::once(encoder.finish()));
        }
        #[cfg(target_arch = "wasm32")]
        self.queue.submit(std::iter::once(encoder.finish()));

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(nr) = self.nr_live.as_mut() {
            nr.after_submit();
        }

        output.present();
        self.maybe_save_light_bake();
        Ok(())
    }

    /// Snapshot the current light map off-thread, then load the bake for `next`
    /// (or clear when there is none).
    fn reset_light_bake(&mut self, next: Option<crate::hw::LightBakeKey>) {
        // Copy first so the staging buffer keeps the old bake while we replace
        // the live texture.
        self.begin_light_bake_save();
        self.light_bake_key = next;
        let Some(world) = self.world.as_ref() else {
            return;
        };
        let size = crate::hw::LIGHTMAP_SIZE;
        if let Some(key) = next {
            if self.load_light_bake(key, &world.light, size) {
                return;
            }
        }
        clear_light(&self.queue, &world.light, size);
    }

    fn maybe_save_light_bake(&mut self) {
        // Only finish in-flight work during play. A full light-map readback is
        // started when leaving the level or quitting, never on a timer.
        self.poll_light_bake();
    }

    /// Start a GPU→CPU copy. The frame is not blocked; [`poll_light_bake`]
    /// finishes the sparse encode and disk write later.
    fn begin_light_bake_save(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.poll_light_bake();
            if self.light_bake_job.is_some() || !self.light_bake_dirty {
                return;
            }
            let Some(key) = self.light_bake_key else {
                return;
            };
            let Some(world) = self.world.as_ref() else {
                return;
            };
            let size = crate::hw::LIGHTMAP_SIZE.max(1);
            // Only the packed band — a full 4096² readback is ~128 MB and
            // stalls quit even when almost every texel is empty.
            let (row_y, row_count) = crate::hw::light_packed_rows().unwrap_or((0, size));
            let row_count = row_count.max(1).min(size.saturating_sub(row_y));
            let bytes_per_row = (size * 8).div_ceil(256) * 256;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("doom_light_readback"),
                size: u64::from(bytes_per_row) * u64::from(row_count),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("doom_light_readback_encoder"),
                });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &world.light,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: row_y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(row_count),
                    },
                },
                wgpu::Extent3d {
                    width: size,
                    height: row_count,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.light_bake_job = Some(LightBakeJob::Mapping {
                key,
                buffer,
                bytes_per_row,
                size,
                row_y,
                row_count,
                rx,
            });
            // The staging buffer holds this snapshot; later samples mark dirty again.
            self.light_bake_dirty = false;
            self.light_bake_saved = Some(std::time::Instant::now());
        }
    }

    fn poll_light_bake(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let job = match self.light_bake_job.take() {
                Some(job) => job,
                None => return,
            };
            match job {
                LightBakeJob::Mapping {
                    key,
                    buffer,
                    bytes_per_row,
                    size,
                    row_y,
                    row_count,
                    rx,
                } => match rx.try_recv() {
                    Ok(Ok(())) => {
                        // Map reads stay on this thread — wgpu/Metal can hang if
                        // another thread touches the mapped buffer.
                        let mapped = buffer.slice(..).get_mapped_range();
                        let texels = crate::light_bake::sparse_from_padded_rows(
                            &mapped,
                            size,
                            row_y,
                            row_count,
                            bytes_per_row,
                        );
                        drop(mapped);
                        buffer.unmap();
                        if texels.is_empty() {
                            return;
                        }
                        let Some(path) = crate::light_bake::bake_path(key) else {
                            log::warn!(
                                "light bake not saved: no savegame directory (e{} m{})",
                                key.episode,
                                key.map
                            );
                            self.light_bake_dirty = true;
                            return;
                        };
                        let handle = std::thread::spawn(move || {
                            match crate::light_bake::write_bake(&path, key, size, &texels) {
                                Ok(()) => log::info!(
                                    "saved light bake {} texels to {}",
                                    texels.len(),
                                    path.display()
                                ),
                                Err(err) => log::warn!("light bake save failed: {err}"),
                            }
                        });
                        self.light_bake_job = Some(LightBakeJob::Writing(handle));
                    }
                    Ok(Err(err)) => {
                        log::warn!("light bake map failed: {err}");
                        self.light_bake_dirty = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        self.light_bake_job = Some(LightBakeJob::Mapping {
                            key,
                            buffer,
                            bytes_per_row,
                            size,
                            row_y,
                            row_count,
                            rx,
                        });
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        log::warn!("light bake map disconnected");
                        self.light_bake_dirty = true;
                    }
                },
                LightBakeJob::Writing(handle) => {
                    if handle.is_finished() {
                        if let Err(err) = handle.join() {
                            log::warn!("light bake writer panicked: {err:?}");
                        }
                    } else {
                        self.light_bake_job = Some(LightBakeJob::Writing(handle));
                    }
                }
            }
        }
    }

    fn finish_light_bake_save(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.light_bake_dirty {
                self.begin_light_bake_save();
            }
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_secs(60);
            while std::time::Instant::now() < deadline {
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                self.poll_light_bake();
                match self.light_bake_job.take() {
                    None => return,
                    Some(LightBakeJob::Writing(handle)) => {
                        if let Err(err) = handle.join() {
                            log::warn!("light bake writer panicked: {err:?}");
                        }
                        return;
                    }
                    Some(mapping) => {
                        self.light_bake_job = Some(mapping);
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                }
            }
            if self.light_bake_job.is_some() {
                log::warn!("light bake still in flight on shutdown");
            }
        }
    }

    fn load_light_bake(
        &self,
        key: crate::hw::LightBakeKey,
        light: &wgpu::Texture,
        size: u32,
    ) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(path) = crate::light_bake::bake_path(key) else {
                return false;
            };
            let Some(texels) = crate::light_bake::read_bake(&path, key, size) else {
                return false;
            };
            if texels.is_empty() {
                return false;
            }
            let (bytes, stride) = crate::light_bake::padded_from_sparse(&texels, size);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: light,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size),
                },
                light_extent(size),
            );
            log::info!(
                "loaded light bake {} texels from {}",
                texels.len(),
                path.display()
            );
            true
        }
        #[cfg(target_arch = "wasm32")]
        {
            crate::web_light::load(key, light.clone(), self.queue.clone(), size, move || {
                crate::platform::GPU.with_borrow(|gpu| {
                    gpu.as_ref().is_some_and(|gpu| gpu.light_bake_key == Some(key))
                })
            });
            false
        }
    }

    fn ensure_play_image(&mut self, width: u32, height: u32, samples: u32) {
        if self
            .play_image
            .as_ref()
            .is_some_and(|image| image.size == (width, height) && image.samples == samples)
        {
            return;
        }
        let format = self.config.format;
        let color = ray_target(&self.device, width, height, format, true);
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let (depth, depth_view) = depth_target_samples(&self.device, width, height, samples);
        let (msaa_color, msaa_view) = if samples > 1 {
            let (c, v) = msaa_color_target(&self.device, width, height, format);
            (Some(c), Some(v))
        } else {
            (None, None)
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_play_blit_group"),
            layout: &self.blit.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit.nearest),
                },
            ],
        });
        self.play_image = Some(PlayImage {
            _color: color,
            view,
            _depth: depth,
            depth_view,
            _msaa_color: msaa_color,
            msaa_view,
            bind,
            size: (width, height),
            samples,
        });
    }

    fn ensure_ray_image(&mut self, width: u32, height: u32) {
        if self
            .ray_image
            .as_ref()
            .is_some_and(|image| image.size == (width, height))
        {
            return;
        }
        let format = wgpu::TextureFormat::Rgba16Float;
        let color = ray_target(&self.device, width, height, format, false);
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let history = ray_target(&self.device, width, height, format, true);
        let history_view = history.create_view(&wgpu::TextureViewDescriptor::default());
        let combined = ray_target(&self.device, width, height, format, true);
        let combined_view = combined.create_view(&wgpu::TextureViewDescriptor::default());
        let display = ray_target(&self.device, width, height, self.config.format, false);
        let display_view = display.create_view(&wgpu::TextureViewDescriptor::default());
        let (depth, depth_view) = depth_target(&self.device, width, height);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("doom_ray_blit_group"),
            layout: &self.blit.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_view),
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
        if let Some(ray) = self.world.as_mut().and_then(|w| w.ray.as_mut()) {
            ray.present_bind = None;
        }
        self.accum_count = 0;
        self.accum_key = None;
        self.ray_image = Some(RayImage {
            _color: color,
            view,
            _history: history,
            _combined: combined,
            combined: combined_view,
            _display: display,
            display: display_view,
            _depth: depth,
            depth_view,
            bind,
            accum_bind,
            size: (width, height),
        });
    }

    fn ensure_ray_present_bind(&mut self) {
        let Some(image) = self.ray_image.as_ref() else { return; };
        let Some(world) = self.world.as_mut() else { return; };
        let Some(ray) = world.ray.as_mut() else { return; };
        if ray.present_bind.is_none() {
            ray.present_bind = Some(ray_bind_group(&self.device, &ray.layout, &ray.uniform,
                &world.atlas, &world.emission, &world.pbr, &image._combined, &world.light_write,
                &world.vertices, &world.indices, &world.light_hits, &ray.tlas));
        }
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

    /// Explicitly restart native accumulation, retaining the user's old bake.
    #[cfg(not(target_arch = "wasm32"))]
    fn restart_light_bake(&mut self) {
        self.finish_light_bake_save();
        if let Some(path) = self.light_bake_key.and_then(crate::light_bake::bake_path) {
            if path.exists() {
                let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default().as_nanos();
                let backup = path.with_extension(format!("rmlt.backup-{stamp}"));
                if let Err(error) = std::fs::copy(&path, &backup) {
                    log::error!("could not preserve native bake; restart cancelled: {error}");
                    return;
                }
                log::info!("preserved native bake at {}", backup.display());
            }
        }
        if let Some(world) = self.world.as_ref() {
            clear_light(&self.queue, &world.light, crate::hw::LIGHTMAP_SIZE);
        }
        self.accum_count = 0;
        self.accum_key = None;
        self.light_bake_dirty = true;
        log::info!("restarted native light accumulation from zero samples");
    }

    /// Clear only the lightmap texels currently visible in the camera view,
    /// allowing the user to rebake what they are looking at without wiping the map.
    pub fn clear_visible(&mut self) {
        self.trace.mode = TraceMode::Bake;
        crate::hw::set_bake_safe(true);
        self.pending_clear_visible = true;
        self.accum_count = 0;
        self.accum_key = None;
        self.light_bake_dirty = true;
        log::info!("clearing visible lightmap texels for selective rebake");
    }

    pub fn normal_strength(&self) -> f32 {
        self.trace.normal_strength
    }

    pub fn set_normal_strength(&mut self, val: f32) {
        self.trace.normal_strength = val.clamp(0.0, 5.0);
    }

    pub fn roughness_scale(&self) -> f32 {
        self.trace.roughness_scale
    }

    pub fn set_roughness_scale(&mut self, val: f32) {
        self.trace.roughness_scale = val.clamp(0.01, 5.0);
    }

    pub fn specular_sheen(&self) -> f32 {
        self.trace.specular_sheen
    }

    pub fn set_specular_sheen(&mut self, val: f32) {
        self.trace.specular_sheen = val.clamp(0.0, 5.0);
    }

    pub fn ssrc(&self) -> bool {
        self.trace.ssrc
    }

    pub fn set_ssrc(&mut self, val: bool) {
        self.trace.ssrc = val;
    }

    pub(crate) fn toggle_ssrc(&mut self) -> bool {
        self.trace.ssrc = !self.trace.ssrc;
        self.trace.ssrc
    }

    pub fn ssrc_strength(&self) -> f32 {
        self.trace.ssrc_strength
    }

    pub fn set_ssrc_strength(&mut self, val: f32) {
        self.trace.ssrc_strength = val.clamp(0.0, 3.0);
    }

    pub fn bounce_boost(&self) -> f32 {
        self.trace.bounce_boost
    }

    pub fn set_bounce_boost(&mut self, val: f32) {
        self.trace.bounce_boost = val.clamp(0.2, 4.0);
    }

    pub fn baked_light(&self) -> bool {
        self.trace.baked_light
    }

    pub fn set_baked_light(&mut self, val: bool) {
        self.trace.baked_light = val;
    }

    pub(crate) fn toggle_baked_light(&mut self) -> bool {
        self.trace.baked_light = !self.trace.baked_light;
        self.trace.baked_light
    }

    pub fn diminish(&self) -> f32 {
        self.trace.diminish
    }

    pub fn set_diminish(&mut self, val: f32) {
        self.trace.diminish = val.clamp(0.0, 3.0);
    }

    pub fn light_blend(&self) -> f32 {
        self.trace.light_blend
    }

    pub fn set_light_blend(&mut self, val: f32) {
        self.trace.light_blend = val.clamp(0.0, 1.0);
    }

    pub fn voxel_blend(&self) -> f32 {
        self.trace.voxel_blend
    }

    pub fn set_voxel_blend(&mut self, val: f32) {
        self.trace.voxel_blend = val.clamp(0.0, 1.0);
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
            let previous_max_light = trace.max_light;
            let (restart_bake, clear_visible) = ui.paint(
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
            let ceiling_changed = previous_max_light != trace.max_light;
            crate::platform::maintain_pointer_capture();
            if ceiling_changed {
                self.accum_key = None;
                self.accum_count = 0;
            }
            if restart_bake {
                self.restart_light_bake();
            }
            if clear_visible {
                self.clear_visible();
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (encoder, view);
        }
    }

    /// The pointer is over the path-tracer panel, so a click stays there.
    pub fn pointer_over_ui(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ui.pointer_over_ui()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Play mode draws monsters and items; bake leaves the map empty.
    pub fn include_actors(&self) -> bool {
        self.trace.include_actors()
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
        screen_weight: f32,
    ) -> bool {
        let light = [
            self.trace.emission,
            self.trace.environment,
            self.trace.bounces as f32,
            if self.trace.stores_light() { 1.0 } else { 0.0 },
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
            up: [scene.up[0], scene.up[1], scene.up[2], self.trace.flashlight],
            forward: [
                scene.forward[0],
                scene.forward[1],
                scene.forward[2],
                screen_weight,
            ],
            viewport: [vx, vy, vw, vh],
            light,
            fill: [
                self.trace.min_light,
                if self.trace.playing() { 1.0 } else { 0.0 },
                crate::hw::VOXEL_LIGHT_ROWS.max(0) as f32,
                self.trace.max_light,
            ],
            pbr: [
                self.trace.normal_strength,
                self.trace.roughness_scale,
                self.trace.specular_sheen,
                self.trace.bounce_boost,
            ],
            // Keep the original texture-emission path for ceilings. Dedicated
            // sources are only for standing lamps/torches; their records leave
            // the panel-coverage flag off, so ceiling emission is not suppressed.
            fixtures: unsafe { crate::fixture_lights::collect() },
        };
        queue.write_buffer(&ray.uniform, 0, bytemuck::bytes_of(&uniform));
        true
    }

    fn ensure_world(&mut self, atlas: Option<(u32, &[u8], &[u8], &[u8])>) {
        let Some((size, rgba, emit, pbr)) = atlas else {
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
            // Atlas pixels can refresh without moving light rectangles. The
            // light map is cleared only when packing resets (`light_reset`).
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
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &world.pbr,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                pbr,
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
        let pbr_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("doom_pbr"),
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
                texture: &pbr_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pbr,
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
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let pbr_view = pbr_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("doom_atlas_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let light = world_light_read(&self.device, crate::hw::LIGHTMAP_SIZE);
        let light_write = world_light_write(&self.device, crate::hw::LIGHTMAP_SIZE);
        clear_light(&self.queue, &light, crate::hw::LIGHTMAP_SIZE);
        let light_view = light.create_view(&wgpu::TextureViewDescriptor::default());
        let emission_view = emission_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("doom_world_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&light_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&emission_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&pbr_view),
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
                    entry_point: Some("fs_weapon"),
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
        let pipeline_msaa = world_msaa_pipeline(&self.device, &pipeline_layout, &shader, self.config.format, false);
        let weapon_pipeline_msaa = world_msaa_pipeline(&self.device, &pipeline_layout, &shader, self.config.format, true);
        let voxel_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("doom_voxel_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_voxel"),
                    buffers: &[voxel_vertex_layout(), voxel_instance_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_voxel"),
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
        let voxel_pipeline_msaa = voxel_msaa_pipeline(&self.device, &pipeline_layout, &shader, self.config.format);
        let (msaa_color, msaa_view) = msaa_color_target(&self.device, 1, 1, self.config.format);
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
        let voxel_vertices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_voxel_vertices"),
            size: 64 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let voxel_indices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_voxel_indices"),
            size: 64 * 1024,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let voxel_instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("doom_voxel_instances"),
            size: 4 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_hits = light_hit_buffer(&self.device, LIGHT_HIT_CAP);
        // Browsers display uploaded bakes; the native baking pipeline blocks
        // on validation and requires compute limits unavailable on WebGL.
        #[cfg(not(target_arch = "wasm32"))]
        let light_bins = Some(create_light_bins(&self.device, &light, &light_write, &light_hits)
            .expect("light bin pipelines"));
        #[cfg(target_arch = "wasm32")]
        let light_bins = None;
        self.world = Some(WorldGpu {
            atlas: atlas_tex,
            emission: emission_tex,
            pbr: pbr_tex,
            light,
            light_write,
            light_hits,
            light_bins,
            atlas_size: size,
            bind_group,
            pipeline,
            weapon_pipeline,
            voxel_pipeline,
            pipeline_msaa,
            weapon_pipeline_msaa,
            voxel_pipeline_msaa,
            _msaa_color: msaa_color,
            msaa_view,
            depth_samples: 1,
            depth,
            depth_view,
            depth_size: (self.config.width, self.config.height),
            uniform,
            vertices,
            indices,
            vertex_cap: 64 * 1024,
            index_cap: 64 * 1024,
            voxel_mesh_cache: std::collections::HashMap::new(),
            voxel_vertices_cpu: Vec::new(),
            voxel_indices_cpu: Vec::new(),
            voxel_mesh_dirty: false,
            voxel_vertices,
            voxel_indices,
            voxel_instances,
            voxel_vert_cap: 64 * 1024,
            voxel_index_cap: 64 * 1024,
            voxel_inst_cap: 4 * 1024,
            voxel_instances_len: 0,
            ray_query: self.ray_query,
            ray: None,
        });
    }
}

const VERTEX_ATTRS: [wgpu::VertexAttribute; 7] = [
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
    // Surface tag: -1 directed wall, 0 ordinary geometry, +1 sky.
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 52,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 56,
        shader_location: 5,
    },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WorldUniform {
    view_proj: [[f32; 4]; 4],
    fill: [f32; 4],
    pbr: [f32; 4],
    extra: [f32; 4],
}

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::hw::Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRS,
    }
}

const VOXEL_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
];

const VOXEL_INSTANCE_ATTRS: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 12,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 16,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 20,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 6,
    },
];

pub(super) fn voxel_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::voxel::VoxelVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VOXEL_VERTEX_ATTRS,
    }
}

pub(super) fn voxel_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::hw::VoxelInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &VOXEL_INSTANCE_ATTRS,
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
    return vec4<f32>(textureSample(image, image_sampler, in.uv).rgb, 1.0);
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
    size: [f32; 4],
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
            size: [
                width as f32,
                height as f32,
                trace.flashlight,
                trace.min_light
                    + if trace.playing() { 100.0 } else { 0.0 }
                    + if trace.world_space { 10.0 } else { 0.0 },
            ],
        }
    }

    fn same(&self, other: &Self) -> bool {
        near(self.eye, other.eye)
            && near(self.right, other.right)
            && near(self.up, other.up)
            && near(self.forward, other.forward)
            && near(self.light, other.light)
            && near(self.size, other.size)
    }
}

fn near<const N: usize>(a: [f32; N], b: [f32; N]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4)
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

impl WorldGpu {
    fn draw_voxels<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, scene: &crate::hw::Scene, msaa: bool) {
        if scene.voxel_batches.is_empty() || self.voxel_instances_len == 0 {
            return;
        }
        pass.set_pipeline(if msaa { &self.voxel_pipeline_msaa } else { &self.voxel_pipeline });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.voxel_vertices.slice(..));
        pass.set_vertex_buffer(1, self.voxel_instances.slice(..));
        pass.set_index_buffer(self.voxel_indices.slice(..), wgpu::IndexFormat::Uint32);

        let mut inst_start: u32 = 0;
        for batch in &scene.voxel_batches {
            let count = batch.instances.len() as u32;
            if count == 0 {
                continue;
            }
            let key = Arc::as_ptr(&batch.mesh) as usize;
            if let Some(&(first_index, index_count)) = self.voxel_mesh_cache.get(&key) {
                pass.draw_indexed(
                    first_index..(first_index + index_count),
                    0,
                    inst_start..(inst_start + count),
                );
            }
            inst_start += count;
        }
    }
}

fn draw_scene<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    world: &'a WorldGpu,
    scene: &crate::hw::Scene,
    trace: bool,
    msaa: bool,
) {
    pass.set_vertex_buffer(0, world.vertices.slice(..));
    pass.set_index_buffer(world.indices.slice(..), wgpu::IndexFormat::Uint32);
    let world_n = scene.world_indices.min(scene.indices.len() as u32);
    if trace {
        let ray = world.ray.as_ref().expect("ray resources");
        pass.set_pipeline(&ray.pipeline);
        pass.set_bind_group(0, &ray.bind_group, &[]);
        pass.draw(0..6, 0..1);
    } else if world_n > 0 {
        pass.set_pipeline(if msaa { &world.pipeline_msaa } else { &world.pipeline });
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(0..world_n, 0, 0..1);
    }
    if !trace {
        world.draw_voxels(pass, scene, msaa);
    }
}

fn draw_weapon<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    world: &'a WorldGpu,
    scene: &crate::hw::Scene,
    msaa: bool,
) {
    let world_n = scene.world_indices.min(scene.indices.len() as u32);
    let total = scene.indices.len() as u32;
    if total > world_n {
        pass.set_vertex_buffer(0, world.vertices.slice(..));
        pass.set_index_buffer(world.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_pipeline(if msaa { &world.weapon_pipeline_msaa } else { &world.weapon_pipeline });
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
    let sprites = scene.opaque_indices.min(world_n);
    if world_n > sprites {
        pass.set_pipeline(&world.pipeline);
        pass.set_bind_group(0, &world.bind_group, &[]);
        pass.draw_indexed(sprites..world_n, 0, 0..1);
    }
    world.draw_voxels(pass, scene, false);
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
    let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("doom_play_blit_nearest"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
    BlitGpu {
        pipeline,
        layout,
        sampler,
        nearest,
    }
}

fn depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    depth_target_samples(device, width, height, 1)
}

pub(crate) fn depth_target_samples(device: &wgpu::Device, width: u32, height: u32, samples: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("doom_depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

    // Voxel mesh caching and static geometry upload
    for batch in &scene.voxel_batches {
        let key = Arc::as_ptr(&batch.mesh) as usize;
        if !world.voxel_mesh_cache.contains_key(&key) {
            let (v, i) = batch.mesh.build_gpu_mesh();
            let base_vertex = world.voxel_vertices_cpu.len() as u32;
            let first_index = world.voxel_indices_cpu.len() as u32;
            let index_count = i.len() as u32;

            world.voxel_vertices_cpu.extend_from_slice(&v);
            world.voxel_indices_cpu.extend(i.iter().map(|&idx| idx + base_vertex));
            world.voxel_mesh_cache.insert(key, (first_index, index_count));
            world.voxel_mesh_dirty = true;
        }
    }
    if world.voxel_mesh_dirty {
        world.voxel_mesh_dirty = false;
        let v_bytes = bytemuck::cast_slice(&world.voxel_vertices_cpu);
        let i_bytes = bytemuck::cast_slice(&world.voxel_indices_cpu);
        if v_bytes.len() as u64 > world.voxel_vert_cap {
            let size = (v_bytes.len() as u64).next_power_of_two().max(64 * 1024);
            world.voxel_vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("doom_voxel_vertices"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            world.voxel_vert_cap = size;
        }
        if i_bytes.len() as u64 > world.voxel_index_cap {
            let size = (i_bytes.len() as u64).next_power_of_two().max(64 * 1024);
            world.voxel_indices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("doom_voxel_indices"),
                size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            world.voxel_index_cap = size;
        }
        if !v_bytes.is_empty() {
            queue.write_buffer(&world.voxel_vertices, 0, v_bytes);
        }
        if !i_bytes.is_empty() {
            queue.write_buffer(&world.voxel_indices, 0, i_bytes);
        }
    }

    // Dynamic instance data for this frame
    let total_instances: usize = scene.voxel_batches.iter().map(|b| b.instances.len()).sum();
    world.voxel_instances_len = total_instances;
    if total_instances > 0 {
        let mut inst_cpu = Vec::with_capacity(total_instances);
        for batch in &scene.voxel_batches {
            inst_cpu.extend_from_slice(&batch.instances);
        }
        let inst_bytes = bytemuck::cast_slice(&inst_cpu);
        if inst_bytes.len() as u64 > world.voxel_inst_cap {
            let size = (inst_bytes.len() as u64).next_power_of_two().max(4 * 1024);
            world.voxel_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("doom_voxel_instances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            world.voxel_inst_cap = size;
        }
        queue.write_buffer(&world.voxel_instances, 0, inst_bytes);
    }
}

/// Packed [`crate::hw::Vertex`] size. The ray shader reads 14 floats per vertex.
const VERTEX_STRIDE: u64 = 64;
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
    /// x: emissive scale, y: environment scale, z: bounce count, w: world light.
    light: [f32; 4],
    /// x: minimum display light.
    fill: [f32; 4],
    /// x: normal strength, y: roughness scale, z: specular sheen.
    pbr: [f32; 4],
    fixtures: crate::fixture_lights::FixtureLights,
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
    let (layout, pipeline, present_pipeline, uniform) = if let Some(ray) = kept {
        (ray.layout, ray.pipeline, ray.present_pipeline, ray.uniform)
    } else {
        let (layout, pipeline, uniform) = create_ray_pipeline_mode(device, wgpu::TextureFormat::Rgba16Float, true)?;
        let present = create_ray_present_pipeline(device, &layout, format)?;
        (layout, pipeline, present, uniform)
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
        &world.pbr,
        &world.light,
        &world.light_write,
        &world.vertices,
        &world.indices,
        &world.light_hits,
        &tlas,
    );
    world.ray = Some(RayGpu {
        pipeline,
        present_pipeline,
        present_bind: None,
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
    create_ray_pipeline_mode(device, format, false)
}

fn create_ray_pipeline_mode(device: &wgpu::Device, format: wgpu::TextureFormat, linear: bool)
    -> Result<(wgpu::BindGroupLayout, wgpu::RenderPipeline, wgpu::Buffer), String> {
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
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
        ],
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_ray_camera"),
        size: std::mem::size_of::<RayUniform>() as u64,
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
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("LINEAR_OUTPUT", if linear { 1.0 } else { 0.0 })],
                ..Default::default()
            },
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

fn create_ray_present_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doom_ray_present_shader"), source: wgpu::ShaderSource::Wgsl(RAY_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("doom_ray_present_layout"), bind_group_layouts: &[Some(layout)], immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("doom_ray_present_pipeline"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_ray"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_present"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
        primitive: Default::default(), depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Always)),
        multisample: Default::default(), multiview_mask: None, cache: None,
    });
    if let Some(error) = pollster::block_on(scope.pop()) { return Err(error.to_string()); }
    Ok(pipeline)
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

fn clear_light(queue: &wgpu::Queue, light: &wgpu::Texture, size: u32) {
    let side = size.max(1) as usize;
    let row = side * 8;
    let stride = row.div_ceil(256) * 256;
    // rgba16float zeros. Alpha is the sample count: untouched stays 0 so the
    // sparse bake only writes texels that actually got light (not the whole
    // 4096² atlas). Display still reads rgb, so unlit surfaces stay black.
    let pixels = vec![0u8; stride * side];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: light,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride as u32),
            rows_per_image: Some(side as u32),
        },
        light_extent(size),
    );
}

fn light_extent(size: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.max(1),
        height: size.max(1),
        depth_or_array_layers: 1,
    }
}

fn world_light_read(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("doom_world_light"),
        size: light_extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn world_light_write(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("doom_world_light_write"),
        size: light_extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

#[cfg(test)]
pub(crate) fn test_dummy_pbr(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let pbr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy_pbr"),
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
            texture: &pbr,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[128, 128, 204, 0],
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
    pbr
}

/// Ray targets never exceed 1080p at 16:9.
const LIGHT_HIT_CAP: u64 = 1920 * 1080;
/// Open-addressed table. Big enough that every ray pixel can land in its own slot.
const LIGHT_SLOTS: u32 = 1 << 22;

struct LightBins {
    bins: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    bin_pipeline: wgpu::ComputePipeline,
    save_pipeline: wgpu::ComputePipeline,
    clear_visible_pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinParams {
    pixels: u32,
    light_size: u32,
    slots: u32,
    voxel_rows: u32,
}

const LIGHT_BIN_SHADER: &str = r#"
struct LightHit {
    r: f32,
    g: f32,
    b: f32,
    packed: u32,
}

struct BinParams {
    pixels: u32,
    light_size: u32,
    slots: u32,
    voxel_rows: u32,
}

@group(0) @binding(0) var<storage, read> light_hits: array<LightHit>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(2) var light_read: texture_2d<f32>;
@group(0) @binding(3) var light_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BinParams;

fn hash_texel(x: u32) -> u32 {
    var h = x * 0x9e3779b9u;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    return h;
}

@compute @workgroup_size(64)
fn cs_bin(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.pixels) {
        return;
    }
    let hit = light_hits[i];
    if (hit.packed == 0u) {
        return;
    }
    let mask = params.slots - 1u;
    var slot = hash_texel(hit.packed) & mask;
    // Stochastic rounding keeps sub-1/64 bounce energy instead of always
    // truncating it to black. The high 16 row bits carry a fresh frame seed.
    let seed = hash_texel(i ^ hit.packed ^ (params.voxel_rows >> 16u) * 0x9e3779b9u);
    let jitter_r = f32(hash_texel(seed ^ 0x1234567u) & 0xffffffu) / 16777216.0;
    let jitter_g = f32(hash_texel(seed ^ 0x7654321u) & 0xffffffu) / 16777216.0;
    let jitter_b = f32(hash_texel(seed ^ 0xabcdefu) & 0xffffffu) / 16777216.0;
    let add_r = u32(clamp(hit.r, 0.0, 65535.0) * 64.0 + jitter_r);
    let add_g = u32(clamp(hit.g, 0.0, 65535.0) * 64.0 + jitter_g);
    let add_b = u32(clamp(hit.b, 0.0, 65535.0) * 64.0 + jitter_b);
    for (var probe = 0u; probe < 128u; probe = probe + 1u) {
        let at = slot * 5u;
        let found = atomicCompareExchangeWeak(&bins[at], 0u, hit.packed);
        if (found.exchanged || found.old_value == hit.packed) {
            atomicAdd(&bins[at + 1u], add_r);
            atomicAdd(&bins[at + 2u], add_g);
            atomicAdd(&bins[at + 3u], add_b);
            atomicAdd(&bins[at + 4u], 1u);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

@compute @workgroup_size(256)
fn cs_save(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if (slot >= params.slots) {
        return;
    }
    let base = slot * 5u;
    let packed = atomicLoad(&bins[base]);
    if (packed == 0u) {
        return;
    }
    let n = atomicLoad(&bins[base + 4u]);
    if (n == 0u) {
        return;
    }
    let texel = packed - 1u;
    let side = params.light_size;
    let coord = vec2<i32>(i32(texel % side), i32(texel / side));
    let scale = 64.0;
    let avg = vec3<f32>(
        f32(atomicLoad(&bins[base + 1u])) / scale / f32(n),
        f32(atomicLoad(&bins[base + 2u])) / scale / f32(n),
        f32(atomicLoad(&bins[base + 3u])) / scale / f32(n),
    );
    let old = textureLoad(light_read, coord, 0);
    // Every look at this texel adds one sample, whether the camera moved or not.
    // Walking the room keeps filling surfaces in.
    let voxel_rows = params.voxel_rows & 0xffffu;
    let voxel_y0 = i32(params.light_size) - i32(voxel_rows);
    let moving = voxel_rows > 0u && coord.y >= voxel_y0;
    // Actors move through light. Cap history so their patch keeps following
    // the room instead of locking the first noisy looks forever.
    let cap = select(1024.0, 24.0, moving);
    let count = min(old.a, cap);
    var weight = 1.0 / (count + 1.0);
    if (moving) {
        weight = max(weight, 0.15);
    }
    let saved = mix(old.rgb, avg, weight);
    textureStore(light_write, coord, vec4<f32>(saved, min(count + 1.0, cap)));
}

@compute @workgroup_size(64)
fn cs_clear_visible(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.pixels) {
        return;
    }
    let hit = light_hits[i];
    if (hit.packed == 0u) {
        return;
    }
    let texel = hit.packed - 1u;
    let side = params.light_size;
    let coord = vec2<i32>(i32(texel % side), i32(texel / side));
    textureStore(light_write, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
"#;

fn light_hit_buffer(device: &wgpu::Device, pixels: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_light_hits"),
        size: pixels.max(1) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_light_bins(
    device: &wgpu::Device,
    light: &wgpu::Texture,
    light_write: &wgpu::Texture,
    hits: &wgpu::Buffer,
) -> Result<LightBins, String> {
    create_light_bins_sized(device, light, light_write, hits, LIGHT_SLOTS)
}

fn create_light_bins_sized(device: &wgpu::Device, light: &wgpu::Texture,
    light_write: &wgpu::Texture, hits: &wgpu::Buffer, slots: u32) -> Result<LightBins, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doom_light_bin_shader"),
        source: wgpu::ShaderSource::Wgsl(LIGHT_BIN_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("doom_light_bin_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bins = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_light_bins"),
        size: u64::from(slots) * 5 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doom_light_bin_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_view = light.create_view(&wgpu::TextureViewDescriptor::default());
    let light_write_view = light_write.create_view(&wgpu::TextureViewDescriptor::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("doom_light_bin_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: hits.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&light_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&light_write_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("doom_light_bin_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let bin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("doom_light_bin"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_bin"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let save_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("doom_light_save"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_save"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let clear_visible_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("doom_light_clear_visible"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_clear_visible"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(err.to_string());
    }
    Ok(LightBins {
        bins,
        uniform,
        bind,
        bin_pipeline,
        save_pipeline,
        clear_visible_pipeline,
    })
}

fn copy_light(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    size: u32,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        light_extent(size),
    );
}

fn ray_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    atlas: &wgpu::Texture,
    emission: &wgpu::Texture,
    pbr: &wgpu::Texture,
    light: &wgpu::Texture,
    light_write: &wgpu::Texture,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    hits: &wgpu::Buffer,
    tlas: &wgpu::Tlas,
) -> wgpu::BindGroup {
    let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
    let emit_view = emission.create_view(&wgpu::TextureViewDescriptor::default());
    let pbr_view = pbr.create_view(&wgpu::TextureViewDescriptor::default());
    let light_view = light.create_view(&wgpu::TextureViewDescriptor::default());
    let light_write_view = light_write.create_view(&wgpu::TextureViewDescriptor::default());
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
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&light_write_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&light_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: hits.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(&pbr_view),
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod raster_test;

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
            let verts: [f32; 48] = [
                -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
                1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
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
            let light = world_light_read(&device, 1);
            let light_write = world_light_write(&device, 1);
            clear_light(&queue, &light, 1);
            let hits = light_hit_buffer(&device, 1);
            let _bins = create_light_bins(&device, &light, &light_write, &hits).expect("light bins");
            let pbr = test_dummy_pbr(&device, &queue);
            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &pbr, &light, &light_write, &vertices,
                &index_buf, &hits, &tlas,
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
                fill: [0.0, 0.0, 0.0, 0.0],
                pbr: [1.0, 1.0, 1.0, 0.0],
                fixtures: Default::default(),
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

            // Not sky. White albedo, emission texel 255. The surface shows
            // the glow blended halfway with the texture, so the pixel is
            // bright without clipping to white. A missed sample stays black.
            let verts: [f32; 48] = [
                -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
                1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
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
            let light = world_light_read(&device, 1);
            let light_write = world_light_write(&device, 1);
            clear_light(&queue, &light, 1);
            let hits = light_hit_buffer(&device, 1);
            let _bins = create_light_bins(&device, &light, &light_write, &hits).expect("light bins");
            let pbr = test_dummy_pbr(&device, &queue);
            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &pbr, &light, &light_write, &vertices,
                &index_buf, &hits, &tlas,
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
                fill: [0.0, 0.0, 0.0, 0.0],
                pbr: [1.0, 1.0, 1.0, 0.0],
                fixtures: Default::default(),
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
                px[0] > 80 && px[1] > 80 && px[2] > 80,
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
            let verts: [f32; 48] = [
                -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
                2.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
                0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
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

            let light = world_light_read(&device, 1);
            let light_write = world_light_write(&device, 1);
            clear_light(&queue, &light, 1);
            let hits = light_hit_buffer(&device, 1);
            let _bins = create_light_bins(&device, &light, &light_write, &hits).expect("light bins");
            let pbr = test_dummy_pbr(&device, &queue);
            let bind = ray_bind_group(
                &device, &layout, &uniform, &atlas, &emission, &pbr, &light, &light_write, &vertices,
                &index_buf, &hits, &tlas,
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
                fill: [0.0, 0.0, 0.0, 0.0],
                pbr: [1.0, 1.0, 1.0, 0.0],
                fixtures: Default::default(),
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
                px[0] > 80 && px[1] > 80 && px[2] > 80,
                "backface hit missed the lit texel: {px:?}"
            );
        });
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod ray_sidedef_test;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod fixture_light_test;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod msaa_test;

mod msaa;
use msaa::{msaa_color_target, voxel_msaa_pipeline, world_msaa_pipeline};

mod ssrc;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod linear_accum_test;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod smaa_test;

#[cfg(not(target_arch = "wasm32"))]
mod neural;
