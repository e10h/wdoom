//! Screen-Space Radiance Cascades (SSRC)
//!
//! Provides real-time, alias-free hierarchical radiance cascades.
//! Radiance is partitioned into nested distance intervals with increasing
//! angular resolution further away, merging from far to near:
//! - Cascade 2: Far field ([24, 160] px, 16 directions, 4x4 atlas)
//! - Cascade 1: Mid/Near field ([1.5, 24] px, 4 directions, 2x2 atlas, merges 4 C2 rays)
//! - Composite: Per-pixel Lambertian cosine weighting of C1 directional radiance

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrcUniforms {
    viewport: [f32; 4],
    params: [f32; 4], // x: strength, y: radius_scale, z: frame_seed, w: 0.0
}

const SSRC_SHADER: &str = r#"
struct SsrcUniforms {
    viewport: vec4<f32>,
    params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
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

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_upper: texture_2d<f32>;
@group(0) @binding(3) var s_sampler: sampler;
@group(0) @binding(4) var<uniform> u: SsrcUniforms;

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn linear_depth(d: f32) -> f32 {
    return 1.0 / max(1.0 - d, 0.0001);
}

fn fetch_depth(coord: vec2<i32>, dims: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(t_depth, clamped, 0);
}

fn reconstruct_pos(px: vec2<i32>, dims: vec2<i32>, z: f32) -> vec3<f32> {
    let ndc_x = (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0;
    let aspect = f32(dims.x) / max(f32(dims.y), 1.0);
    return vec3<f32>(ndc_x * z, (ndc_y / aspect) * z, z);
}

fn reconstruct_normal(px: vec2<i32>, dims: vec2<i32>, p0: vec3<f32>, z0: f32) -> vec3<f32> {
    let z_r = linear_depth(fetch_depth(px + vec2<i32>(1, 0), dims));
    let z_l = linear_depth(fetch_depth(px - vec2<i32>(1, 0), dims));
    let z_d = linear_depth(fetch_depth(px + vec2<i32>(0, 1), dims));
    let z_u = linear_depth(fetch_depth(px - vec2<i32>(0, 1), dims));

    var dp_dx: vec3<f32>;
    if (abs(z_r - z0) < abs(z_l - z0)) {
        dp_dx = reconstruct_pos(px + vec2<i32>(1, 0), dims, z_r) - p0;
    } else {
        dp_dx = p0 - reconstruct_pos(px - vec2<i32>(1, 0), dims, z_l);
    }

    var dp_dy: vec3<f32>;
    if (abs(z_d - z0) < abs(z_u - z0)) {
        dp_dy = reconstruct_pos(px + vec2<i32>(0, 1), dims, z_d) - p0;
    } else {
        dp_dy = p0 - reconstruct_pos(px - vec2<i32>(0, 1), dims, z_u);
    }

    var n = cross(dp_dx, dp_dy);
    let len = length(n);
    if (len < 1e-4) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    n = n / len;
    if (dot(n, p0) > 0.0) {
        n = -n;
    }
    return n;
}

// ----------------------------------------------------------------------------
// Cascade 2: Far Field (interval [24.0, 160.0] px, 16 directions, 4x4 atlas)
// ----------------------------------------------------------------------------
@fragment
fn fs_cascade_2(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(t_color);
    let dims = vec2<i32>(dims_u);
    let frag = vec2<i32>(in.clip_position.xy);

    let tile_w = f32(dims.x) / 4.0;
    let tile_h = f32(dims.y) / 4.0;
    let dx = clamp(i32(f32(frag.x) / tile_w), 0, 3);
    let dy = clamp(i32(f32(frag.y) / tile_h), 0, 3);
    let dir_idx = dy * 4 + dx;

    let u_probe = fract(f32(frag.x) / tile_w);
    let v_probe = fract(f32(frag.y) / tile_h);
    let probe_px = vec2<i32>(vec2<f32>(u_probe * f32(dims.x), v_probe * f32(dims.y)));

    let d0 = fetch_depth(probe_px, dims);
    if (d0 >= 0.9999) {
        return vec4<f32>(0.0);
    }
    let z0 = linear_depth(d0);
    let p0 = reconstruct_pos(probe_px, dims, z0);

    let angle = (f32(dir_idx) + 0.5) / 16.0 * 6.2831853;
    let dir = vec2<f32>(cos(angle), sin(angle));

    let r_start = 24.0 * u.params.y;
    let r_end = 160.0 * u.params.y;
    let inv_dims = 1.0 / vec2<f32>(dims);

    var ray_radiance = vec3<f32>(0.0);

    for (var step = 1; step <= 4; step = step + 1) {
        let dist = mix(r_start, r_end, (f32(step) - 0.5) / 4.0);
        let sample_px = vec2<f32>(probe_px) + dir * dist;
        let sample_coord = vec2<i32>(sample_px);

        if (sample_coord.x < 0 || sample_coord.x >= dims.x || sample_coord.y < 0 || sample_coord.y >= dims.y) {
            break;
        }

        let ds = fetch_depth(sample_coord, dims);
        let sample_uv = sample_px * inv_dims;
        let edge_dist = min(min(sample_uv.x, 1.0 - sample_uv.x), min(sample_uv.y, 1.0 - sample_uv.y));
        let edge_fade = clamp(edge_dist * 20.0, 0.0, 1.0);

        if (ds >= 0.9999) {
            let sky_sample = textureSampleLevel(t_color, s_sampler, sample_uv, 0.0).rgb;
            ray_radiance = sky_sample * (1.2 * edge_fade);
            break;
        }

        let zs = linear_depth(ds);
        let depth_diff = z0 - zs;
        let hit_sample = textureSampleLevel(t_color, s_sampler, sample_uv, 0.0);
        let hit_color = hit_sample.rgb;

        let is_explosion = abs(hit_sample.a - 0.20) < 0.10;
        let is_drop = abs(hit_sample.a - 0.40) < 0.10;

        if (depth_diff > max(z0 * 0.08, 12.0) && !is_explosion && !is_drop) {
            break;
        }

        let ps = reconstruct_pos(sample_coord, dims, zs);
        let dist_3d = length(ps - p0);

        if (is_explosion) {
            if (dist_3d < 220.0) {
                let lum = dot(hit_color, vec3<f32>(0.299, 0.587, 0.114));
                let lum_norm = clamp((lum - 0.35) / 0.65, 0.0, 1.0);
                let emit_boost = 14.0 + pow(lum_norm, 1.5) * 40.0;
                let r_norm = dist_3d / 220.0;
                let falloff = (1.0 - r_norm) * (1.0 - r_norm);
                ray_radiance = hit_color * (emit_boost * falloff * edge_fade);
                break;
            }
        }
    }

    return vec4<f32>(ray_radiance, 1.0);
}

// ----------------------------------------------------------------------------
// Cascade 1: Mid/Near Field (interval [1.5, 24.0] px, 4 directions, merges 4 C2 rays)
// ----------------------------------------------------------------------------
@fragment
fn fs_cascade_1(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(t_color);
    let dims = vec2<i32>(dims_u);
    let frag = vec2<i32>(in.clip_position.xy);

    let tile_w = f32(dims.x) / 2.0;
    let tile_h = f32(dims.y) / 2.0;
    let dx = clamp(i32(f32(frag.x) / tile_w), 0, 1);
    let dy = clamp(i32(f32(frag.y) / tile_h), 0, 1);
    let dir_idx = dy * 2 + dx;

    let u_probe = fract(f32(frag.x) / tile_w);
    let v_probe = fract(f32(frag.y) / tile_h);
    let probe_px = vec2<i32>(vec2<f32>(u_probe * f32(dims.x), v_probe * f32(dims.y)));

    let d0 = fetch_depth(probe_px, dims);
    if (d0 >= 0.9999) {
        return vec4<f32>(0.0);
    }
    let z0 = linear_depth(d0);
    let p0 = reconstruct_pos(probe_px, dims, z0);

    let angle = (f32(dir_idx) + 0.5) / 4.0 * 6.2831853;
    let dir = vec2<f32>(cos(angle), sin(angle));

    let r_start = 1.5 * u.params.y;
    let r_end = 24.0 * u.params.y;
    let inv_dims = 1.0 / vec2<f32>(dims);

    var ray_radiance = vec3<f32>(0.0);
    var hit_found = false;

    for (var step = 1; step <= 5; step = step + 1) {
        let dist = mix(r_start, r_end, (f32(step) - 0.5) / 5.0);
        let sample_px = vec2<f32>(probe_px) + dir * dist;
        let sample_coord = vec2<i32>(sample_px);

        if (sample_coord.x < 0 || sample_coord.x >= dims.x || sample_coord.y < 0 || sample_coord.y >= dims.y) {
            break;
        }

        let ds = fetch_depth(sample_coord, dims);
        let sample_uv = sample_px * inv_dims;
        let edge_dist = min(min(sample_uv.x, 1.0 - sample_uv.x), min(sample_uv.y, 1.0 - sample_uv.y));
        let edge_fade = clamp(edge_dist * 20.0, 0.0, 1.0);

        if (ds >= 0.9999) {
            let sky_sample = textureSampleLevel(t_color, s_sampler, sample_uv, 0.0).rgb;
            ray_radiance = sky_sample * (1.2 * edge_fade);
            hit_found = true;
            break;
        }

        let zs = linear_depth(ds);
        let depth_diff = z0 - zs;
        let hit_sample = textureSampleLevel(t_color, s_sampler, sample_uv, 0.0);
        let hit_color = hit_sample.rgb;

        let is_explosion = abs(hit_sample.a - 0.20) < 0.10;
        let is_drop = abs(hit_sample.a - 0.40) < 0.10;

        if (depth_diff > max(z0 * 0.08, 12.0) && !is_explosion && !is_drop) {
            hit_found = true;
            break;
        }

        let ps = reconstruct_pos(sample_coord, dims, zs);
        let dist_3d = length(ps - p0);

        if (is_explosion) {
            if (dist_3d < 220.0) {
                let lum = dot(hit_color, vec3<f32>(0.299, 0.587, 0.114));
                let lum_norm = clamp((lum - 0.35) / 0.65, 0.0, 1.0);
                let emit_boost = 18.0 + pow(lum_norm, 1.5) * 50.0;
                let r_norm = dist_3d / 220.0;
                let falloff = (1.0 - r_norm) * (1.0 - r_norm);
                ray_radiance = hit_color * (emit_boost * falloff * edge_fade);
                hit_found = true;
                break;
            }
        } else if (is_drop) {
            if (dist_3d < 50.0) {
                let r_norm = dist_3d / 50.0;
                let falloff = (1.0 - r_norm) * (1.0 - r_norm);
                ray_radiance = hit_color * (18.0 * falloff * edge_fade);
                hit_found = true;
                break;
            }
        } else {
            let thickness = max(z0 * 0.12, 16.0);
            if (depth_diff > 2.0 && depth_diff < thickness && dist_3d < 35.0) {
                let falloff = 1.0 / (1.0 + dist_3d * 0.05);
                ray_radiance = hit_color * (0.25 * falloff * edge_fade);
                hit_found = true;
                break;
            }
        }
    }

    if (!hit_found) {
        let ray_end = vec2<f32>(probe_px) + dir * r_end;
        let norm_pos = clamp(ray_end * inv_dims, vec2<f32>(0.0), vec2<f32>(1.0));
        var c2_sum = vec3<f32>(0.0);
        for (var k = 0; k < 4; k = k + 1) {
            let d2 = dir_idx * 4 + k;
            let d2x = d2 % 4;
            let d2y = d2 / 4;
            let uv2 = (vec2<f32>(f32(d2x), f32(d2y)) + norm_pos) / 4.0;
            c2_sum = c2_sum + textureSampleLevel(t_upper, s_sampler, uv2, 0.0).rgb;
        }
        ray_radiance = ray_radiance + c2_sum * 0.25;
    }

    return vec4<f32>(ray_radiance, 1.0);
}
"#;

const COMPOSITE_SHADER: &str = r#"
struct SsrcUniforms {
    viewport: vec4<f32>,
    params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
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

@group(0) @binding(0) var t_c1: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_sampler: sampler;
@group(0) @binding(3) var<uniform> u: SsrcUniforms;

fn linear_depth(d: f32) -> f32 {
    return 1.0 / max(1.0 - d, 0.0001);
}

fn fetch_depth(coord: vec2<i32>, dims: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(t_depth, clamped, 0);
}

fn reconstruct_pos(px: vec2<i32>, dims: vec2<i32>, z: f32) -> vec3<f32> {
    let ndc_x = (f32(px.x) + 0.5) / f32(dims.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(px.y) + 0.5) / f32(dims.y) * 2.0;
    let aspect = f32(dims.x) / max(f32(dims.y), 1.0);
    return vec3<f32>(ndc_x * z, (ndc_y / aspect) * z, z);
}

fn reconstruct_normal(px: vec2<i32>, dims: vec2<i32>, p0: vec3<f32>, z0: f32) -> vec3<f32> {
    let z_r = linear_depth(fetch_depth(px + vec2<i32>(1, 0), dims));
    let z_l = linear_depth(fetch_depth(px - vec2<i32>(1, 0), dims));
    let z_d = linear_depth(fetch_depth(px + vec2<i32>(0, 1), dims));
    let z_u = linear_depth(fetch_depth(px - vec2<i32>(0, 1), dims));

    var dp_dx: vec3<f32>;
    if (abs(z_r - z0) < abs(z_l - z0)) {
        dp_dx = reconstruct_pos(px + vec2<i32>(1, 0), dims, z_r) - p0;
    } else {
        dp_dx = p0 - reconstruct_pos(px - vec2<i32>(1, 0), dims, z_l);
    }

    var dp_dy: vec3<f32>;
    if (abs(z_d - z0) < abs(z_u - z0)) {
        dp_dy = reconstruct_pos(px + vec2<i32>(0, 1), dims, z_d) - p0;
    } else {
        dp_dy = p0 - reconstruct_pos(px - vec2<i32>(0, 1), dims, z_u);
    }

    var n = cross(dp_dx, dp_dy);
    let len = length(n);
    if (len < 1e-4) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    n = n / len;
    if (dot(n, p0) > 0.0) {
        n = -n;
    }
    return n;
}

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(t_c1);
    let dims = vec2<i32>(dims_u);
    let px = vec2<i32>(in.clip_position.xy);
    let d0 = fetch_depth(px, dims);

    if (d0 >= 0.9999) {
        discard;
    }

    let z0 = linear_depth(d0);
    let p0 = reconstruct_pos(px, dims, z0);
    let n0 = reconstruct_normal(px, dims, p0, z0);
    let inv_dims = 1.0 / vec2<f32>(dims);
    let norm_pos = (vec2<f32>(px) + 0.5) * inv_dims;

    var total_irradiance = vec3<f32>(0.0);

    for (var d = 0; d < 4; d = d + 1) {
        let angle = (f32(d) + 0.5) / 4.0 * 6.2831853;
        let dir = vec2<f32>(cos(angle), sin(angle));

        let d1x = d % 2;
        let d1y = d / 2;
        let uv1 = (vec2<f32>(f32(d1x), f32(d1y)) + norm_pos) / 2.0;
        let ray_radiance = textureSampleLevel(t_c1, s_sampler, uv1, 0.0).rgb;

        let dir_view = normalize(vec3<f32>(dir.x, -dir.y, 0.0));
        let ndotl = max(dot(n0, dir_view), 0.0);

        total_irradiance = total_irradiance + ray_radiance * ndotl;
    }

    let diffuse = (total_irradiance * 0.25) * u.params.x;
    return vec4<f32>(diffuse, 0.0);
}
"#;

pub struct SsrcPass {
    width: u32,
    height: u32,
    c2_texture: wgpu::Texture,
    c2_view: wgpu::TextureView,
    c1_texture: wgpu::Texture,
    c1_view: wgpu::TextureView,
    dummy_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    c2_pipeline: wgpu::RenderPipeline,
    c1_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    cascade_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    frame_index: u32,
}

impl SsrcPass {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let cascade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssrc_cascade_shader"),
            source: wgpu::ShaderSource::Wgsl(SSRC_SHADER.into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssrc_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssrc_cascade_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssrc_composite_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cascade_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssrc_cascade_pipeline_layout"),
            bind_group_layouts: &[Some(&cascade_layout)],
            immediate_size: 0,
        });

        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssrc_composite_pipeline_layout"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });

        let cascade_format = wgpu::TextureFormat::Rgba8Unorm;

        let c2_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssrc_c2_pipeline"),
            layout: Some(&cascade_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cascade_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cascade_shader,
                entry_point: Some("fs_cascade_2"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: cascade_format,
                    blend: None,
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

        let c1_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssrc_c1_pipeline"),
            layout: Some(&cascade_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cascade_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cascade_shader,
                entry_point: Some("fs_cascade_1"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: cascade_format,
                    blend: None,
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

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssrc_composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
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
            label: Some("ssrc_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssrc_dummy_texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: cascade_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (c2_texture, c2_view) = Self::create_cascade_target(device, width, height, "ssrc_c2");
        let (c1_texture, c1_view) = Self::create_cascade_target(device, width, height, "ssrc_c1");

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssrc_uniforms"),
            size: std::mem::size_of::<SsrcUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            c2_texture,
            c2_view,
            c1_texture,
            c1_view,
            dummy_view,
            uniform_buf,
            c2_pipeline,
            c1_pipeline,
            composite_pipeline,
            cascade_layout,
            composite_layout,
            sampler,
            frame_index: 0,
        }
    }

    fn create_cascade_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let (c2_tex, c2_v) = Self::create_cascade_target(device, width, height, "ssrc_c2");
        let (c1_tex, c1_v) = Self::create_cascade_target(device, width, height, "ssrc_c1");
        self.c2_texture = c2_tex;
        self.c2_view = c2_v;
        self.c1_texture = c1_tex;
        self.c1_view = c1_v;
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        strength: f32,
    ) {
        if strength <= 0.001 {
            return;
        }
        self.resize(device, width, height);
        self.frame_index = self.frame_index.wrapping_add(1);

        let uniforms = SsrcUniforms {
            viewport: [0.0, 0.0, width as f32, height as f32],
            params: [strength, 1.0, (self.frame_index % 64) as f32, 0.0],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Pass 1: Cascade 2 (Far Field: [24, 160] px, 16 directions) -> c2_view
        let c2_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssrc_c2_bind"),
            layout: &self.cascade_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssrc_c2_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.c2_view,
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
            pass.set_pipeline(&self.c2_pipeline);
            pass.set_bind_group(0, &c2_bind, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Cascade 1 (Mid Field: [1.5, 24] px, 4 directions, merges 4 C2 rays) -> c1_view
        let c1_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssrc_c1_bind"),
            layout: &self.cascade_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.c2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssrc_c1_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.c1_view,
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
            pass.set_pipeline(&self.c1_pipeline);
            pass.set_bind_group(0, &c1_bind, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 3: Composite (Reads c1_view & depth_view, writes target_view) -> target_view
        let composite_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssrc_composite_bind"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.c1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssrc_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
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
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bind, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssrc_pass_renders_without_validation_errors() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = match instance.request_adapter(&Default::default()).await {
                Ok(a) => a,
                Err(_) => return,
            };
            let (device, queue) = adapter.request_device(&Default::default()).await.expect("device");
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let width = 320;
            let height = 200;

            let color_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_color"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color_tex.create_view(&Default::default());

            let (_depth_tex, depth_view) = crate::gpu::depth_target_samples(&device, width, height, 1);

            let mut ssrc = SsrcPass::new(&device, format, width, height);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test_encoder") });

            ssrc.render(
                &mut encoder,
                &device,
                &queue,
                &color_view,
                &depth_view,
                &color_view,
                width,
                height,
                1.0,
            );

            queue.submit(Some(encoder.finish()));
        });
    }
}
