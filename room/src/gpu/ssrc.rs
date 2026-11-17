//! Screen-Space Radiance Cascades (SSRC)
//!
//! Provides real-time, noise-free secondary diffuse bounce and ambient fill.
//! Cascades partition screen-space raymarching into nested distance intervals,
//! allowing outdoor sky radiance to flood through windows into interior rooms
//! while adding natural contact ambient occlusion in corners.

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
@group(0) @binding(2) var s_sampler: sampler;
@group(0) @binding(3) var<uniform> u: SsrcUniforms;

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

@fragment
fn fs_cascade(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(t_color);
    let dims = vec2<i32>(dims_u);
    let px = vec2<i32>(in.clip_position.xy);
    let d0 = fetch_depth(px, dims);

    // Sky geometry does not receive indirect bounce
    if (d0 >= 0.9999) {
        return vec4<f32>(0.0);
    }

    let z0 = linear_depth(d0);
    let jitter = hash12(in.clip_position.xy + vec2<f32>(u.params.z * 17.0, u.params.z * 29.0));
    let base_angle = jitter * 6.2831853;

    var total_irradiance = vec3<f32>(0.0);
    var total_weight = 0.0;

    // 8 primary directions around the 2D horizon
    let NUM_DIRS = 8;
    let inv_dims = 1.0 / vec2<f32>(dims);
    let radius_scale = u.params.y;

    // Cascade intervals:
    // Cascade 0: [2.0, 12.0]   (near contact / self-bounce)
    // Cascade 1: [12.0, 48.0]  (room scale)
    // Cascade 2: [48.0, 180.0] (window / corridor scale)
    let intervals = array<vec2<f32>, 3>(
        vec2<f32>(2.0, 12.0) * radius_scale,
        vec2<f32>(12.0, 48.0) * radius_scale,
        vec2<f32>(48.0, 180.0) * radius_scale,
    );

    for (var i = 0; i < NUM_DIRS; i = i + 1) {
        let angle = base_angle + (f32(i) / f32(NUM_DIRS)) * 6.2831853;
        let dir = vec2<f32>(cos(angle), sin(angle));

        var ray_radiance = vec3<f32>(0.0);
        var hit_found = false;

        for (var c = 0; c < 3; c = c + 1) {
            if (hit_found) {
                break;
            }
            let r_start = intervals[c].x;
            let r_end = intervals[c].y;

            for (var step = 1; step <= 3; step = step + 1) {
                let dist = mix(r_start, r_end, (f32(step) - 0.5 + jitter * 0.3) / 3.0);
                let sample_px = vec2<f32>(px) + dir * dist;
                let sample_coord = vec2<i32>(sample_px);

                if (sample_coord.x < 0 || sample_coord.x >= dims.x || sample_coord.y < 0 || sample_coord.y >= dims.y) {
                    break;
                }

                let ds = fetch_depth(sample_coord, dims);

                if (ds >= 0.9999) {
                    // Reached the open sky through a window or opening:
                    let sky_sample = textureSampleLevel(t_color, s_sampler, sample_px * inv_dims, 0.0).rgb;
                    ray_radiance = sky_sample * 1.6;
                    hit_found = true;
                    break;
                }

                let zs = linear_depth(ds);
                let depth_diff = z0 - zs;

                // An occluding surface is in front of the ray within a plausible thickness
                let thickness = max(z0 * 0.12, 16.0);
                if (depth_diff > 2.0 && depth_diff < thickness) {
                    let hit_color = textureSampleLevel(t_color, s_sampler, sample_px * inv_dims, 0.0).rgb;
                    let falloff = 1.0 / (1.0 + dist * 0.02);
                    ray_radiance = hit_color * falloff;
                    hit_found = true;
                    break;
                }
            }
        }

        if (hit_found) {
            total_irradiance = total_irradiance + ray_radiance;
            total_weight = total_weight + 1.0;
        }
    }

    if (total_weight > 0.0) {
        return vec4<f32>(total_irradiance / f32(NUM_DIRS), 1.0);
    }
    return vec4<f32>(0.0);
}

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(t_color);
    let dims = vec2<i32>(dims_u);
    let px = vec2<i32>(in.clip_position.xy);
    let d0 = fetch_depth(px, dims);

    if (d0 >= 0.9999) {
        discard;
    }

    let z0 = linear_depth(d0);
    let inv_dims = 1.0 / vec2<f32>(dims);

    // Cross-bilateral depth-aware smoothing filter
    var sum_col = vec3<f32>(0.0);
    var sum_w = 0.0;

    let offsets = array<vec2<i32>, 9>(
        vec2<i32>(0, 0),
        vec2<i32>(-1, 0), vec2<i32>(1, 0),
        vec2<i32>(0, -1), vec2<i32>(0, 1),
        vec2<i32>(-2, 0), vec2<i32>(2, 0),
        vec2<i32>(0, -2), vec2<i32>(0, 2),
    );

    for (var i = 0; i < 9; i = i + 1) {
        let p = px + offsets[i] * 2;
        let d = fetch_depth(p, dims);
        let z = linear_depth(d);

        let dz = abs(z - z0);
        let spatial_w = select(0.5, 1.0, i == 0);
        let depth_w = exp(-dz / max(z0 * 0.08, 8.0));
        let w = spatial_w * depth_w;

        let col = textureSampleLevel(t_color, s_sampler, vec2<f32>(p) * inv_dims, 0.0).rgb;
        sum_col = sum_col + col * w;
        sum_w = sum_w + w;
    }

    let indirect = sum_col / max(sum_w, 0.001);
    let final_diffuse = indirect * u.params.x * 0.65;
    return vec4<f32>(final_diffuse, 0.0);
}
"#;

pub struct SsrcPass {
    width: u32,
    height: u32,
    irradiance_texture: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    cascade_pipeline: wgpu::RenderPipeline,
    cascade_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    frame_index: u32,
}

impl SsrcPass {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssrc_shader"),
            source: wgpu::ShaderSource::Wgsl(SSRC_SHADER.into()),
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

        let irradiance_format = wgpu::TextureFormat::Rgba8Unorm;

        let cascade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssrc_cascade_pipeline"),
            layout: Some(&cascade_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cascade"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: irradiance_format,
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

        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssrc_composite_pipeline_layout"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssrc_composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        let irradiance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssrc_irradiance_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: irradiance_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let irradiance_view = irradiance_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssrc_uniforms"),
            size: std::mem::size_of::<SsrcUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            irradiance_texture,
            irradiance_view,
            uniform_buf,
            cascade_pipeline,
            cascade_layout,
            composite_pipeline,
            composite_layout,
            sampler,
            frame_index: 0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let irradiance_format = wgpu::TextureFormat::Rgba8Unorm;
        self.irradiance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssrc_irradiance_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: irradiance_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.irradiance_view = self.irradiance_texture.create_view(&wgpu::TextureViewDescriptor::default());
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

        // Pass 1: Radiance Cascades raymarching -> irradiance_view
        let cascade_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssrc_cascade_bind_group"),
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
                label: Some("ssrc_cascade_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.irradiance_view,
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
            pass.set_pipeline(&self.cascade_pipeline);
            pass.set_bind_group(0, &cascade_bind, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Bilateral smooth & additive blend -> target_view
        let composite_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssrc_composite_bind_group"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.irradiance_view),
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

