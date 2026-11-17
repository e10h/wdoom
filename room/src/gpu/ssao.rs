//! Screen-Space Ambient Occlusion (SSAO) for natural contact shadows and corner darkening.
//!
//! Evaluates normal-oriented hemisphere occlusion from the linear depth buffer,
//! followed by a depth-aware bilateral filter that multiplicatively darkens
//! corners, floor/ceiling creases, and object crevices without noise or halos.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    viewport: [f32; 4], // x: width, y: height, z: 1/width, w: 1/height
    params: [f32; 4],   // x: strength, y: radius_world, z: bias, w: aspect
}

const SSAO_EVAL_SHADER: &str = r#"
struct SsaoUniforms {
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

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var s_sampler: sampler;
@group(0) @binding(2) var<uniform> u: SsaoUniforms;

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

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let dims = vec2<i32>(textureDimensions(t_depth));
    let px = vec2<i32>(in.clip_position.xy);
    let d0 = fetch_depth(px, dims);
    if (d0 >= 0.9999) {
        return 1.0;
    }
    let z0 = linear_depth(d0);
    let p0 = reconstruct_pos(px, dims, z0);
    let n0 = reconstruct_normal(px, dims, p0, z0);

    let radius_world = u.params.y;
    // Projected screen radius in pixels
    let screen_rad = clamp((radius_world / z0) * f32(dims.y) * 0.70, 2.0, 48.0);
    let rot = hash12(in.clip_position.xy) * 6.2831853;

    var occ = 0.0;
    let num_samples = 12;
    for (var i = 0; i < num_samples; i = i + 1) {
        let alpha = (f32(i) + 0.5) / f32(num_samples);
        let angle = f32(i) * 2.3999632 + rot;
        let r = sqrt(alpha) * screen_rad;
        let offset_px = vec2<f32>(cos(angle), sin(angle)) * r;
        let sample_coord = clamp(px + vec2<i32>(offset_px), vec2<i32>(0), dims - vec2<i32>(1));

        let ds = fetch_depth(sample_coord, dims);
        if (ds < 0.9999) {
            let zs = linear_depth(ds);
            let ps = reconstruct_pos(sample_coord, dims, zs);
            let v = ps - p0;
            let dist = length(v);
            let ndotv = dot(n0, v / max(dist, 0.001));
            // Sample is in front of the tangent plane (occluding adjacent surface/crease)
            if (ndotv > 0.08 && dist < radius_world * 1.6) {
                let range_fade = clamp(1.0 - dist / (radius_world * 1.6), 0.0, 1.0);
                occ = occ + ndotv * range_fade;
            }
        }
    }
    let raw_occ = (occ / f32(num_samples)) * u.params.x * 2.5;
    let ao = clamp(1.0 - raw_occ, 0.20, 1.0);
    return ao;
}
"#;

const SSAO_BLUR_SHADER: &str = r#"
struct SsaoUniforms {
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

@group(0) @binding(0) var t_ssao: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_sampler: sampler;
@group(0) @binding(3) var<uniform> u: SsaoUniforms;

fn linear_depth(d: f32) -> f32 {
    return 1.0 / max(1.0 - d, 0.0001);
}

fn fetch_depth(coord: vec2<i32>, dims: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(t_depth, clamped, 0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t_depth));
    let px = vec2<i32>(in.clip_position.xy);
    let d0 = fetch_depth(px, dims);
    if (d0 >= 0.9999) {
        return vec4<f32>(1.0);
    }
    let z0 = linear_depth(d0);

    // Cross-bilateral filter preserving geometric depth edges
    var total_ao = 0.0;
    var total_weight = 0.0;

    let offsets = array<vec2<i32>, 9>(
        vec2<i32>( 0,  0),
        vec2<i32>(-1,  0),
        vec2<i32>( 1,  0),
        vec2<i32>( 0, -1),
        vec2<i32>( 0,  1),
        vec2<i32>(-1, -1),
        vec2<i32>( 1, -1),
        vec2<i32>(-1,  1),
        vec2<i32>( 1,  1),
    );

    let depth_threshold = max(z0 * 0.08, 8.0);

    for (var i = 0; i < 9; i = i + 1) {
        let sample_coord = clamp(px + offsets[i], vec2<i32>(0), dims - vec2<i32>(1));
        let ds = fetch_depth(sample_coord, dims);
        let zs = linear_depth(ds);
        let depth_diff = abs(zs - z0);

        let spatial_weight = select(0.707, 1.0, offsets[i].x == 0 || offsets[i].y == 0);
        let depth_weight = clamp(1.0 - depth_diff / depth_threshold, 0.001, 1.0);
        let w = spatial_weight * depth_weight;

        let ao_sample = textureLoad(t_ssao, sample_coord, 0).r;
        total_ao = total_ao + ao_sample * w;
        total_weight = total_weight + w;
    }

    let filtered_ao = total_ao / max(total_weight, 0.0001);
    return vec4<f32>(vec3<f32>(filtered_ao), 1.0);
}
"#;

pub struct SsaoPass {
    width: u32,
    height: u32,
    ssao_texture: wgpu::Texture,
    ssao_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    eval_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    eval_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl SsaoPass {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_eval_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_EVAL_SHADER.into()),
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_BLUR_SHADER.into()),
        });

        let eval_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_eval_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_layout"),
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

        let eval_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_eval_pipeline_layout"),
            bind_group_layouts: &[Some(&eval_layout)],
            immediate_size: 0,
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_blur_pipeline_layout"),
            bind_group_layouts: &[Some(&blur_layout)],
            immediate_size: 0,
        });

        let eval_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_eval_pipeline"),
            layout: Some(&eval_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &eval_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &eval_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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

        // Multiplicative blend into the target color buffer: result = Dst * Src
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_blur_pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Dst,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (ssao_texture, ssao_view) = Self::create_ssao_target(device, width.max(1), height.max(1));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            width: width.max(1),
            height: height.max(1),
            ssao_texture,
            ssao_view,
            uniform_buf,
            eval_pipeline,
            blur_pipeline,
            eval_layout,
            blur_layout,
            sampler,
        }
    }

    fn create_ssao_target(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_raw_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.width == w && self.height == h {
            return;
        }
        self.width = w;
        self.height = h;
        let (tex, view) = Self::create_ssao_target(device, w, h);
        self.ssao_texture = tex;
        self.ssao_view = view;
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        strength: f32,
        radius: f32,
    ) {
        if strength <= 0.01 {
            return;
        }
        self.resize(device, width, height);

        let aspect = width as f32 / height.max(1) as f32;
        let uniforms = SsaoUniforms {
            viewport: [width as f32, height as f32, 1.0 / width.max(1) as f32, 1.0 / height.max(1) as f32],
            params: [strength, radius, 0.05, aspect],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Pass 1: Evaluate raw SSAO
        let eval_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_eval_bind"),
            layout: &self.eval_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_eval_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.eval_pipeline);
            pass.set_bind_group(0, &eval_bind, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 2: Bilateral filter & multiplicative blend into target_view (Dst * ao)
        let blur_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_bind"),
            layout: &self.blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ssao_view),
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
                label: Some("ssao_blur_pass"),
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
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bind, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssao_pass_renders_without_validation_errors() {
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

            let mut ssao = SsaoPass::new(&device, format, width, height);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test_ssao_encoder") });

            ssao.render(
                &mut encoder,
                &device,
                &queue,
                &depth_view,
                &color_view,
                width,
                height,
                1.0,
                24.0,
            );

            queue.submit(Some(encoder.finish()));
        });
    }
}
