//! Exercise the production scene -> SMAA -> HUD ordering on the GPU.
use super::*;

#[test]
fn smaa_smooths_edges_and_keeps_hud_crisp_after_resize() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.expect("GPU");
        eprintln!("SMAA adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter.request_device(&Default::default()).await.expect("device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(r#"
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 3>(vec2<f32>(-0.9,-0.9), vec2<f32>(0.9,-0.7), vec2<f32>(-0.7,0.9));
                    return vec4<f32>(p[i], 0.0, 1.0);
                }
                @vertex fn vs_hud(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
                    return vec4<f32>(p[i], 0.0, 1.0);
                }
                @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
            "#.into()),
        });
        for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8UnormSrgb] {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: None,
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(format.into())], compilation_options: Default::default() }),
                primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
            });
            let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: None,
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_hud"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(format.into())], compilation_options: Default::default() }),
                primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
            });
            let mut target = smaa::SmaaTarget::new(&device, &queue, 64, 64, format, smaa::SmaaMode::Smaa1X);
            for size in [64u32, 80] {
                if size != 64 { target.resize(&device, size, size); }
                let output = device.create_texture(&wgpu::TextureDescriptor {
                    label: None, size: light_extent(size), mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2, format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
                });
                let view = output.create_view(&Default::default());
                let frame = target.start_frame(&device, &queue, &view);
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &frame, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })], ..Default::default()
                    });
                    pass.set_pipeline(&pipeline); pass.draw(0..3, 0..1);
                }
                queue.submit([encoder.finish()]);
                frame.resolve();
                // A small HUD patch drawn afterwards must retain exact pixel edges.
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })], ..Default::default()
                    });
                    pass.set_pipeline(&hud_pipeline);
                    pass.set_scissor_rect(0, 0, 4, 4);
                    pass.draw(0..3, 0..1);
                }
                let stride = (size * 4).div_ceil(256) * 256;
                let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(stride*size), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
                encoder.copy_texture_to_buffer(output.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(size) } }, output.size());
                queue.submit([encoder.finish()]);
                let (tx,rx) = std::sync::mpsc::channel();
                read.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap(); rx.recv().unwrap().unwrap();
                let data = read.slice(..).get_mapped_range();
                for y in 0..6usize {
                    for x in 0..6usize {
                        assert_eq!(data[y*stride as usize+x*4], if x < 4 && y < 4 { 255 } else { 0 }, "HUD edge must stay crisp");
                    }
                }
                let values: Vec<_> = (0..size).flat_map(|y| (0..size).map(move |x| (y*stride+x*4) as usize)).map(|i| data[i]).collect();
                assert!(values.iter().any(|&v| v > 0 && v < 255), "SMAA must produce edge coverage");
                assert!(values.contains(&0) && values.contains(&255), "interior/background must remain sharp");
            }
            assert!(scope.pop().await.is_none(), "SMAA validation failed");
        }
    });
}

#[test]
fn post_pixelization_smaa_smooths_chunky_upscaled_blocks() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.expect("GPU");
        let (device, queue) = adapter.request_device(&Default::default()).await.expect("device");
        let format = wgpu::TextureFormat::Rgba8Unorm;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 6>(
                        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
                    );
                    return vec4<f32>(p[i], 0.0, 1.0);
                }
                @group(0) @binding(0) var t: texture_2d<f32>;
                @group(0) @binding(1) var s: sampler;
                @fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                    let uv = p.xy / vec2<f32>(64.0, 64.0);
                    return textureSample(t, s, uv);
                }
            "#.into()),
        });

        // Create a 16x16 chunky image with a high-contrast diagonal edge
        let mut chunky_pixels = vec![0u8; 16 * 16 * 4];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 4;
                let val = if x >= y { 255u8 } else { 0u8 };
                chunky_pixels[idx..idx + 4].copy_from_slice(&[val, val, val, 255]);
            }
        }
        let low_res_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("low_res_tex"),
            size: wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            low_res_tex.as_image_copy(),
            &chunky_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16 * 4),
                rows_per_image: Some(16),
            },
            wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
        );
        let low_res_view = low_res_tex.create_view(&Default::default());
        let nearest_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&low_res_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&nearest_samp) },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(format.into())], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut target = smaa::SmaaTarget::new(&device, &queue, 64, 64, format, smaa::SmaaMode::Smaa1X);
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = output.create_view(&Default::default());
        let frame = target.start_frame(&device, &queue, &view);

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upscale_blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        queue.submit([encoder.finish()]);
        frame.resolve();

        let stride = (64u32 * 4).div_ceil(256) * 256;
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(stride * 64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(64) },
            },
            output.size(),
        );
        queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let view = read.slice(..).get_mapped_range();
        let data = &*view;
        let values: Vec<_> = (0..64)
            .flat_map(|y| (0..64).map(move |x| data[(y * stride as usize + x * 4)]))
            .collect();

        // Edge pixels on the chunky upscaled diagonal must have blended/intermediate values from SMAA
        assert!(values.iter().any(|&v| v > 0 && v < 255), "SMAA must smooth chunky blocky edges");
        assert!(values.contains(&0) && values.contains(&255), "Interior block areas must remain pure 0 and 255");
    });
}
