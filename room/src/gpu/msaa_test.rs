//! A resolved diagonal must gain coverage shades without blurring its interior.
use super::*;

#[test]
fn four_samples_smooth_geometry_edges() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coverage regression"),
            source: wgpu::ShaderSource::Wgsl(r#"
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 3>(vec2<f32>(-0.9,-0.9), vec2<f32>(0.9,-0.9), vec2<f32>(-0.9,0.9));
                    return vec4<f32>(p[i], 0.5, 1.0);
                }
                @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
            "#.into()),
        });
        for samples in [1, 4] {
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("coverage regression"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });
            let resolved = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("resolved coverage"),
                size: wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = resolved.create_view(&Default::default());
            let (_color, aa_view) = msaa_color_target(&device, 16, 16, resolved.format());
            let (_depth, depth_view) = depth_target_samples(&device, 16, 16, samples);
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: if samples == 4 { &aa_view } else { &view },
                        resolve_target: if samples == 4 { Some(&view) } else { None },
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
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_buffer(
                resolved.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(16),
                    },
                },
                resolved.size(),
            );
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = read.slice(..).get_mapped_range();
            let pixels: Vec<u8> = (0..16)
                .flat_map(|y| (0..16).map(move |x| y * 256 + x * 4))
                .map(|i| bytes[i])
                .collect();
            assert!(
                pixels.contains(&255),
                "triangle interior must stay sharp/white"
            );
            assert!(pixels.contains(&0), "background must stay black");
            assert_eq!(
                pixels.iter().any(|v| *v > 0 && *v < 255),
                samples == 4,
                "MSAA must create fractional edge coverage"
            );
        }
    });
}
