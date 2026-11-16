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
