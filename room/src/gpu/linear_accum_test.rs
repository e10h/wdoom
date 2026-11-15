use super::*;

fn half_value(bits: u16) -> f32 {
    let exponent = ((bits >> 10) & 31) as i32;
    let fraction = (bits & 1023) as f32;
    if exponent == 0 { fraction * 2.0f32.powi(-24) }
    else { (1.0 + fraction / 1024.0) * 2.0f32.powi(exponent - 15) }
}

#[test]
fn hdr_accumulation_and_faint_world_energy() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await else {
            eprintln!("no adapter; skipping HDR test"); return;
        };
        let ray = adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: if ray { wgpu::Features::EXPERIMENTAL_RAY_QUERY } else { wgpu::Features::empty() },
            required_limits: wgpu::Limits::default().using_acceleration_structure_values(adapter.limits()),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }).await.expect("device");
        if ray {
            let (layout, _, _) = create_ray_pipeline_mode(&device, wgpu::TextureFormat::Rgba16Float, true).expect("HDR ray pipeline");
            create_ray_present_pipeline(&device, &layout, wgpu::TextureFormat::Rgba8Unorm).expect("display pipeline");
        }
        let sample = |name| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name), size: light_extent(1), mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let current = sample("current");
        let history = sample("history");
        let combined = sample("combined");
        let acc = create_accum(&device, wgpu::TextureFormat::Rgba16Float);
        let current_view = current.create_view(&Default::default());
        let history_view = history.create_view(&Default::default());
        let combined_view = combined.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &acc.layout, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&current_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&history_view) },
                wgpu::BindGroupEntry { binding: 2, resource: acc.uniform.as_entire_binding() },
            ],
        });
        let world = world_light_read(&device, 1);
        let write = world_light_write(&device, 1);
        let hits = light_hit_buffer(&device, 1024);
        let bins = create_light_bins_sized(&device, &world, &write, &hits, 1).expect("bins");
        // Sparse moderate, sparse HDR, and faint energy below the old 1/64 cutoff.
        for (blue, half, expected) in [(1.25f32, 0x3d00u16, 0.3125), (8.0, 0x4800, 2.0), (1.0/256.0, 0x1c00, 1.0/1024.0)] {
            clear_light(&queue, &history, 1);
            clear_light(&queue, &world, 1);
            for frame in 0..4 {
                let value = if frame == 1 { half } else { 0 };
                queue.write_texture(current.as_image_copy(), bytemuck::cast_slice(&[0u16, 0, value, 0x3c00]),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) }, light_extent(1));
                queue.write_buffer(&acc.uniform, 0, bytemuck::cast_slice(&[1.0f32/(frame+1) as f32, 0.0, 0.0, 0.0]));
                // Independent samples make stochastic rounding measurable in one dispatch.
                let mut records = Vec::new();
                for _ in 0..1024 { records.extend_from_slice(&[0u32, 0, if frame == 1 { blue.to_bits() } else { 0 }, 1]); }
                queue.write_buffer(&hits, 0, bytemuck::cast_slice(&records));
                queue.write_buffer(&bins.uniform, 0, bytemuck::bytes_of(&BinParams { pixels: 1024, light_size: 1, slots: 1, voxel_rows: frame << 16 }));
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.clear_buffer(&bins.bins, 0, None);
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_bind_group(0, &bins.bind, &[]);
                    pass.set_pipeline(&bins.bin_pipeline); pass.dispatch_workgroups(16, 1, 1);
                    pass.set_pipeline(&bins.save_pipeline); pass.dispatch_workgroups(1, 1, 1);
                }
                copy_light(&mut encoder, &write, &world, 1);
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &combined_view, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })], ..Default::default()
                    });
                    pass.set_pipeline(&acc.pipeline); pass.set_bind_group(0, &bind, &[]); pass.draw(0..6, 0..1);
                }
                copy_light(&mut encoder, &combined, &history, 1);
                queue.submit([encoder.finish()]);
            }
            let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 512,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut encoder = device.create_command_encoder(&Default::default());
            for (offset, tex) in [(0, &combined), (256, &world)] {
                encoder.copy_texture_to_buffer(tex.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(256), rows_per_image: Some(1) } }, light_extent(1));
            }
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            read.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap(); rx.recv().unwrap().unwrap();
            let data = read.slice(..).get_mapped_range();
            let screen = half_value(u16::from_le_bytes([data[4], data[5]]));
            let world = half_value(u16::from_le_bytes([data[260], data[261]]));
            assert!((screen - expected).abs() < 0.001, "screen {screen} expected {expected}");
            assert!((world - expected).abs() < expected * 0.2 + 0.00001, "world {world} expected {expected}");
            assert!(world > 0.0, "faint energy was discarded");
        }
    });
}
