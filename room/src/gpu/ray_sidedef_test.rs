use super::*;

#[test]
fn rays_select_the_visible_sidedef_in_both_draw_orders() {
    for opposite in [false, true] {
        for reverse_order in [false, true] {
            run_case(opposite, reverse_order);
        }
    }
}

fn run_case(opposite: bool, reverse_order: bool) {
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

        // Coincident sidedefs have different albedo and opposite winding.
        let points = [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];
        let mut verts: Vec<crate::hw::Vertex> = [false, true]
            .into_iter()
            .flat_map(|back| {
                let order = if back { [0, 1, 2, 3] } else { [3, 2, 1, 0] };
                order.map(|i| crate::hw::Vertex {
                    pos: points[i],
                    uv: [0.5; 2],
                    atlas: [0.0, 0.0, 1.0, 1.0],
                    light: 1.0,
                    tint: if back { [1.0, 0.0, 0.0] } else { [0.5; 3] },
                    _pad: -1.0,
                    light_uv: [if back { 0.75 } else { 0.25 }, 0.5],
                })
            })
            .collect();
        crate::hw::reconcile_wall_sides(&mut verts, 0);
        let indices: [u32; 12] = if reverse_order {
            [4, 5, 6, 4, 6, 7, 0, 1, 2, 0, 2, 3]
        } else {
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        };
        let (vertex_usage, index_usage) = mesh_usage(true);
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (verts.len() * std::mem::size_of::<crate::hw::Vertex>()) as u64,
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

        let blas = make_blas(&device, 8, 12);
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
        let light = world_light_read(&device, 2);
        let light_write = world_light_write(&device, 2);
        queue.write_texture(
            light.as_image_copy(),
            bytemuck::cast_slice(&[
                0x3c00u16, 0x3c00, 0x3c00, 0x3c00, 0x3800, 0x3800, 0x3800, 0x3c00, 0x3c00, 0x3c00,
                0x3c00, 0x3c00, 0x3800, 0x3800, 0x3800, 0x3c00,
            ]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(2),
            },
            light.size(),
        );
        let hits = light_hit_buffer(&device, 1);
        let _bins = create_light_bins(&device, &light, &light_write, &hits).expect("light bins");
        let pbr = test_dummy_pbr(&device, &queue);
        let bind = ray_bind_group(
            &device,
            &layout,
            &uniform,
            &atlas,
            &emission,
            &pbr,
            &light,
            &light_write,
            &vertices,
            &index_buf,
            &hits,
            &tlas,
        );
        let cam = RayUniform {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            origin_aspect: [0.0, 0.0, if opposite { 2.0 } else { -2.0 }, 1.0],
            right: [1.0, 0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0, 0.0],
            forward: [0.0, 0.0, if opposite { -1.0 } else { 1.0 }, 0.0],
            viewport: [0.0, 0.0, 1.0, 1.0],
            light: [4.0, 1.0, 6.0, 0.0],
            fill: [0.0, 1.0, 0.0, 0.0],
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
        let size = tri_size(8, 12);
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
        assert_eq!(
            px,
            if opposite {
                [128, 0, 0, 255]
            } else {
                [128, 128, 128, 255]
            },
            "wrong ray sidedef: opposite={opposite}, reverse_order={reverse_order}"
        );
    });
}
