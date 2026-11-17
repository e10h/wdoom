//! Dedicated fixtures must light surfaces without fixture geometry.
use super::*;

#[test]
fn dedicated_source_lights_a_receiver_and_respects_occlusion() {
    let mut area_reference = None;
    for (blocked, count, area, padding) in [
        (false, 1u32, false, 0),
        (false, 1u32, false, 127),
        (false, 128u32, false, 0),
        (true, 1u32, false, 0),
        (false, 1u32, true, 0),
        (false, 128u32, true, 0),
        (true, 1u32, true, 0),
    ] {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
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

            // The camera sees an unlit receiver at z=0. A source at (4,0,-4)
            // lights its front. A blocker at x=2 is outside the camera ray.
            let bx = if blocked { 2.0 } else { 20000.0 };
            let mut verts = Vec::<f32>::new();
            for pos in [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [bx, -100.0, -100.0],
                [bx, 100.0, -100.0],
                [bx, 0.0, 100.0],
            ] {
                verts.extend_from_slice(&[
                    pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                    -1.0, -1.0,
                ]);
            }
            let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
            let (vertex_usage, index_usage) = mesh_usage(true);
            let vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (verts.len() * 4) as u64,
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
                    width: 2,
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
                &[255, 255, 255, 255, 255, 0, 0, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 2,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let emission = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 2,
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
            emit_row[1] = 255;
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
                    width: 2,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            let blas = make_blas(&device, 6, 6);
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
            let _bins =
                create_light_bins(&device, &light, &light_write, &hits).expect("light bins");
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
                origin_aspect: [0.0, 0.0, -2.0, 1.0],
                right: [1.0, 0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0, 0.0],
                forward: [0.0, 0.0, 1.0, 0.0],
                viewport: [0.0, 0.0, 1.0, 1.0],
                light: [40.0, 0.0, 2.0, 0.0],
                fill: [0.0, 0.0, 0.0, 0.0],
                fixtures: {
                    let mut sources = crate::fixture_lights::FixtureLights::default();
                    sources.meta[0] = count + padding;
                    sources.meta[2] = if area { 0 } else { count };
                    let source = if area {
                        crate::fixture_lights::FixtureLight {
                            position_radius: [3.0, 4.0, -5.0, -1.0],
                            color_power: [5.0, 4.0, -5.0, 2.0 / count as f32],
                            triangle_c: [4.0, 4.0, -3.0, 0.0],
                            atlas: [0.75, 0.5, 0.75, 0.5],
                            ..Default::default()
                        }
                    } else {
                        crate::fixture_lights::FixtureLight {
                            position_radius: [4.0, 0.0, -4.0, 0.0],
                            color_power: [1.0, 0.0, 0.0, 16.0 / count as f32],
                            ..Default::default()
                        }
                    };
                    sources.lights.fill(source);
                    for light in &mut sources.lights[count as usize..(count + padding) as usize] {
                        *light = crate::fixture_lights::FixtureLight {
                            position_radius: [0.0, 0.0, 0.0, -1.0],
                            ..Default::default()
                        };
                    }
                    sources
                },
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
            let size = tri_size(6, 6);
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &vertices,
                        first_vertex: 0,
                        vertex_stride: VERTEX_STRIDE,
                        index_buffer: Some(&index_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
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
            if blocked {
                assert!(
                    px[0] < 8 && px[1] < 8 && px[2] < 8,
                    "light leaked through blocker: {px:?}"
                );
            } else if area {
                assert!(
                    px[0] > 15 && px[0] < 150 && px[1] < 8 && px[2] < 8,
                    "panel failed to emit: {px:?}"
                );
                if let Some(ref reference) = area_reference {
                    assert_eq!(
                        &px, reference,
                        "area sampling must conserve energy across source counts"
                    );
                } else {
                    area_reference = Some(px);
                }
            } else {
                assert!(
                    (85..=95).contains(&px[0]) && px[1] < 8 && px[2] < 8,
                    "source failed to light receiver: {px:?}"
                );
            }
        });
    }
}
