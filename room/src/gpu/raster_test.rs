use super::*;
use wgpu::util::DeviceExt;

// Opposite Doom sidedefs occupy the same plane, with independent materials
// and light UVs. Depth testing cannot choose which sidedef faces the player.
#[test]
fn opposite_wall_materials_stay_on_their_own_side() {
    for samples in [1, 4] {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("GPU device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production world shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opposite sidedef regression"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_world"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_world"),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            multiview_mask: None,
            cache: None,
        });
        let tex = device.create_texture(&wgpu::TextureDescriptor {
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
            tex.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            tex.size(),
        );
        let tex_view = tex.create_view(&Default::default());
        let light = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("baked or untouched light"),
            size: tex.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let light_view = light.create_view(&Default::default());
        let emission = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("emissive surface mask"),
            size: tex.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let emission_view = emission.create_view(&Default::default());
        let pbr = test_dummy_pbr(&device, &queue);
        let pbr_view = pbr.create_view(&Default::default());
        let sampler = device.create_sampler(&Default::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<WorldUniform>() as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
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
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
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
        let view = output.create_view(&Default::default());
        let (_depth, depth) = depth_target_samples(&device, 16, 16, samples);
        let (_msaa_color, msaa_view) = msaa_color_target(&device, 16, 16, wgpu::TextureFormat::Rgba8Unorm);
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        // A baked white texel, an untouched texel, and a baked black texel.
        for glow in [0u8, 255] {
            queue.write_texture(
                emission.as_image_copy(),
                &[glow],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                emission.size(),
            );
            for sample in [
                [0x3c00u16, 0x3c00, 0x3c00, 0x3c00],
                [0, 0, 0, 0],
                [0, 0, 0, 0x3c00],
                [0x4c00, 0x4c00, 0x4c00, 0x3c00], // HDR 16 must display like capped 1
            ] {
                queue.write_texture(
                    light.as_image_copy(),
                    bytemuck::cast_slice(&sample),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(8),
                        rows_per_image: Some(1),
                    },
                    light.size(),
                );
                for distance in [32.0, 512.0, 4096.0] {
                    for reverse_order in [false, true] {
                        for opposite_view in [false, true] {
                            let near = 1.0;
                            let far = 16000.0;
                            let nf = 1.0 / (near - far);
                            let sign = if opposite_view { -1.0 } else { 1.0 };
                            let camera = WorldUniform {
                                view_proj: [
                                    [sign, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, sign * far * nf, -sign],
                                    [
                                        0.0,
                                        0.0,
                                        far * near * nf
                                            + if opposite_view {
                                                -2.0 * distance * far * nf
                                            } else {
                                                0.0
                                            },
                                        if opposite_view { 2.0 * distance } else { 0.0 },
                                    ],
                                ],
                                fill: [0.0, 40.0, 1.0, 0.0],
                                eye: [0.0; 4],
                                pbr: [1.0, 1.0, 1.0, 0.0],
                                extra: [1.0, 0.0, 0.0, 0.0],
                                probe_meta: [0; 4],
                                probes: [crate::hw::LightProbe::default(); crate::hw::MAX_LIGHT_PROBES],
                            };
                            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&camera));
                            let mut vertices = Vec::new();
                            for back in if reverse_order {
                                [false, true]
                            } else {
                                [true, false]
                            } {
                                let points = [
                                    [-distance, -distance, -distance],
                                    [-distance, distance, -distance],
                                    [distance, distance, -distance],
                                    [distance, -distance, -distance],
                                ];
                                for i in if back {
                                    [0, 2, 1, 0, 3, 2]
                                } else {
                                    [0, 1, 2, 0, 2, 3]
                                } {
                                    vertices.push(crate::hw::Vertex {
                                        pos: points[i],
                                        uv: [0.5; 2],
                                        atlas: [0.0, 0.0, 1.0, 1.0],
                                        light: 1.0,
                                        tint: if back { [1.0, 0.0, 0.0] } else { [0.5; 3] },
                                        _pad: -1.0,
                                        light_uv: [0.5; 2],
                                    });
                                }
                            }
                            let buffer =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: None,
                                    contents: bytemuck::cast_slice(&vertices),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });
                            let mut encoder = device.create_command_encoder(&Default::default());
                            {
                                let mut pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: if samples == 4 { &msaa_view } else { &view },
                                                depth_slice: None,
                                                resolve_target: if samples == 4 { Some(&view) } else { None },
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: Some(
                                            wgpu::RenderPassDepthStencilAttachment {
                                                view: &depth,
                                                depth_ops: Some(wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(1.0),
                                                    store: wgpu::StoreOp::Store,
                                                }),
                                                stencil_ops: None,
                                            },
                                        ),
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                        multiview_mask: None,
                                    });
                                pass.set_pipeline(&pipeline);
                                pass.set_bind_group(0, &bind, &[]);
                                pass.set_vertex_buffer(0, buffer.slice(..));
                                pass.draw(0..12, 0..1);
                            }
                            encoder.copy_texture_to_buffer(
                                wgpu::TexelCopyTextureInfo {
                                    origin: wgpu::Origin3d { x: 8, y: 8, z: 0 },
                                    ..output.as_image_copy()
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
                            queue.submit([encoder.finish()]);
                            let (tx, rx) = std::sync::mpsc::channel();
                            read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                                tx.send(r).unwrap();
                            });
                            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                            rx.recv().unwrap().unwrap();
                            let px = read.slice(..).get_mapped_range()[..4].to_vec();
                            read.unmap();
                            let expected = if glow != 0 {
                                if opposite_view {
                                    [255, 0, 0, 255]
                                } else {
                                    [128, 128, 128, 255]
                                }
                            } else if sample == [0, 0, 0, 0x3c00] {
                                [0, 0, 0, 255]
                            } else if opposite_view {
                                [255, 0, 0, 255]
                            } else {
                                [128, 128, 128, 255]
                            };
                            assert_eq!(px, expected, "wrong sidedef at distance {distance}, reverse_order={reverse_order}, opposite_view={opposite_view}");
                        }
                    }
                }
            }
        }
    });
    }
}

#[test]
fn test_weapon_render_scanline_121() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../doom1.wad");
    if !path.exists() {
        eprintln!("doom1.wad not found at {:?}", path);
        return;
    }
    let args: Vec<_> = [
        "room",
        "-iwad",
        path.to_str().unwrap(),
        "-warp",
        "1",
        "1",
        "-nosound",
        "-nomusic",
    ]
    .into_iter()
    .map(|s| std::ffi::CString::new(s).unwrap())
    .collect();
    let mut argv: Vec<_> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
    argv.push(std::ptr::null_mut());
    crate::platform::init_start_time();
    unsafe {
        crate::doom::d_loop::singletics = 1;
        crate::doom::doomgeneric::doomgeneric_Create(args.len() as i32, argv.as_mut_ptr());
        for _ in 0..35 {
            crate::doom::d_main::doomgeneric_Tick();
        }
    }

    let scene = unsafe { crate::hw::build_scene(true) };
    println!("scene verts len: {}", scene.verts.len());
    println!("scene indices len: {}", scene.indices.len());
    println!(
        "world_indices: {}, opaque_indices: {}",
        scene.world_indices, scene.opaque_indices
    );
    let total = scene.indices.len() as u32;
    let world_n = scene.world_indices;
    println!("weapon indices count: {}", total - world_n);
    for i in world_n..total {
        let vi = scene.indices[i as usize] as usize;
        let v = &scene.verts[vi];
        println!(
            "idx {}: vert {}: pos={:?}, uv={:?}, atlas={:?}, surface={}",
            i, vi, v.pos, v.uv, v.atlas, v._pad
        );
    }
    println!("view_proj: {:?}", scene.view_proj);
    println!("view_rect: {:?}", scene.view_rect);

    let weapon_atlas = scene.verts[scene.indices[world_n as usize] as usize].atlas;
    let mut other_weapon_verts = Vec::new();
    let num_world_verts = scene.verts.len().saturating_sub(4);
    for (i, v) in scene.verts[..num_world_verts].iter().enumerate() {
        if v.atlas == weapon_atlas {
            other_weapon_verts.push(i);
        }
    }
    println!("other verts with weapon atlas in 0..world_verts: {:?}", other_weapon_verts);

    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("GPU device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production world shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test pipeline layout"),
            bind_group_layouts: &[Some(&device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            }))],
            immediate_size: 0,
        });

        let weapon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("doom_weapon_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_world"),
                buffers: &[vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_weapon"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        let (atlas_size, atlas_rgba, atlas_emit, atlas_pbr) = scene.atlas.expect("scene atlas");
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
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
            atlas_tex.as_image_copy(),
            &atlas_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_size * 4),
                rows_per_image: Some(atlas_size),
            },
            atlas_tex.size(),
        );

        let emit_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("emit"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
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
            emit_tex.as_image_copy(),
            &atlas_emit,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_size),
                rows_per_image: Some(atlas_size),
            },
            emit_tex.size(),
        );

        let pbr_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
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
            pbr_tex.as_image_copy(),
            &atlas_pbr,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_size * 4),
                rows_per_image: Some(atlas_size),
            },
            pbr_tex.size(),
        );

        let light_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("doom_atlas_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let cam = WorldUniform {
            view_proj: scene.view_proj,
            eye: [scene.eye[0], scene.eye[1], scene.eye[2], 1.0],
            fill: [0.23, 1.0, 2.0, 0.0],
            pbr: [1.0, 1.0, 1.0, 0.0],
            extra: [1.0, 0.0, 0.0, 0.0],
            probe_meta: [0; 4],
            probes: [crate::hw::LightProbe::default(); crate::hw::MAX_LIGHT_PROBES],
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<WorldUniform>() as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&cam));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &weapon_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_tex.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&light_tex.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&emit_tex.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&pbr_tex.create_view(&Default::default())),
                },
            ],
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("play_target"),
            size: wgpu::Extent3d {
                width: 320,
                height: 168,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&scene.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, 320.0, 168.0, 0.0, 1.0);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_pipeline(&weapon_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw_indexed(world_n..total, 0, 0..1);
        }

        // Readback
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 320 * 4 * 168,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(320 * 4),
                    rows_per_image: Some(168),
                },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range();

        for y in 118..126 {
            let row = &data[y * 320 * 4..(y + 1) * 320 * 4];
            let non_zero: Vec<(usize, [u8; 4])> = row
                .chunks_exact(4)
                .enumerate()
                .filter(|(_, p)| p[3] > 0)
                .map(|(x, p)| (x, [p[0], p[1], p[2], p[3]]))
                .collect();
            if y <= 121 {
                assert_eq!(
                    non_zero.len(),
                    0,
                    "Scanline {y} above weapon must not contain any ghost pixels (got {})",
                    non_zero.len()
                );
            } else if y == 122 {
                assert!(
                    !non_zero.is_empty(),
                    "Scanline 122 should contain the pistol sight tip"
                );
            }
        }
    });
}

