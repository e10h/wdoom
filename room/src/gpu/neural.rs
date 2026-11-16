//! Optional native NR input target and shared-device inference.
use neural_nr::{
    frame::{Frame, InputBindings},
    gpu::Gpu,
    model::Model,
};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const DOWNSAMPLE_WGSL: &str = r#"
struct DownsampleParams {
    viewport: vec4<f32>,
    source_size: vec2<f32>,
    is_doom_res: u32,
    pad: f32,
}
@group(0) @binding(0) var<uniform> dp: DownsampleParams;
@group(0) @binding(1) var s_sampler: sampler;
@group(0) @binding(2) var t_source: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i == 1u) * 4.0 - 1.0;
    let y = f32(i == 2u) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn srgb_decode(value: f32) -> f32 {
    let bounded = clamp(value, 0.0, 1.0);
    return select(pow((bounded + 0.055) / 1.055, 2.4), bounded / 12.92, bounded <= 0.04045);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    if (dp.is_doom_res != 0u) {
        let pixel_coord = clamp(vec2<i32>(floor(in.uv * dp.viewport.zw)), vec2<i32>(0), vec2<i32>(dp.source_size) - 1);
        let col = textureLoad(t_source, pixel_coord, 0);
        let lin = vec3<f32>(srgb_decode(col.r), srgb_decode(col.g), srgb_decode(col.b));
        return vec4<f32>(lin, 1.0);
    }
    let uv_in_source = (dp.viewport.xy + in.uv * dp.viewport.zw) / dp.source_size;
    let col = textureSample(t_source, s_sampler, uv_in_source);
    let lin = vec3<f32>(srgb_decode(col.r), srgb_decode(col.g), srgb_decode(col.b));
    return vec4<f32>(lin, 1.0);
}
"#;

const DISPLAY_SHADER_WGSL: &str = r#"
struct DisplayParams {
    split_x: f32,
    mode: u32,
    screen_size: vec2<f32>,
    viewport: vec4<f32>,
    dest_size: vec2<f32>,
    fsr_enabled: u32,
    sharpness: f32,
    intensity: f32,
    has_neural: u32,
    is_doom_res: u32,
    pad: f32,
}
@group(0) @binding(0) var<uniform> dp: DisplayParams;
@group(0) @binding(1) var s_sampler: sampler;
@group(0) @binding(2) var t_native: texture_2d<f32>;
@group(0) @binding(3) var t_neural: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i == 1u) * 4.0 - 1.0;
    let y = f32(i == 2u) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn fsr_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn fsr_sample_neural(uv: vec2<f32>, sharpness: f32) -> vec3<f32> {
    let dims = textureDimensions(t_neural);
    let tex_size = vec2<f32>(f32(dims.x), f32(dims.y));
    let inv_tex_size = 1.0 / tex_size;

    let pos = uv * tex_size - vec2<f32>(0.5, 0.5);
    let base = floor(pos);
    let f = pos - base;

    let p_00 = (base + vec2<f32>(0.5, 0.5)) * inv_tex_size;
    let p_10 = p_00 + vec2<f32>(inv_tex_size.x, 0.0);
    let p_01 = p_00 + vec2<f32>(0.0, inv_tex_size.y);
    let p_11 = p_00 + inv_tex_size;

    let c_b = textureSampleLevel(t_neural, s_sampler, p_00, 0.0).rgb;
    let c_c = textureSampleLevel(t_neural, s_sampler, p_10, 0.0).rgb;
    let c_i = textureSampleLevel(t_neural, s_sampler, p_01, 0.0).rgb;
    let c_j = textureSampleLevel(t_neural, s_sampler, p_11, 0.0).rgb;

    let l_b = fsr_luma(c_b);
    let l_c = fsr_luma(c_c);
    let l_i = fsr_luma(c_i);
    let l_j = fsr_luma(c_j);

    let dir_x = (l_c - l_b) + (l_j - l_i);
    let dir_y = (l_i - l_b) + (l_j - l_c);

    let len_sq = dir_x * dir_x + dir_y * dir_y;
    var base_col: vec3<f32>;

    if (len_sq > 0.0025) {
        let inv_len = inverseSqrt(len_sq);
        let dir = vec2<f32>(dir_x * inv_len, dir_y * inv_len);
        let len = sqrt(len_sq);
        let stretch = clamp((len - 0.05) * 3.5, 0.0, 1.0);

        let scale_tangent = 1.0 + stretch * 0.85;
        let scale_normal = max(0.2, 1.0 - stretch * 0.5);

        let rot = mat2x2<f32>(
            dir.y, -dir.x,
            dir.x, dir.y
        );

        let c_a = textureSampleLevel(t_neural, s_sampler, p_00 - vec2<f32>(0.0, inv_tex_size.y), 0.0).rgb;
        let c_d = textureSampleLevel(t_neural, s_sampler, p_10 - vec2<f32>(0.0, inv_tex_size.y), 0.0).rgb;
        let c_e = textureSampleLevel(t_neural, s_sampler, p_00 - vec2<f32>(inv_tex_size.x, 0.0), 0.0).rgb;
        let c_f = textureSampleLevel(t_neural, s_sampler, p_10 + vec2<f32>(inv_tex_size.x, 0.0), 0.0).rgb;
        let c_g = textureSampleLevel(t_neural, s_sampler, p_01 - vec2<f32>(inv_tex_size.x, 0.0), 0.0).rgb;
        let c_h = textureSampleLevel(t_neural, s_sampler, p_11 + vec2<f32>(inv_tex_size.x, 0.0), 0.0).rgb;
        let c_k = textureSampleLevel(t_neural, s_sampler, p_01 + vec2<f32>(0.0, inv_tex_size.y), 0.0).rgb;
        let c_l = textureSampleLevel(t_neural, s_sampler, p_11 + vec2<f32>(0.0, inv_tex_size.y), 0.0).rgb;

        let taps = array<vec2<f32>, 12>(
            vec2<f32>(0.0, -1.0) - f, vec2<f32>(1.0, -1.0) - f,
            vec2<f32>(-1.0, 0.0) - f, vec2<f32>(0.0, 0.0) - f, vec2<f32>(1.0, 0.0) - f, vec2<f32>(2.0, 0.0) - f,
            vec2<f32>(-1.0, 1.0) - f, vec2<f32>(0.0, 1.0) - f, vec2<f32>(1.0, 1.0) - f, vec2<f32>(2.0, 1.0) - f,
            vec2<f32>(0.0, 2.0) - f,  vec2<f32>(1.0, 2.0) - f
        );

        let colors = array<vec3<f32>, 12>(
            c_a, c_d,
            c_e, c_b, c_c, c_f,
            c_g, c_i, c_j, c_h,
            c_k, c_l
        );

        var easu_col = vec3<f32>(0.0);
        var weight_sum = 0.0;
        for (var idx = 0; idx < 12; idx = idx + 1) {
            let v = rot * taps[idx];
            let d = vec2<f32>(v.x * scale_tangent, v.y * scale_normal);
            let dist_sq = dot(d, d);
            let w = max(0.0, 1.0 - dist_sq * 0.35);
            let weight = w * w;
            easu_col += colors[idx] * weight;
            weight_sum += weight;
        }
        base_col = easu_col / max(weight_sum, 1e-5);
    } else {
        let s = f * f * (vec2<f32>(3.0) - vec2<f32>(2.0) * f);
        base_col = mix(mix(c_b, c_c, s.x), mix(c_i, c_j, s.x), s.y);
    }

    let min_col = min(min(min(c_b, c_c), c_i), c_j);
    let max_col = max(max(max(c_b, c_c), c_i), c_j);
    let delta = max_col - min_col;
    let min_bound = max(vec3<f32>(0.0), min_col - delta * 0.45);
    let max_bound = min(vec3<f32>(1.0), max_col + delta * 0.45);

    let cross_sum = c_b + c_c + c_i + c_j;
    let laplacian = base_col * 4.0 - cross_sum;
    let nz = abs(0.25 * (l_b + l_c + l_i + l_j) - fsr_luma(base_col));
    let noise_suppression = clamp(1.0 - nz * 3.0, 0.0, 1.0);
    let rcas_col = clamp(base_col + laplacian * (sharpness * 0.22 * noise_suppression), min_bound, max_bound);

    return rcas_col;
}

fn sample_native_highres(uv: vec2<f32>, sharpness: f32) -> vec4<f32> {
    let native_uv = (dp.viewport.xy + uv * dp.viewport.zw) / dp.screen_size;
    let inv = 1.0 / dp.screen_size;
    let raw = textureSample(t_native, s_sampler, native_uv);
    let col = raw.rgb;
    if (sharpness <= 0.01) {
        return vec4<f32>(col, raw.a);
    }
    let c_up = textureSample(t_native, s_sampler, native_uv - vec2<f32>(0.0, inv.y)).rgb;
    let c_dn = textureSample(t_native, s_sampler, native_uv + vec2<f32>(0.0, inv.y)).rgb;
    let c_lt = textureSample(t_native, s_sampler, native_uv - vec2<f32>(inv.x, 0.0)).rgb;
    let c_rt = textureSample(t_native, s_sampler, native_uv + vec2<f32>(inv.x, 0.0)).rgb;

    let min_c = min(min(min(c_up, c_dn), c_lt), c_rt);
    let max_c = max(max(max(c_up, c_dn), c_lt), c_rt);
    let delta = max_c - min_c;
    let min_bound = max(vec3<f32>(0.0), min_c - delta * 0.45);
    let max_bound = min(vec3<f32>(1.0), max_c + delta * 0.45);

    let lap = col * 4.0 - (c_up + c_dn + c_lt + c_rt);
    let sharpened = clamp(col + lap * (sharpness * 0.18), min_bound, max_bound);
    return vec4<f32>(sharpened, raw.a);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    var native_col: vec3<f32>;
    var neural_sample: vec3<f32>;

    if (dp.is_doom_res != 0u) {
        // Locked to authentic Doom pixel grid: exact integer texel fetch for both native and neural
        let pixel_coord = clamp(vec2<i32>(floor(uv * dp.screen_size)), vec2<i32>(0), vec2<i32>(dp.screen_size) - 1);
        native_col = textureLoad(t_native, pixel_coord, 0).rgb;
        neural_sample = textureLoad(t_neural, pixel_coord, 0).rgb;
    } else {
        let native_sample = sample_native_highres(uv, dp.sharpness);
        native_col = native_sample.rgb;
        if (dp.fsr_enabled != 0u) {
            neural_sample = fsr_sample_neural(uv, dp.sharpness);
        } else {
            neural_sample = textureSample(t_neural, s_sampler, uv).rgb;
        }
    }

    if (dp.has_neural == 0u) {
        return vec4<f32>(native_col, 1.0);
    }

    let n_luma = fsr_luma(native_col);
    let e_luma = fsr_luma(neural_sample);

    // Pristine Detail-Preserving Relighting:
    let base_ratio = clamp((e_luma + 0.05) / (n_luma + 0.05), 0.35, 2.8);
    let eff_ratio = clamp(pow(base_ratio, dp.intensity), 0.25, 3.2);

    let raw_chroma = (neural_sample + 0.04) / (e_luma + 0.04);
    let chroma = mix(vec3<f32>(1.0), raw_chroma, 0.40);

    let relit = clamp(native_col * eff_ratio * chroma, vec3<f32>(0.0), vec3<f32>(1.0));
    let spec_glint = max(vec3<f32>(0.0), native_col - vec3<f32>(0.75)) * 0.30;
    let neural_col = clamp(relit + spec_glint, vec3<f32>(0.0), vec3<f32>(1.0));

    if (dp.mode == 1u) {
        return vec4<f32>(neural_col, 1.0);
    }
    if (dp.mode == 2u) {
        return vec4<f32>(native_col, 1.0);
    }

    // Split screen comparison
    let split_screen_x = dp.split_x;
    let dist = abs(uv.x - split_screen_x) * dp.dest_size.x;
    if (dist < 1.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    if (dist < 2.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.7);
    }
    if (uv.x < dp.split_x) {
        return vec4<f32>(native_col, 1.0);
    } else {
        return vec4<f32>(neural_col, 1.0);
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleParams {
    viewport: [f32; 4],
    source_size: [f32; 2],
    is_doom_res: u32,
    pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayParams {
    split_x: f32,
    mode: u32,
    screen_size: [f32; 2],
    viewport: [f32; 4],
    dest_size: [f32; 2],
    fsr_enabled: u32,
    sharpness: f32,
    intensity: f32,
    has_neural: u32,
    is_doom_res: u32,
    pad: f32,
}

pub(super) struct Live {
    pub frame: Frame,
    pub gpu: Gpu,

    // Native high-res scene target (rendered by Doom at full window resolution)
    _native_texture: wgpu::Texture,
    pub view: wgpu::TextureView,

    // Low-res linear HDR scene target (256x256, fed into Frame / OpenDLSS-NR)
    _scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    bindings: InputBindings,

    // Downsample pass (native_view -> scene_view with sRGB decode)
    downsample_buf: wgpu::Buffer,
    downsample_layout: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::RenderPipeline,
    downsample_bind_group: wgpu::BindGroup,

    // Neural enhanced output texture (256x256 Rgba8Unorm, written by compose.wgsl)
    _enhanced_texture: wgpu::Texture,
    enhanced_view: wgpu::TextureView,

    // Display pipeline (FSR 3 EASU + RCAS + multiplicative relighting display pass)
    display_buf: wgpu::Buffer,
    display_layout: wgpu::BindGroupLayout,
    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,

    // Runtime state
    size: (u32, u32),
    render_size: (u32, u32),
    format: wgpu::TextureFormat,
    has_neural: bool,
    scheduled: bool,
    in_flight: Arc<AtomicBool>,
    last_inference: Instant,
    pub inferences: u64,
}

impl Live {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        _aspect: f32,
    ) -> Result<Self, String> {
        let required = wgpu::Features::SHADER_F16
            | wgpu::Features::SUBGROUP
            | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        if !device.features().contains(required) {
            return Err("Neural rendering needs a compatible Metal device; set WDOOM_NR_MODEL before launching".into());
        }
        let path = std::env::var_os("WDOOM_NR_MODEL")
            .ok_or("Set WDOOM_NR_MODEL to the extracted model directory before launching")?;
        let model = Model::load(Path::new(&path)).map_err(|e| e.to_string())?;

        let render_w = std::env::var("WDOOM_NR_WIDTH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(320)
            .clamp(64, 320);
        let render_h = std::env::var("WDOOM_NR_HEIGHT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(200)
            .clamp(33, 320);

        let gpu = Gpu {
            device: device.clone(),
            queue: queue.clone(),
            adapter: "wDoom Metal Apple Silicon".into(),
            cooperative_matrix: true,
            subgroups: true,
        };

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = Frame::new(&gpu, &model, render_w, render_h, wgpu::TextureFormat::Rgba8Unorm)
            .map_err(|e| e.to_string());
        let _ = device.poll(wgpu::PollType::Poll);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let frame = result?;
        log::info!(
            "OpenDLSS-NR ready (FSR 3 + EASU/RCAS): {} blocks, {} dispatches, {}x{} OG Doom neural target",
            frame.network.blocks,
            frame.network.dispatches(),
            render_w,
            render_h,
        );

        // 1. Native Full-Resolution Texture (Doom renders scene here)
        let (native_texture, native_view) = target(device, format, size, "NR native scene");

        // 2. Low-Res 320x200 Linear Scene Texture
        let (scene_texture, scene_view) = target(
            device,
            wgpu::TextureFormat::Rgba16Float,
            (render_w, render_h),
            "NR low-res linear scene",
        );
        let bindings = frame.bind(&gpu, &scene_view);

        // 3. Neural Output Texture (320x200 Rgba8Unorm)
        let (enhanced_texture, enhanced_view) = target(
            device,
            wgpu::TextureFormat::Rgba8Unorm,
            (render_w, render_h),
            "NR enhanced view",
        );

        // 4. Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("NR linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("NR nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // 5. Downsample Pipeline
        let (downsample_layout, downsample_pipeline) =
            downsample_pipeline(device, wgpu::TextureFormat::Rgba16Float);
        let downsample_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("downsample_params"),
            size: std::mem::size_of::<DownsampleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let downsample_bind_group =
            downsample_bind(device, &downsample_layout, &downsample_buf, &sampler, &native_view);

        // 6. Display Pipeline
        let (display_layout, display_pipeline) = display_pipeline(device, format);
        let display_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display_params"),
            size: std::mem::size_of::<DisplayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let display_bind_group = display_bind(
            device,
            &display_layout,
            &display_buf,
            &sampler,
            &native_view,
            &enhanced_view,
        );

        Ok(Self {
            frame,
            gpu,
            _native_texture: native_texture,
            view: native_view,
            _scene_texture: scene_texture,
            scene_view,
            bindings,
            downsample_buf,
            downsample_layout,
            downsample_pipeline,
            downsample_bind_group,
            _enhanced_texture: enhanced_texture,
            enhanced_view,
            display_buf,
            display_layout,
            display_pipeline,
            display_bind_group,
            sampler,
            nearest_sampler,
            size,
            render_size: (render_w, render_h),
            format,
            has_neural: false,
            scheduled: false,
            in_flight: Arc::new(AtomicBool::new(false)),
            last_inference: Instant::now() - Duration::from_secs(1),
            inferences: 0,
        })
    }

    pub fn resize(&mut self, format: wgpu::TextureFormat, size: (u32, u32)) {
        if self.size == size && self.format == format {
            return;
        }
        let (native_texture, native_view) = target(&self.gpu.device, format, size, "NR native scene");
        self.downsample_bind_group = downsample_bind(
            &self.gpu.device,
            &self.downsample_layout,
            &self.downsample_buf,
            &self.sampler,
            &native_view,
        );
        if self.format != format {
            let (layout, pipeline) = display_pipeline(&self.gpu.device, format);
            self.display_layout = layout;
            self.display_pipeline = pipeline;
            self.format = format;
        }
        self.display_bind_group = display_bind(
            &self.gpu.device,
            &self.display_layout,
            &self.display_buf,
            &self.sampler,
            &native_view,
            &self.enhanced_view,
        );
        self._native_texture = native_texture;
        self.view = native_view;
        self.size = size;
    }

    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        rect: [f32; 4],
        strength: f32,
        style: f32,
        mode: u32,
        split_x: f32,
        fsr_enabled: bool,
        sharpness: f32,
    ) -> Result<Option<wgpu::CommandBuffer>, String> {
        let size = self.size;
        self.encode_impl(
            encoder,
            output,
            None,
            size,
            rect,
            rect,
            strength,
            style,
            mode,
            split_x,
            fsr_enabled,
            sharpness,
            false,
        )
    }

    pub fn encode_custom(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        source_view: &wgpu::TextureView,
        source_size: (u32, u32),
        source_viewport: [f32; 4],
        dest_viewport: [f32; 4],
        strength: f32,
        style: f32,
        mode: u32,
        split_x: f32,
        fsr_enabled: bool,
        sharpness: f32,
        is_doom_res: bool,
    ) -> Result<Option<wgpu::CommandBuffer>, String> {
        self.encode_impl(
            encoder,
            output,
            Some(source_view),
            source_size,
            source_viewport,
            dest_viewport,
            strength,
            style,
            mode,
            split_x,
            fsr_enabled,
            sharpness,
            is_doom_res,
        )
    }

    fn encode_impl(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        source_view: Option<&wgpu::TextureView>,
        source_size: (u32, u32),
        source_viewport: [f32; 4],
        dest_viewport: [f32; 4],
        strength: f32,
        style: f32,
        mode: u32,
        split_x: f32,
        fsr_enabled: bool,
        sharpness: f32,
        is_doom_res: bool,
    ) -> Result<Option<wgpu::CommandBuffer>, String> {
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        let busy = self.in_flight.load(Ordering::Acquire);
        let min_interval = Duration::from_millis(16); // Up to 60 FPS
        let mut nr_cmd = None;

        let custom_down_bind;
        let custom_disp_bind;
        let (down_bind, disp_bind) = if let Some(src) = source_view {
            let sampler_to_use = if is_doom_res {
                &self.nearest_sampler
            } else {
                &self.sampler
            };
            custom_down_bind = downsample_bind(
                &self.gpu.device,
                &self.downsample_layout,
                &self.downsample_buf,
                sampler_to_use,
                src,
            );
            custom_disp_bind = display_bind(
                &self.gpu.device,
                &self.display_layout,
                &self.display_buf,
                sampler_to_use,
                src,
                &self.enhanced_view,
            );
            (&custom_down_bind, &custom_disp_bind)
        } else {
            (&self.downsample_bind_group, &self.display_bind_group)
        };

        if !busy && self.last_inference.elapsed() >= min_interval {
            self.in_flight.store(true, Ordering::Release);
            self.last_inference = Instant::now();

            let mut nr_encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("neural_encoder"),
                });

            // Step 1: Downsample native scene into linear HDR
            let ds_params = DownsampleParams {
                viewport: source_viewport,
                source_size: [source_size.0 as f32, source_size.1 as f32],
                is_doom_res: if is_doom_res { 1 } else { 0 },
                pad: 0.0,
            };
            self.gpu
                .queue
                .write_buffer(&self.downsample_buf, 0, bytemuck::bytes_of(&ds_params));

            let target_w = if is_doom_res { source_viewport[2] } else { self.render_size.0 as f32 };
            let target_h = if is_doom_res { source_viewport[3] } else { self.render_size.1 as f32 };

            {
                let mut dspass = nr_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("NR downsample to neural target"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                dspass.set_viewport(0.0, 0.0, target_w, target_h, 0.0, 1.0);
                dspass.set_pipeline(&self.downsample_pipeline);
                dspass.set_bind_group(0, down_bind, &[]);
                dspass.draw(0..3, 0..1);
            }

            // Step 2: Neural network inference (71 blocks + compose pass into enhanced_view)
            self.frame.encode(
                &self.gpu,
                &mut nr_encoder,
                &self.bindings,
                &self.enhanced_view,
                [0.0, 0.0, target_w, target_h],
                strength,
                style,
                42,
            );

            nr_cmd = Some(nr_encoder.finish());
            self.scheduled = true;
            self.has_neural = true;
            self.inferences += 1;
        }

        // Step 3: FSR 3 EASU + RCAS Display Pass on presentation encoder
        let disp_params = DisplayParams {
            split_x,
            mode,
            screen_size: [source_size.0 as f32, source_size.1 as f32],
            viewport: source_viewport,
            dest_size: [dest_viewport[2], dest_viewport[3]],
            fsr_enabled: if fsr_enabled { 1 } else { 0 },
            sharpness,
            intensity: strength,
            has_neural: if self.has_neural { 1 } else { 0 },
            is_doom_res: if is_doom_res { 1 } else { 0 },
            pad: 0.0,
        };
        self.gpu
            .queue
            .write_buffer(&self.display_buf, 0, bytemuck::bytes_of(&disp_params));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("NR FSR 3 display pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: if is_doom_res {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_viewport(
                dest_viewport[0],
                dest_viewport[1],
                dest_viewport[2],
                dest_viewport[3],
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                dest_viewport[0] as u32,
                dest_viewport[1] as u32,
                dest_viewport[2] as u32,
                dest_viewport[3] as u32,
            );
            pass.set_pipeline(&self.display_pipeline);
            pass.set_bind_group(0, disp_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(nr_cmd)
    }

    pub fn after_submit(&mut self) {
        if !self.scheduled {
            return;
        }
        self.scheduled = false;
        let in_flight = self.in_flight.clone();
        self.gpu.queue.on_submitted_work_done(move || {
            in_flight.store(false, Ordering::Release);
        });
    }
}

fn target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: (u32, u32),
    label: &'static str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn downsample_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NR downsample layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("NR downsample shader"),
        source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_WGSL.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("NR downsample pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("NR downsample pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    (layout, pipeline)
}

fn downsample_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    source: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NR downsample bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(source),
            },
        ],
    })
}

fn display_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NR display layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("NR display shader"),
        source: wgpu::ShaderSource::Wgsl(DISPLAY_SHADER_WGSL.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("NR display pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("NR display pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    (layout, pipeline)
}

fn display_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    native: &wgpu::TextureView,
    neural: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NR display bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(native),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(neural),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "one bounded Metal inference; requires WDOOM_NR_MODEL"]
    fn full_network_and_cached_preview_on_metal() {
        pollster::block_on(async {
            let g = Gpu::new().await.expect("Metal device");
            let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut live = Live::new(
                &g.device,
                &g.queue,
                wgpu::TextureFormat::Rgba8Unorm,
                (320, 200),
                16.0 / 9.0,
            )
            .expect("NR live preview");
            let output = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("NR integration test"),
                size: wgpu::Extent3d {
                    width: 320,
                    height: 200,
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
            let mut encoder = g.device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("test source"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &live.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.25,
                                g: 0.4,
                                b: 0.6,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            let cmd1 = live
                .encode(&mut encoder, &view, [0.0, 0.0, 320.0, 180.0], 1.0, 2.0, 1, 0.5, true, 0.8)
                .expect("first live frame");
            assert_eq!(live.inferences, 1);
            let cmd2 = live
                .encode(&mut encoder, &view, [0.0, 0.0, 320.0, 180.0], 1.0, 2.0, 1, 0.5, true, 0.8)
                .expect("cached frame");
            assert_eq!(
                live.inferences, 1,
                "in-flight inference must never be duplicated"
            );
            let read = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1280 * 200,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let head = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: live.frame.network.head.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                output.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(1280),
                        rows_per_image: Some(200),
                    },
                },
                output.size(),
            );
            encoder.copy_buffer_to_buffer(&live.frame.network.head, 0, &head, 0, head.size());
            let mut cmds = Vec::new();
            if let Some(cmd) = cmd1 {
                cmds.push(cmd);
            }
            if let Some(cmd) = cmd2 {
                cmds.push(cmd);
            }
            cmds.push(encoder.finish());
            g.queue.submit(cmds);
            live.after_submit();
            let map = |b: &wgpu::Buffer| {
                let (tx, rx) = std::sync::mpsc::channel();
                b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                g.device
                    .poll(wgpu::PollType::Wait {
                        timeout: Some(std::time::Duration::from_secs(10)),
                        submission_index: None,
                    })
                    .expect("bounded GPU wait");
                rx.recv().unwrap().expect("map");
                let data = b.slice(..).get_mapped_range().to_vec();
                b.unmap();
                data
            };
            let data = map(&read);
            let heads = map(&head);
            assert!(heads
                .chunks_exact(2)
                .all(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7c00) != 0x7c00));
            let center = &data[(90 * 320 + 160) * 4..(90 * 320 + 160) * 4 + 4];
            assert_ne!(&center[..3], &[255, 0, 255], "invalid-head marker");
            assert!(
                center[0] > 0 && center[1] > 0 && center[2] > 0,
                "preview must show scene colour"
            );
            let source = [64i32, 102, 153];
            let delta: f64 = (0..180usize)
                .flat_map(|y| (0..320usize).map(move |x| (y * 320 + x) * 4))
                .map(|i| {
                    (0..3)
                        .map(|c| (data[i + c] as i32 - source[c]).abs() as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                / (320.0 * 180.0 * 3.0);
            eprintln!("mean neural colour correction: {delta:.3} code steps");
            assert!(
                delta > 0.5,
                "trained network correction must not collapse to a near no-op"
            );
            assert_eq!(
                data[(190 * 320 + 160) * 4],
                0,
                "outside viewport remains clear for the HUD"
            );
            assert!(!live.in_flight.load(std::sync::atomic::Ordering::Acquire));
            assert!(scope.pop().await.is_none(), "NR frame validation failed");
            eprintln!(
                "71-block Metal frame and cached preview passed; only one inference submitted"
            );
        });
    }
}
