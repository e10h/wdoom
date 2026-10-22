//! egui panel for the hardware path tracer.

use std::sync::Arc;

use winit::event::WindowEvent;
use winit::window::Window;

use crate::gpu::TraceSettings;

pub struct TraceUi {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    window: Arc<Window>,
}

impl TraceUi {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, window: Arc<Window>) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        Self {
            ctx,
            state,
            renderer,
            window,
        }
    }

    pub fn on_event(&mut self, event: &WindowEvent) -> bool {
        self.state.on_window_event(&self.window, event).consumed
    }

    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        controls: &mut TraceSettings,
        traced: (u32, u32),
    ) {
        let input = self.state.take_egui_input(&self.window);
        self.ctx.begin_pass(input);
        egui::Window::new("Path tracer")
            .default_pos(egui::pos2(12.0, 12.0))
            .resizable(false)
            .show(&self.ctx, |ui| {
                ui.label("F1 frees the mouse. Option-Return toggles fullscreen.");
                ui.add(
                    egui::Slider::new(&mut controls.emission, 0.0..=80.0)
                        .text("Emissive photons"),
                );
                ui.add(
                    egui::Slider::new(&mut controls.environment, 0.0..=8.0)
                        .text("Environment photons"),
                );
                ui.add(egui::Slider::new(&mut controls.bounces, 1..=8).text("Bounces"));
                let mut height = controls.height as i32;
                ui.add(egui::Slider::new(&mut height, 160..=1080).text("Ray height"));
                controls.height = height as u32;
                ui.label(format!("Tracing {}×{}", traced.0, traced.1));
            });
        let output = self.ctx.end_pass();
        self.state
            .handle_platform_output(&self.window, output.platform_output);
        // The font atlas is Managed(0). It has to be uploaded before the panel draws.
        for (id, image) in &output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image);
        }
        let primitives = self
            .ctx
            .tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: output.pixels_per_point,
        };
        let _ = self
            .renderer
            .update_buffers(device, queue, encoder, &primitives, &screen);
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("doom_trace_ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            })
            .forget_lifetime();
        self.renderer.render(&mut pass, &primitives, &screen);
        drop(pass);
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
