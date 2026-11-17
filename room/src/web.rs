//! WebAssembly browser entry point for `room`.
//!
//! Provides the winit web integration, canvas binding, and VFS registration
//! so Doom can run natively in modern web browsers via WebGPU/WebGL2.

#![cfg(target_arch = "wasm32")]

use std::cell::Cell;
use std::ffi::{c_char, c_int, CString};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{Window, WindowId};

use crate::doom::doomgeneric::{DOOMGENERIC_RESX, DOOMGENERIC_RESY};
use crate::gpu::GpuState;
use crate::platform::{
    init_start_time, on_key, on_mouse_button, on_mouse_motion, set_pointer_locked, GPU,
    QUIT_REQUESTED, WINDOW,
};

/// Embedded Doom shareware episode 1 WAD (4 MB).
pub const EMBEDDED_DOOM1_WAD: &[u8] = include_bytes!("../../doom1.wad");

/// Application state for the web browser event loop.
pub struct WebApp {
    args: Vec<CString>,
    argv: Vec<*mut c_char>,
    canvas_id: String,
    doom_initialized: bool,
    gpu_initializing: bool,
}

unsafe impl Send for WebApp {}

impl WebApp {
    pub fn new(canvas_id: String) -> Self {
        let args = vec![
            CString::new("room").unwrap(),
            CString::new("-iwad").unwrap(),
            CString::new("doom1.wad").unwrap(),
        ];
        let mut argv: Vec<*mut c_char> = args.iter().map(|s| s.as_ptr() as *mut c_char).collect();
        argv.push(std::ptr::null_mut());

        Self {
            args,
            argv,
            canvas_id,
            doom_initialized: false,
            gpu_initializing: false,
        }
    }
}

impl ApplicationHandler for WebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!(
            "Web application resumed – attaching canvas '#{}' ({}×{})",
            self.canvas_id,
            DOOMGENERIC_RESX,
            DOOMGENERIC_RESY
        );

        let window_obj = match web_sys::window() {
            Some(w) => w,
            None => {
                log::error!("No global `window` exists");
                return;
            }
        };

        let document = match window_obj.document() {
            Some(d) => d,
            None => {
                log::error!("No global `document` exists");
                return;
            }
        };

        let canvas = match document.get_element_by_id(&self.canvas_id) {
            Some(elem) => match elem.dyn_into::<web_sys::HtmlCanvasElement>() {
                Ok(c) => c,
                Err(_) => {
                    log::error!("Element '#{}' is not a canvas", self.canvas_id);
                    return;
                }
            },
            None => {
                log::error!("Could not find element '#{}'", self.canvas_id);
                return;
            }
        };

        let window_attrs = Window::default_attributes()
            .with_title("Room – Doom Web")
            .with_inner_size(LogicalSize::new(
                DOOMGENERIC_RESX as u32,
                DOOMGENERIC_RESY as u32,
            ))
            .with_canvas(Some(canvas));

        let window = match event_loop.create_window(window_attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("Failed to create window on canvas: {e}");
                return;
            }
        };

        WINDOW.with_borrow_mut(|w| *w = Some(window.clone()));
        watch_pointer_lock();

        if !self.gpu_initializing {
            self.gpu_initializing = true;
            let window_clone = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match GpuState::new_async(window_clone).await {
                    Ok(gpu) => {
                        GPU.with_borrow_mut(|g| *g = Some(gpu));
                        init_start_time();
                        log::info!("WebGPU initialized successfully");
                        WINDOW.with_borrow(|w| {
                            if let Some(win) = w.as_ref() {
                                win.request_redraw();
                            }
                        });
                    }
                    Err(e) => {
                        log::error!("Failed to initialize WebGPU: {e}");
                        set_status(&format!("GPU init failed: {e}"));
                    }
                }
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                QUIT_REQUESTED.with(|q| q.set(true));
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                GPU.with_borrow_mut(|g| {
                    if let Some(gpu) = g.as_mut() {
                        gpu.resize(new_size);
                    }
                });
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == winit::event::ElementState::Pressed;
                    on_key(code, pressed, event.repeat);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == winit::event::ElementState::Pressed;
                on_mouse_button(button, pressed);
            }
            WindowEvent::RedrawRequested => {
                let gpu_ready = GPU.with_borrow(|g| g.is_some());
                if !gpu_ready {
                    log::warn!("RedrawRequested: GPU not ready");
                    WINDOW.with_borrow(|w| {
                        if let Some(win) = w.as_ref() {
                            win.request_redraw();
                        }
                    });
                    return;
                }

                if QUIT_REQUESTED.with(|q| q.get()) {
                    event_loop.exit();
                    return;
                }

                if !self.doom_initialized {
                    log::info!("RedrawRequested: initializing doomgeneric_Create...");
                    let argc = (self.argv.len() - 1) as c_int;
                    unsafe {
                        crate::doom::doomgeneric::doomgeneric_Create(argc, self.argv.as_mut_ptr());
                    }
                    self.doom_initialized = true;
                    set_status("");
                    log::info!("RedrawRequested: doomgeneric_Create completed!");
                } else {
                    crate::doom::d_main::doomgeneric_Tick();
                }

                WINDOW.with_borrow(|w| {
                    if let Some(win) = w.as_ref() {
                        win.request_redraw();
                    }
                });
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            on_mouse_motion(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // `ControlFlow::Poll` is driven by `scheduler.postTask`, which drops
        // the callback if the task is rejected and then never runs again.
        // Request the next frame from `requestAnimationFrame` instead.
        schedule_frame();
    }
}

fn watch_pointer_lock() {
    thread_local! {
        static WATCHING: Cell<bool> = const { Cell::new(false) };
    }
    if WATCHING.with(|watching| watching.replace(true)) {
        return;
    }
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    let closure = Closure::wrap(Box::new(move || {
        let locked = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|doc| doc.pointer_lock_element())
            .is_some();
        set_pointer_locked(locked);
    }) as Box<dyn FnMut()>);
    let _ = document.add_event_listener_with_callback(
        "pointerlockchange",
        closure.as_ref().unchecked_ref(),
    );
    closure.forget();
}

fn schedule_frame() {
    let Some(window) = web_sys::window() else {
        return;
    };
    thread_local! {
        static FRAME_PENDING: Cell<bool> = const { Cell::new(false) };
        static FRAME: Closure<dyn FnMut()> = Closure::new(|| {
            FRAME_PENDING.with(|pending| pending.set(false));
            WINDOW.with_borrow(|w| {
                if let Some(win) = w.as_ref() {
                    win.request_redraw();
                }
            });
        });
    }
    // Input, resize, and GPU events can all reach about_to_wait before the
    // next display frame. Coalesce them into one animation callback.
    if FRAME_PENDING.with(|pending| pending.replace(true)) {
        return;
    }
    FRAME.with(|frame| {
        if window.request_animation_frame(frame.as_ref().unchecked_ref()).is_err() {
            FRAME_PENDING.with(|pending| pending.set(false));
        }
    });
}

fn set_status(message: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    if let Some(el) = document.get_element_by_id("status") {
        el.set_text_content(Some(message));
    }
}

struct WebLogger;
impl log::Log for WebLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool { true }
    fn log(&self, record: &log::Record) {
        let msg = format!("[{}] {}", record.level(), record.args());
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&msg));
    }
    fn flush(&self) {}
}
static LOGGER: WebLogger = WebLogger;

/// Start running Doom inside the web browser.
///
/// # Arguments
/// * `canvas_id` - The HTML `id` of the `<canvas>` element to render into.
/// * `custom_wad` - Optional raw bytes of a user-provided WAD file. If omitted, the bundled `doom1.wad` is used.
#[wasm_bindgen]
pub fn start_doom(canvas_id: String, custom_wad: Option<Vec<u8>>) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Debug));

    // Register WAD file into the virtual filesystem
    let wad_bytes = custom_wad.unwrap_or_else(|| EMBEDDED_DOOM1_WAD.to_vec());
    libc::register_file("doom1.wad", wad_bytes);

    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&e.to_string()))?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    let app = WebApp::new(canvas_id);
    event_loop.spawn_app(app);

    Ok(())
}

/// Toggle 4× MSAA on/off. Works in both native resolution and Doom resolution modes.
#[wasm_bindgen]
pub fn toggle_msaa() -> bool {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.toggle_msaa()
        } else {
            false
        }
    })
}

/// Toggle game pause state without opening any menu.
#[wasm_bindgen]
pub fn toggle_pause() -> bool {
    unsafe {
        crate::doom::g_game::paused ^= 1;
        if crate::doom::g_game::paused != 0 {
            crate::doom::s_sound::S_PauseSound();
        } else {
            crate::doom::s_sound::S_ResumeSound();
        }
        crate::doom::g_game::paused != 0
    }
}

/// Check whether the game scene is currently paused.
#[wasm_bindgen]
pub fn is_paused() -> bool {
    unsafe { crate::doom::g_game::paused != 0 }
}

/// Explicitly enable or disable 4× MSAA.
#[wasm_bindgen]
pub fn set_msaa(enabled: bool) -> bool {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_msaa(enabled)
        } else {
            false
        }
    })
}

/// Check whether 4× MSAA is currently enabled.
#[wasm_bindgen]
pub fn is_msaa_enabled() -> bool {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().is_some_and(|g| g.msaa())
    })
}

/// Toggle SMAA post-antialiasing on/off. Works in both native resolution and Doom resolution modes.
#[wasm_bindgen]
pub fn toggle_smaa() -> bool {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.toggle_smaa()
        } else {
            false
        }
    })
}

/// Explicitly enable or disable SMAA.
#[wasm_bindgen]
pub fn set_smaa(enabled: bool) -> bool {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_smaa(enabled)
        } else {
            false
        }
    })
}

/// Check whether SMAA is currently enabled.
#[wasm_bindgen]
pub fn is_smaa_enabled() -> bool {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().is_some_and(|g| g.smaa())
    })
}

/// Clear only the lightmap texels currently visible in the camera view for selective rebaking.
#[wasm_bindgen]
pub fn clear_visible() {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.clear_visible();
        }
    });
}

#[wasm_bindgen]
pub fn set_normal_strength(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_normal_strength(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_normal_strength() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.0, |g| g.normal_strength())
    })
}

#[wasm_bindgen]
pub fn set_roughness_scale(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_roughness_scale(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_roughness_scale() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.0, |g| g.roughness_scale())
    })
}

#[wasm_bindgen]
pub fn set_specular_sheen(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_specular_sheen(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_specular_sheen() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.0, |g| g.specular_sheen())
    })
}

#[wasm_bindgen]
pub fn set_ssrc(val: bool) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_ssrc(val);
        }
    });
}

#[wasm_bindgen]
pub fn toggle_ssrc() -> bool {
    GPU.with_borrow_mut(|gpu| {
        gpu.as_mut().map_or(true, |g| g.toggle_ssrc())
    })
}

#[wasm_bindgen]
pub fn is_ssrc_enabled() -> bool {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(true, |g| g.ssrc())
    })
}

#[wasm_bindgen]
pub fn set_ssrc_strength(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_ssrc_strength(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_ssrc_strength() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.0, |g| g.ssrc_strength())
    })
}

#[wasm_bindgen]
pub fn set_bounce_boost(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_bounce_boost(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_bounce_boost() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.8, |g| g.bounce_boost())
    })
}

#[wasm_bindgen]
pub fn toggle_baked_light() -> bool {
    GPU.with_borrow_mut(|gpu| {
        gpu.as_mut().map_or(true, |g| g.toggle_baked_light())
    })
}

#[wasm_bindgen]
pub fn set_baked_light(val: bool) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_baked_light(val);
        }
    });
}

#[wasm_bindgen]
pub fn is_baked_light_enabled() -> bool {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(true, |g| g.baked_light())
    })
}

#[wasm_bindgen]
pub fn set_diminish(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_diminish(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_diminish() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(1.5, |g| g.diminish())
    })
}

#[wasm_bindgen]
pub fn set_light_blend(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_light_blend(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_light_blend() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(0.5, |g| g.light_blend())
    })
}

#[wasm_bindgen]
pub fn set_voxel_blend(val: f32) {
    GPU.with_borrow_mut(|gpu| {
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_voxel_blend(val);
        }
    });
}

#[wasm_bindgen]
pub fn get_voxel_blend() -> f32 {
    GPU.with_borrow(|gpu| {
        gpu.as_ref().map_or(0.4, |g| g.voxel_blend())
    })
}
