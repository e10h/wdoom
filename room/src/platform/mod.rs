//! Platform implementation for the doomgeneric engine.
//!
//! This module provides the six C-callable functions that doomgeneric
//! requires from a platform back-end:
//!
//! | C function | Purpose |
//! |------------|---------|
//! | `DG_Init` | Platform initialisation (window / GPU already set up) |
//! | `DG_DrawFrame` | Present the current frame from `DG_ScreenBuffer` |
//! | `DG_SleepMs` | Sleep for a number of milliseconds |
//! | `DG_GetTicksMs` | Return elapsed milliseconds since startup |
//! | `DG_GetKey` | Pop the next keyboard event from the queue |
//! | `DG_SetWindowTitle` | Update the OS window title |
//!
//! All functions are exported with `#[no_mangle]` so the C linker can
//! resolve the symbols when linking the doomgeneric static library.
//!
//! ## Global state
//!
//! Because these functions are called from C code (which has no concept of
//! Rust ownership), all shared mutable state is stored in [`thread_local!`]
//! statics protected by [`std::cell::RefCell`].  This is safe because
//! **all** calls originate from the main thread (the winit event loop).
//!
//! This design is an intentional *functional approximation* of the original
//! C globals, to be refactored towards idiomatic Rust in future iterations.

pub mod keys;

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::c_char;
use std::ffi::CStr;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use winit::event::MouseButton;
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Fullscreen, Window};

use crate::gpu::GpuState;

// ---------------------------------------------------------------------------
// Thread-local platform state
// ---------------------------------------------------------------------------

// GPU, window, keyboard and time state used by the DG_* callbacks.
// Doc comments are deliberately omitted on the macro invocation since
// rustdoc does not process comments attached to `thread_local!` calls.
thread_local! {
    // GPU rendering state.  `None` until `GpuState` is created in `resumed`.
    pub static GPU: RefCell<Option<GpuState>> = const { RefCell::new(None) };

    // OS window handle used for title updates.
    pub static WINDOW: RefCell<Option<Arc<Window>>> = const { RefCell::new(None) };

    // Keyboard event queue consumed by `DG_GetKey`.
    // Each entry is `(pressed, doom_key_byte)`.
    pub static KEY_QUEUE: RefCell<VecDeque<(bool, u8)>> =
        const { RefCell::new(VecDeque::new()) };

    // Absolute time of programme launch, used by `DG_GetTicksMs`.
    static START_TIME: Cell<Option<Instant>> = const { Cell::new(None) };

    // Set to `true` when the user asks to close the window.
    pub static QUIT_REQUESTED: Cell<bool> = const { Cell::new(false) };

    // Pointer-lock aim. Yaw is forwarded to Doom. Pitch is view-only:
    // looking up must not walk the player forward.
    static MOUSE: RefCell<Mouse> = const { RefCell::new(Mouse::new()) };
    static CURSOR_LOCKED: Cell<bool> = const { Cell::new(false) };
    static LOOK_PITCH: Cell<f64> = const { Cell::new(0.0) };
}

struct Mouse {
    dx: f64,
    buttons: i32,
    dirty: bool,
}

impl Mouse {
    const fn new() -> Self {
        Self { dx: 0.0, buttons: 0, dirty: false }
    }
}

/// Record a key. Escape releases the cursor so the menu can be used.
pub fn on_key(code: KeyCode, pressed: bool, repeat: bool) {
    if repeat {
        return;
    }
    if code == KeyCode::Escape && pressed {
        release_pointer();
    }
    if let Some(doom_key) = keys::to_doom_key(code) {
        KEY_QUEUE.with_borrow_mut(|q| q.push_back((pressed, doom_key)));
    }
}

/// Left click captures the cursor and fires. Other buttons keep their
/// Doom bindings (right strafe, middle forward).
pub fn on_mouse_button(button: MouseButton, pressed: bool) {
    if button == MouseButton::Left && pressed && level_playing() {
        capture_pointer();
    }
    let Some(bit) = mouse_bit(button) else {
        return;
    };
    MOUSE.with_borrow_mut(|mouse| {
        if pressed {
            mouse.buttons |= 1 << bit;
        } else {
            mouse.buttons &= !(1 << bit);
        }
        mouse.dirty = true;
    });
}

/// Turn and look while the cursor is captured.
///
/// Horizontal motion is Doom's yaw. Vertical motion pitches the 3D camera
/// from straight down to straight up, and is not sent to the engine.
pub fn on_mouse_motion(dx: f64, dy: f64) {
    if !CURSOR_LOCKED.with(|locked| locked.get()) {
        return;
    }
    if dx != 0.0 {
        MOUSE.with_borrow_mut(|mouse| {
            mouse.dx += dx;
            mouse.dirty = true;
        });
    }
    if dy != 0.0 {
        // Same radians-per-pixel as `MOUSEX * 8` shifted into a BAM angle.
        let sens = unsafe { crate::doom::m_menu::mouseSensitivity as f64 };
        let scale = (sens + 5.0) / 10.0 * 8.0 * 65536.0 / 4294967296.0 * std::f64::consts::TAU;
        let limit = std::f64::consts::FRAC_PI_2 - 0.01;
        LOOK_PITCH.with(|pitch| {
            pitch.set((pitch.get() - dy * scale).clamp(-limit, limit));
        });
    }
}

/// Vertical look in radians. Positive looks up. Clamped just short of
/// ±90° so the camera can face the sky or the floor without flipping.
pub fn look_pitch() -> f32 {
    LOOK_PITCH.with(|pitch| pitch.get() as f32)
}

/// One mouse sample for this tic: `(x_delta, button_mask)`.
///
/// `None` when nothing changed. The button mask stays latched in the engine
/// until a later sample clears a bit, so a held fire button does not need
/// to be repeated every tic.
pub fn take_mouse() -> Option<(i32, i32)> {
    MOUSE.with_borrow_mut(|mouse| {
        if !mouse.dirty {
            return None;
        }
        mouse.dirty = false;
        let dx = mouse.dx.round() as i32;
        mouse.dx = 0.0;
        Some((dx, mouse.buttons))
    })
}

/// Browser pointer-lock can end without a key event (Escape is swallowed).
pub fn set_pointer_locked(locked: bool) {
    CURSOR_LOCKED.with(|c| c.set(locked));
    if !locked {
        WINDOW.with_borrow(|w| {
            if let Some(win) = w.as_ref() {
                win.set_cursor_visible(true);
            }
        });
    }
}

fn capture_pointer() {
    let grabbed = WINDOW.with_borrow(|w| {
        let Some(win) = w.as_ref() else {
            return false;
        };
        if win.set_cursor_grab(CursorGrabMode::Locked).is_err() {
            return false;
        }
        win.set_cursor_visible(false);
        true
    });
    if grabbed {
        CURSOR_LOCKED.with(|c| c.set(true));
    }
}

/// Borderless fullscreen on the current monitor. The green window button
/// does the same thing once the window is resizable.
pub fn toggle_fullscreen() {
    WINDOW.with_borrow(|w| {
        let Some(win) = w.as_ref() else {
            return;
        };
        if win.fullscreen().is_some() {
            win.set_fullscreen(None);
        } else {
            win.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
    });
}

pub fn release_pointer() {
    CURSOR_LOCKED.with(|c| c.set(false));
    WINDOW.with_borrow(|w| {
        if let Some(win) = w.as_ref() {
            let _ = win.set_cursor_grab(CursorGrabMode::None);
            win.set_cursor_visible(true);
        }
    });
}

fn level_playing() -> bool {
    // `GS_LEVEL` is 0. The menu uses the keyboard, so leave the cursor free.
    unsafe { crate::doom::g_game::gamestate == 0 && crate::doom::m_menu::menuactive == 0 }
}

fn mouse_bit(button: MouseButton) -> Option<u32> {
    match button {
        MouseButton::Left => Some(0),
        MouseButton::Right => Some(1),
        MouseButton::Middle => Some(2),
        MouseButton::Back | MouseButton::Forward | MouseButton::Other(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Initialisation helper
// ---------------------------------------------------------------------------

/// Record the programme start time so [`DG_GetTicksMs`] can compute elapsed
/// milliseconds.  Must be called once, before `doomgeneric_Create`.
pub fn init_start_time() {
    START_TIME.with(|t| t.set(Some(Instant::now())));
}

// ---------------------------------------------------------------------------
// DG_* callbacks (C-callable)
// ---------------------------------------------------------------------------

/// Initialise the platform back-end.
///
/// Called by doomgeneric once, inside `doomgeneric_Create`, before the first
/// game tick.  Because the window and GPU are created by the winit event loop
/// *before* `doomgeneric_Create` is called, this function is a no-op: all
/// resources are already available.
#[no_mangle]
pub extern "C" fn DG_Init() {
    log::debug!("DG_Init: platform already initialised");
}

/// Present the current Doom frame to the screen.
///
/// Called by the engine's `I_FinishUpdate` function at the end of each
/// rendered frame.  Reads pixels from the C-owned `DG_ScreenBuffer` (640 ×
/// 400 pixels in BGRA8 format) and uploads them to the wgpu texture, then
/// issues a render pass that blits the texture to the surface.
///
/// # Safety
///
/// `DG_ScreenBuffer` must be non-null and point to at least
/// `DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4` bytes of valid memory.
/// This is guaranteed by `doomgeneric_Create`, which allocates the buffer
/// before calling `DG_Init`.
#[no_mangle]
pub extern "C" fn DG_DrawFrame() {
    // SAFETY: DG_ScreenBuffer is allocated by doomgeneric_Create and is
    // valid for DOOMGENERIC_PIXELS * 4 bytes.
    let pixel_bytes = unsafe {
        let ptr = crate::doom::doomgeneric::DG_ScreenBuffer as *const u8;
        if ptr.is_null() {
            log::warn!("DG_DrawFrame: DG_ScreenBuffer is null, skipping frame");
            return;
        }
        std::slice::from_raw_parts(ptr, crate::doom::doomgeneric::DOOMGENERIC_PIXELS * 4)
    };

    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT <= 5 || FRAME_COUNT % 60 == 0 {
            let mid = (200 * 640 + 320) * 4;
            let sample_0 = &pixel_bytes[0..8];
            let sample_mid = &pixel_bytes[mid..mid + 8];
            log::info!("DG_DrawFrame #{FRAME_COUNT}: p0={sample_0:02x?} p_mid={sample_mid:02x?}");
        }
    }

    GPU.with_borrow_mut(|opt| {
        let Some(gpu) = opt.as_mut() else {
            log::warn!("DG_DrawFrame: GPU state not ready");
            return;
        };
        let result = if crate::hw::active() {
            let scene = unsafe { crate::hw::build_scene() };
            let result = gpu.render_world(pixel_bytes, &scene);
            crate::hw::recycle(scene);
            result
        } else {
            gpu.render(pixel_bytes)
        };
        if let Err(e) = result {
            log::warn!("DG_DrawFrame: render error: {e}");
        }
    });
}

/// Sleep the calling thread for `ms` milliseconds.
///
/// Used by the engine to throttle the game loop when running ahead of
/// the target tick rate.
#[no_mangle]
pub extern "C" fn DG_SleepMs(ms: u32) {
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::sleep(std::time::Duration::from_millis(u64::from(ms)));
    #[cfg(target_arch = "wasm32")]
    let _ = ms;
}

/// Return the number of milliseconds elapsed since `doomgeneric_Create`.
///
/// The engine uses this to drive its fixed-rate game loop.
#[no_mangle]
pub extern "C" fn DG_GetTicksMs() -> u32 {
    START_TIME.with(|t| {
        t.get()
            .map(|start| start.elapsed().as_millis() as u32)
            .unwrap_or(0)
    })
}

/// Pop the next keyboard event from the queue.
///
/// Returns `1` if an event was available, `0` if the queue is empty.
/// On success, writes the pressed state (1 = pressed, 0 = released) to
/// `*pressed` and the Doom key byte to `*doom_key`.
///
/// # Safety
///
/// `pressed` and `doom_key` must be valid, non-null pointers to writable
/// memory for their respective types.
#[no_mangle]
pub extern "C" fn DG_GetKey(pressed: *mut i32, doom_key: *mut u8) -> i32 {
    KEY_QUEUE.with_borrow_mut(|q| {
        if let Some((is_pressed, key)) = q.pop_front() {
            // SAFETY: caller guarantees the pointers are valid.
            unsafe {
                *pressed = i32::from(is_pressed);
                *doom_key = key;
            }
            1
        } else {
            0
        }
    })
}

/// Update the OS window title.
///
/// Called by the engine when it discovers the game description from the
/// loaded WAD file (e.g. "The Ultimate DOOM").
///
/// # Safety
///
/// `title` must be a valid, null-terminated C string for the duration of
/// this call.
#[no_mangle]
pub extern "C" fn DG_SetWindowTitle(title: *const c_char) {
    if title.is_null() {
        return;
    }

    // SAFETY: caller guarantees `title` is a valid C string.
    let s = unsafe { CStr::from_ptr(title) };
    let title_str = s.to_string_lossy();

    WINDOW.with_borrow(|w| {
        if let Some(window) = w.as_ref() {
            window.set_title(&title_str);
        }
    });
}
