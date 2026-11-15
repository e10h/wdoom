//! Load native RMLT bakes from the web server without blocking the game loop.
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::hw::LightBakeKey;

pub fn load(key: LightBakeKey, texture: wgpu::Texture, queue: wgpu::Queue, size: u32, is_current: impl Fn() -> bool + 'static) {
    wasm_bindgen_futures::spawn_local(async move {
        let url = format!("light/light_e{}m{}_{:016x}.rmlt", key.episode, key.map, key.pack_hash);
        match download(&url, key, size).await {
            Ok(bytes) => {
                if !is_current() {
                    return;
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 8),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                );
                web_sys::console::info_1(&format!("Loaded light bake {url}").into());
            }
            Err(error) => web_sys::console::warn_1(&format!("Light bake {url}: {error}").into()),
        }
    });
}

async fn download(url: &str, key: LightBakeKey, size: u32) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("No browser window")?;
    let response = JsFuture::from(window.fetch_with_str(url)).await
        .map_err(|e| format!("fetch failed: {e:?}"))?
        .dyn_into::<web_sys::Response>().map_err(|_| "Invalid response")?;
    if !response.ok() {
        return Err(format!("HTTP {}", response.status()));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(|_| "No response body")?)
        .await.map_err(|_| "Could not read response body")?;
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
    decode(&bytes, key, size).ok_or_else(|| "Invalid or incompatible bake".into())
}

fn decode(bytes: &[u8], key: LightBakeKey, size: u32) -> Option<Vec<u8>> {
    let header = bytes.get(..32)?;
    let word = |offset| Some(u32::from_le_bytes(header.get(offset..offset + 4)?.try_into().ok()?));
    if &header[..4] != b"RMLT" || word(4)? != 1
        || word(8)? as i32 != key.episode || word(12)? as i32 != key.map
        || u64::from_le_bytes(header[16..24].try_into().ok()?) != key.pack_hash
        || word(24)? != size {
        return None;
    }
    let count = word(28)? as usize;
    let pixels = (size as usize).checked_mul(size as usize)?;
    if count > pixels || bytes.len() != 32usize.checked_add(count.checked_mul(12)?)? {
        return None;
    }
    let mut upload = vec![0; pixels.checked_mul(8)?];
    for texel in bytes[32..].chunks_exact(12) {
        let index = u32::from_le_bytes(texel[..4].try_into().ok()?) as usize;
        upload.get_mut(index.checked_mul(8)?..index.checked_add(1)?.checked_mul(8)?)?
            .copy_from_slice(&texel[4..]);
    }
    Some(upload)
}
