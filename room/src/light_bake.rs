//! Persistent path-traced light for a level.
//!
//! The GPU light map is sparse: only texels that have received samples are
//! written. Files live next to Doom savegames as `light_e{E}m{M}_{hash}.rmlt`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::hw::LightBakeKey;

const MAGIC: &[u8; 4] = b"RMLT";
const VERSION: u32 = 1;

/// One lit texel. Colour channels are IEEE half floats (`Rgba16Float` bits).
#[derive(Clone, Copy)]
pub struct BakeTexel {
    pub index: u32,
    pub rgba: [u16; 4],
}

/// Directory that already holds `.savegame` files, when Doom has opened one.
fn bake_dir() -> Option<PathBuf> {
    unsafe {
        let dir = crate::doom::d_main::savegamedir;
        if dir.is_null() {
            return None;
        }
        let c = std::ffi::CStr::from_ptr(dir);
        let s = c.to_str().ok()?;
        if s.is_empty() {
            return None;
        }
        Some(PathBuf::from(s))
    }
}

pub fn bake_path(key: LightBakeKey) -> Option<PathBuf> {
    let dir = bake_dir()?;
    Some(dir.join(format!(
        "light_e{}m{}_{:016x}.rmlt",
        key.episode, key.map, key.pack_hash
    )))
}

pub fn write_bake(path: &Path, key: LightBakeKey, size: u32, texels: &[BakeTexel]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(key.episode as i32).to_le_bytes())?;
    file.write_all(&(key.map as i32).to_le_bytes())?;
    file.write_all(&key.pack_hash.to_le_bytes())?;
    file.write_all(&size.to_le_bytes())?;
    file.write_all(&(texels.len() as u32).to_le_bytes())?;
    for t in texels {
        file.write_all(&t.index.to_le_bytes())?;
        for c in t.rgba {
            file.write_all(&c.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn read_bake(path: &Path, expect: LightBakeKey, size: u32) -> Option<Vec<BakeTexel>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let mut u32b = [0u8; 4];
    let mut u64b = [0u8; 8];
    file.read_exact(&mut u32b).ok()?;
    if u32::from_le_bytes(u32b) != VERSION {
        return None;
    }
    file.read_exact(&mut u32b).ok()?;
    let episode = i32::from_le_bytes(u32b);
    file.read_exact(&mut u32b).ok()?;
    let map = i32::from_le_bytes(u32b);
    file.read_exact(&mut u64b).ok()?;
    let pack_hash = u64::from_le_bytes(u64b);
    file.read_exact(&mut u32b).ok()?;
    let file_size = u32::from_le_bytes(u32b);
    if episode != expect.episode
        || map != expect.map
        || pack_hash != expect.pack_hash
        || file_size != size
    {
        return None;
    }
    file.read_exact(&mut u32b).ok()?;
    let count = u32::from_le_bytes(u32b) as usize;
    let mut texels = Vec::with_capacity(count);
    let mut idx = [0u8; 4];
    let mut half = [0u8; 2];
    for _ in 0..count {
        file.read_exact(&mut idx).ok()?;
        let index = u32::from_le_bytes(idx);
        if index >= size * size {
            return None;
        }
        let mut rgba = [0u16; 4];
        for c in &mut rgba {
            file.read_exact(&mut half).ok()?;
            *c = u16::from_le_bytes(half);
        }
        // Alpha 0 means empty; skip corrupt zeros.
        if rgba[3] != 0 {
            texels.push(BakeTexel { index, rgba });
        }
    }
    Some(texels)
}

/// Pull non-empty texels out of a tightly packed `Rgba16Float` download
/// (`bytes_per_row` may be padded to 256).
pub fn sparse_from_padded(bytes: &[u8], size: u32, bytes_per_row: u32) -> Vec<BakeTexel> {
    let side = size as usize;
    let stride = bytes_per_row as usize;
    let mut out = Vec::new();
    for y in 0..side {
        let row = &bytes[y * stride..y * stride + side * 8];
        for x in 0..side {
            let o = x * 8;
            let a = u16::from_le_bytes([row[o + 6], row[o + 7]]);
            if a == 0 {
                continue;
            }
            let r = u16::from_le_bytes([row[o], row[o + 1]]);
            let g = u16::from_le_bytes([row[o + 2], row[o + 3]]);
            let b = u16::from_le_bytes([row[o + 4], row[o + 5]]);
            out.push(BakeTexel {
                index: (y * side + x) as u32,
                rgba: [r, g, b, a],
            });
        }
    }
    out
}

/// Build a padded upload buffer from sparse texels.
pub fn padded_from_sparse(texels: &[BakeTexel], size: u32) -> (Vec<u8>, u32) {
    let side = size as usize;
    let row = side * 8;
    let stride = row.div_ceil(256) * 256;
    let mut bytes = vec![0u8; stride * side];
    for t in texels {
        let x = (t.index % size) as usize;
        let y = (t.index / size) as usize;
        if y >= side {
            continue;
        }
        let o = y * stride + x * 8;
        for (i, c) in t.rgba.iter().enumerate() {
            let b = c.to_le_bytes();
            bytes[o + i * 2] = b[0];
            bytes[o + i * 2 + 1] = b[1];
        }
    }
    (bytes, stride as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_round_trip_keeps_lit_texels() {
        let size = 4u32;
        let texels = vec![
            BakeTexel {
                index: 0,
                rgba: [1, 2, 3, 4],
            },
            BakeTexel {
                index: 15,
                rgba: [9, 8, 7, 6],
            },
        ];
        let (bytes, stride) = padded_from_sparse(&texels, size);
        let back = sparse_from_padded(&bytes, size, stride);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].index, 0);
        assert_eq!(back[0].rgba, [1, 2, 3, 4]);
        assert_eq!(back[1].index, 15);
        assert_eq!(back[1].rgba, [9, 8, 7, 6]);
    }
}
