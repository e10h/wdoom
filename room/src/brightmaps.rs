//! Authored emission masks from the Brightmaps Plus pack used by the octree tracer.
//!
//! A texel emits only when a mask covers that texture or flat. The stored value
//! matches the octree tracer's shader: 0 is dark, 128 is half, 255 is full.
//! The path tracer multiplies that by 4, the same `texture_emission` scale.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

struct Mask {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

impl Mask {
    /// Emission byte at Doom texel `(u, v)` of a source image `src_w` by `src_h`.
    fn level(&self, u: i32, v: i32, src_w: i32, src_h: i32) -> u8 {
        let sw = src_w.max(1) as u32;
        let sh = src_h.max(1) as u32;
        let x = ((u.rem_euclid(src_w.max(1)) as u32) * self.w / sw).min(self.w - 1);
        let y = ((v.rem_euclid(src_h.max(1)) as u32) * self.h / sh).min(self.h - 1);
        let i = ((y * self.w + x) * 4) as usize;
        let pixel = &self.rgba[i..i + 4];
        let mask = pixel[0].max(pixel[1]).max(pixel[2]) as u32 * pixel[3] as u32 / 255;
        emission_byte(mask)
    }
}

/// `mask` is the brightmap's max channel times alpha, 0..=255.
/// 0 stays dark. Mid masks become half strength. Bright masks become full.
fn emission_byte(mask: u32) -> u8 {
    if mask == 0 {
        return 0;
    }
    let code = (1 + (mask * 6 + 254) / 255).min(7);
    if code <= 4 { 128 } else { 255 }
}

struct Pack {
    textures: HashMap<String, Mask>,
    flats: HashMap<String, Mask>,
}

impl Pack {
    fn load() -> Self {
        let Some(root) = find_dir() else {
            log::info!("no brightmap pack; texture emission is off");
            return Self {
                textures: HashMap::new(),
                flats: HashMap::new(),
            };
        };
        let mut textures = load_dir(&root);
        let mut flats = load_dir(&root.join("flats"));
        if let Some(definitions) = root.parent() {
            apply_definitions(
                &mut textures,
                &definitions.join("doom_textures.bm"),
                "texture",
            );
            apply_definitions(&mut flats, &definitions.join("doom_flats.bm"), "flat");
        }
        log::info!(
            "brightmaps: {} textures, {} flats from {}",
            textures.len(),
            flats.len(),
            root.display()
        );
        Self { textures, flats }
    }
}

fn installed() -> &'static Pack {
    static PACK: OnceLock<Pack> = OnceLock::new();
    PACK.get_or_init(Pack::load)
}

/// Emission for wall texture `name` (8-byte Doom name, uppercase, NUL padded).
pub fn wall_level(name: &[u8; 8], u: i32, v: i32, w: i32, h: i32) -> u8 {
    let key = doom_key(name);
    installed()
        .textures
        .get(&key)
        .map(|mask| mask.level(u, v, w, h))
        .unwrap_or(0)
}

/// Emission for flat `name`.
pub fn flat_level(name: &[u8; 8], u: i32, v: i32) -> u8 {
    let key = doom_key(name);
    installed()
        .flats
        .get(&key)
        .map(|mask| mask.level(u, v, 64, 64))
        .unwrap_or(0)
}

fn doom_key(name: &[u8; 8]) -> String {
    name.iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as char)
        .collect()
}

fn find_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut bases = Vec::new();
    if let Ok(path) = std::env::var("DOOM_BRIGHTMAPS_DIR") {
        bases.push(PathBuf::from(path));
    }
    // The pack lives in the octree-tracer checkout, which may sit next to
    // wdoom, next to doomit, or next to the Code directory.
    let mut dir = manifest.as_path();
    for _ in 0..6 {
        bases.push(dir.join("octree-tracer/files/gzdoom_brightmaps"));
        bases.push(dir.join("../octree-tracer/files/gzdoom_brightmaps"));
        let Some(parent) = dir.parent() else {
            break;
        };
        if parent == dir {
            break;
        }
        dir = parent;
    }
    for base in bases {
        for candidate in [
            base.join("bmplus_textures/brightmaps/textures/doom"),
            base.join("brightmaps/textures/doom"),
            base.clone(),
        ] {
            if dir_has_png(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn dir_has_png(path: &Path) -> bool {
    std::fs::read_dir(path).ok().into_iter().flatten().any(|entry| {
        entry.ok().is_some_and(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
    })
}

fn load_dir(path: &Path) -> HashMap<String, Mask> {
    let mut maps = HashMap::new();
    let Ok(entries) = std::fs::read_dir(path) else {
        return maps;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(mask) = read_png(&path) {
            maps.insert(name.to_ascii_uppercase(), mask);
        }
    }
    maps
}

fn read_png(path: &Path) -> Option<Mask> {
    let image = image::open(path).ok()?.to_rgba8();
    let w = image.width();
    let h = image.height();
    if w == 0 || h == 0 {
        return None;
    }
    Some(Mask {
        w,
        h,
        rgba: image.into_raw(),
    })
}

fn apply_definitions(maps: &mut HashMap<String, Mask>, path: &Path, kind: &str) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    for line in text.lines() {
        let line = line.split("//").next().unwrap_or_default().trim();
        let mut words = line.split_whitespace();
        if words.next() != Some("brightmap") || words.next() != Some(kind) {
            continue;
        }
        let Some(material) = words.next() else {
            continue;
        };
        let Some(start) = line.find("map \"") else {
            continue;
        };
        let rest = &line[start + 5..];
        let Some(end) = rest.find('"') else {
            continue;
        };
        let Some(stem) = Path::new(&rest[..end])
            .file_stem()
            .and_then(|name| name.to_str())
        else {
            continue;
        };
        if let Some(mask) = maps.get(&stem.to_ascii_uppercase()).cloned() {
            maps.insert(material.to_ascii_uppercase(), mask);
        }
    }
}

impl Clone for Mask {
    fn clone(&self) -> Self {
        Self {
            w: self.w,
            h: self.h,
            rgba: self.rgba.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::emission_byte;

    #[test]
    fn dark_mask_is_off_and_white_mask_is_full() {
        assert_eq!(emission_byte(0), 0);
        assert_eq!(emission_byte(255), 255);
        assert_eq!(emission_byte(1), 128);
    }

    #[test]
    fn pack_loads_computer_screen() {
        let pack = super::Pack::load();
        let mask = pack
            .textures
            .get("COMPUTE1")
            .expect("COMPUTE1 brightmap missing");
        let lit = (0..mask.w as i32)
            .flat_map(|x| (0..mask.h as i32).map(move |y| mask.level(x, y, mask.w as i32, mask.h as i32)))
            .filter(|level| *level > 0)
            .count();
        assert!(lit > 100, "COMPUTE1 has {lit} emissive texels");
    }
}
