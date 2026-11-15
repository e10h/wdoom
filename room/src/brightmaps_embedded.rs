//! Native-authored mask data; browser control-panel self-glow is disabled.
use std::collections::HashMap;
use std::sync::OnceLock;

struct Mask {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}
fn masks() -> &'static HashMap<(u8, [u8; 8]), Mask> {
    static MASKS: OnceLock<HashMap<(u8, [u8; 8]), Mask>> = OnceLock::new();
    MASKS.get_or_init(|| {
        let data = include_bytes!("../assets/brightmaps/doom.rle");
        assert_eq!(&data[..4], b"RBM1");
        let mut at = 8;
        let mut masks = HashMap::new();
        for _ in 0..u32::from_le_bytes(data[4..8].try_into().unwrap()) {
            let kind = data[at];
            let name = data[at + 1..at + 9].try_into().unwrap();
            let width = u16::from_le_bytes(data[at + 9..at + 11].try_into().unwrap()) as usize;
            let height = u16::from_le_bytes(data[at + 11..at + 13].try_into().unwrap()) as usize;
            let bytes = u32::from_le_bytes(data[at + 13..at + 17].try_into().unwrap()) as usize;
            at += 17;
            let mut pixels = Vec::with_capacity(width * height);
            for run in data[at..at + bytes].chunks_exact(3) {
                let count = u16::from_le_bytes([run[0], run[1]]) as usize;
                pixels.resize(pixels.len() + count, run[2]);
            }
            assert_eq!(pixels.len(), width * height);
            masks.insert(
                (kind, name),
                Mask {
                    width,
                    height,
                    pixels,
                },
            );
            at += bytes;
        }
        assert_eq!(at, data.len());
        masks
    })
}
fn level(kind: u8, name: &[u8; 8], u: i32, v: i32, width: i32, height: i32) -> u8 {
    let Some(mask) = masks().get(&(kind, *name)) else {
        return 0;
    };
    let (width, height) = (width.max(1), height.max(1));
    let x = u.rem_euclid(width) as usize * mask.width / width as usize;
    let y = v.rem_euclid(height) as usize * mask.height / height as usize;
    mask.pixels[y * mask.width + x]
}
pub fn wall_level(name: &[u8; 8], u: i32, v: i32, width: i32, height: i32) -> u8 {
    let authored = level(0, name, u, v, width, height);
    // Keep panels lit only by the native bake. Liquids and lamps retain
    // their authored self-glow.
    if name.starts_with(b"COMP") || name.starts_with(b"SW1") || name.starts_with(b"SW2") {
        0
    } else {
        authored
    }
}
pub fn flat_level(name: &[u8; 8], u: i32, v: i32) -> u8 {
    level(1, name, u, v, 64, 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_browser_mask_matches_native() {
        assert!(masks().len() > 50);
        for (&(kind, name), mask) in masks() {
            for y in 0..mask.height {
                for x in 0..mask.width {
                    let native = if kind == 0 {
                        crate::brightmaps::wall_level(
                            &name,
                            x as i32,
                            y as i32,
                            mask.width as i32,
                            mask.height as i32,
                        )
                    } else {
                        crate::brightmaps::flat_level(&name, x as i32, y as i32)
                    };
                    assert_eq!(
                        mask.pixels[y * mask.width + x],
                        native,
                        "mask {:?}, pixel {x},{y}",
                        name
                    );
                }
            }
        }
        assert!((0..64).any(|y| (0..64).any(|x| flat_level(b"NUKAGE1\0", x, y) > 0)));
    }
    #[test]
    fn panels_are_restrained_while_lamps_and_liquids_keep_authored_glow() {
        let mut panels = 0;
        for (&(kind, name), mask) in masks() {
            for y in 0..mask.height {
                for x in 0..mask.width {
                    let raw = mask.pixels[y * mask.width + x];
                    if kind == 0 {
                        let panel = name.starts_with(b"COMP")
                            || name.starts_with(b"SW1")
                            || name.starts_with(b"SW2");
                        assert_eq!(
                            wall_level(
                                &name,
                                x as i32,
                                y as i32,
                                mask.width as i32,
                                mask.height as i32
                            ),
                            if panel { 0 } else { raw }
                        );
                        if panel && raw > 0 {
                            panels += 1;
                        }
                    } else {
                        assert_eq!(
                            level(
                                1,
                                &name,
                                x as i32,
                                y as i32,
                                mask.width as i32,
                                mask.height as i32
                            ),
                            raw
                        );
                    }
                }
            }
        }
        assert!(panels > 100);
    }
}
