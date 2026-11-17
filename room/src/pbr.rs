//! PBR material masks (Normal map, Roughness, Metallic) for WDoom.
//! Combines authored GZDoom PBR definitions with procedural normal/roughness generation.

use std::collections::HashMap;
use std::sync::OnceLock;

pub struct PbrMask {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 4]>,
}

fn authored_masks() -> &'static HashMap<[u8; 8], PbrMask> {
    static MASKS: OnceLock<HashMap<[u8; 8], PbrMask>> = OnceLock::new();
    MASKS.get_or_init(|| {
        let data = include_bytes!("../assets/pbr/doom_pbr.bin");
        if data.len() < 8 || &data[..4] != b"RPBR" {
            return HashMap::new();
        }
        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let mut at = 8;
        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            if at + 20 > data.len() {
                break;
            }
            let name: [u8; 8] = data[at..at + 8].try_into().unwrap();
            let w = u16::from_le_bytes(data[at + 8..at + 10].try_into().unwrap()) as usize;
            let h = u16::from_le_bytes(data[at + 10..at + 12].try_into().unwrap()) as usize;
            let comp_len = u32::from_le_bytes(data[at + 12..at + 16].try_into().unwrap()) as usize;
            let uncomp_len = u32::from_le_bytes(data[at + 16..at + 20].try_into().unwrap()) as usize;
            at += 20;
            if at + comp_len > data.len() {
                break;
            }
            if let Ok(decompressed) = miniz_oxide::inflate::decompress_to_vec_zlib(&data[at..at + comp_len]) {
                if decompressed.len() == uncomp_len && decompressed.len() == w * h * 4 {
                    let mut pixels = Vec::with_capacity(w * h);
                    for chunk in decompressed.chunks_exact(4) {
                        pixels.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    map.insert(name, PbrMask { width: w, height: h, pixels });
                }
            }
            at += comp_len;
        }
        map
    })
}

/// Sample authored PBR pixel if present: [NormalX, NormalY, Roughness, Metallic].
pub fn sample_authored(name: &[u8; 8], u: i32, v: i32, width: i32, height: i32) -> Option<[u8; 4]> {
    let mask = authored_masks().get(name)?;
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let x = (u.rem_euclid(w as i32) as usize * mask.width / w).min(mask.width - 1);
    let y = (v.rem_euclid(h as i32) as usize * mask.height / h).min(mask.height - 1);
    Some(mask.pixels[y * mask.width + x])
}

fn is_metallic_name(name: &[u8; 8]) -> bool {
    const PREFIXES: &[&[u8]] = &[
        b"COMP", b"LITE", b"SHAWN", b"TEKW", b"DOOR", b"EXIT", b"STARG",
        b"METAL", b"BRONZE", b"SILVER", b"PIPE", b"GRAY", b"PANEL",
        b"SUPPORT", b"DOORTRAK", b"STEP",
    ];
    PREFIXES.iter().any(|p| name.starts_with(p))
}

fn is_liquid_name(name: &[u8; 8]) -> bool {
    const PREFIXES: &[&[u8]] = &[b"NUKAGE", b"SLIME", b"BLOOD", b"LAVA", b"WATER", b"FWATER", b"FLAVA"];
    PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Procedural PBR generator: Sobel filter for normals, roughness and metalness.
pub fn generate_procedural(name: &[u8; 8], w: i32, h: i32, rgb_pixels: &[u8]) -> Vec<[u8; 4]> {
    let (uw, uh) = (w.max(1) as usize, h.max(1) as usize);
    let mut lum = Vec::with_capacity(uw * uh);
    for chunk in rgb_pixels.chunks_exact(3) {
        let (r, g, b) = (chunk[0] as f32 / 255.0, chunk[1] as f32 / 255.0, chunk[2] as f32 / 255.0);
        lum.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    while lum.len() < uw * uh {
        lum.push(0.5);
    }

    let is_metal = is_metallic_name(name);
    let is_liquid = is_liquid_name(name);
    let bump_scale = if is_liquid { 1.0 } else { 2.5 };

    let mut out = Vec::with_capacity(uw * uh);
    for y in 0..uh {
        let yu = if y == 0 { uh - 1 } else { y - 1 };
        let yd = if y + 1 == uh { 0 } else { y + 1 };
        for x in 0..uw {
            let xl = if x == 0 { uw - 1 } else { x - 1 };
            let xr = if x + 1 == uw { 0 } else { x + 1 };

            let tl = lum[yu * uw + xl];
            let tc = lum[yu * uw + x];
            let tr = lum[yu * uw + xr];
            let ml = lum[y * uw + xl];
            let mr = lum[y * uw + xr];
            let bl = lum[yd * uw + xl];
            let bc = lum[yd * uw + x];
            let br = lum[yd * uw + xr];

            let dx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * bump_scale / 8.0;
            let dy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * bump_scale / 8.0;
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let nx = ((-dx / len) * 0.5 + 0.5).clamp(0.0, 1.0);
            let ny = ((-dy / len) * 0.5 + 0.5).clamp(0.0, 1.0);

            let edge = (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0);
            let (rough, met) = if is_liquid {
                (0.12 + edge * 0.1, 0.0)
            } else if is_metal {
                (0.35 + edge * 0.45, (0.85 - edge * 0.5).max(0.0))
            } else {
                (0.75 + edge * 0.2, 0.0)
            };

            out.push([
                (nx * 255.0).round() as u8,
                (ny * 255.0).round() as u8,
                (rough * 255.0).round() as u8,
                (met * 255.0).round() as u8,
            ]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_masks_load_and_have_records() {
        let masks = authored_masks();
        assert!(!masks.is_empty(), "expected authored PBR masks");
        assert!(masks.contains_key(b"TEKWALL4"), "TEKWALL4 should be in PBR pack");
    }

    #[test]
    fn procedural_pbr_generates_valid_tangent_normals() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let pbr = generate_procedural(b"STARTAN3", 64, 64, &pixels);
        assert_eq!(pbr.len(), 64 * 64);
        // Flat gray texture has normal [128, 128] (facing directly outward)
        assert_eq!(pbr[0][0], 128);
        assert_eq!(pbr[0][1], 128);
    }
}

