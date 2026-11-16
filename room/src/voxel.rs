//! Cheello's voxel models, used in place of world sprite billboards.
//!
//! The models are Daniel Peterson's (Cheello) Doom Voxel Mod, GZDoom KVX
//! files. A sprite frame named `POSSA` looks up `possa` in `VOXELDEF` and
//! draws that solid instead of a camera-facing card. Frames with no voxel
//! (the view weapon, blood, and anything the pack skips) stay billboards.
//!
//! KVX stores a column of vertical slabs. File Z grows downward: the pivot
//! sits on the actor's feet, so model height is `zpivot/256 - z`. File X is
//! the actor's right and file Y is the direction it faces. One voxel is one
//! map unit. Slab bytes index the VGA palette in the last 768 bytes of the
//! file, not Doom's PLAYPAL.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const TAU: f32 = std::f32::consts::TAU;

/// One face of a greedy-meshed voxel model, in model space.
///
/// X is the actor's right, Y is forward, Z is up. The origin is the actor's
/// feet.
#[derive(Clone)]
pub struct VoxelQuad {
    pub p: [[f32; 3]; 4],
    pub color: u8,
}

#[derive(Clone)]
pub struct VoxelMesh {
    pub quads: Vec<VoxelQuad>,
    /// RGB from the 768-byte VGA palette at the end of the KVX file.
    /// Indexes in the slabs point here, not at PLAYPAL.
    pub palette: [[u8; 3]; 256],
}

/// GPU-ready local-space voxel vertex (24 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct VoxelVertex {
    pub pos: [f32; 3],
    pub tint: [f32; 3],
}

impl VoxelMesh {
    pub fn build_gpu_mesh(&self) -> (Vec<VoxelVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.quads.len() * 4);
        let mut indices = Vec::with_capacity(self.quads.len() * 6);
        for quad in &self.quads {
            let base = verts.len() as u32;
            let rgb = self.palette[quad.color as usize];
            let tint = [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ];
            for i in 0..4 {
                verts.push(VoxelVertex {
                    pos: quad.p[i],
                    tint,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        (verts, indices)
    }
}

struct Def {
    file: String,
    /// Extra yaw in radians, including the spin accumulated so far is applied
    /// by the caller from `spin`.
    angle: f32,
    /// Radians per second while the thing sits in the world.
    spin: f32,
}

/// Mesh and extra yaw (radians) for a sprite frame, when this pack has one.
pub fn model_for(sprite: i32, frame: i32, leveltime: i32) -> Option<(Arc<VoxelMesh>, f32)> {
    let key = sprite_key(sprite, frame)?;
    let def = defs().get(&key)?;
    let mesh = mesh_for(&def.file)?;
    let seconds = leveltime.max(0) as f32 / 35.0;
    Some((mesh, def.angle + def.spin * seconds))
}

/// Height of the model's top relative to its feet, without decoding/meshing it.
/// KVX and sprite pivots differ (the COLU bulb is at z=43, not z=33).
pub fn model_top(sprite: i32, frame: i32) -> Option<f32> {
    let def = defs().get(&sprite_key(sprite, frame)?)?;
    let data = file_bytes(&def.file)?;
    Some(i32::from_le_bytes(data.get(24..28)?.try_into().ok()?) as f32 / 256.0)
}

fn sprite_key(sprite: i32, frame: i32) -> Option<String> {
    if sprite < 0 {
        return None;
    }
    let name = unsafe {
        let names = &raw const crate::doom::info::sprnames;
        if sprite as usize >= (*names).len() {
            return None;
        }
        let ptr = (*names)[sprite as usize];
        if ptr.is_null() {
            return None;
        }
        std::ffi::CStr::from_ptr(ptr).to_bytes()
    };
    if name.len() < 4 {
        return None;
    }
    let fr = (frame & 0x7fff) as u8;
    if fr >= 26 {
        return None;
    }
    let mut key = String::with_capacity(5);
    for &b in &name[..4] {
        key.push((b as char).to_ascii_lowercase());
    }
    key.push((b'a' + fr) as char);
    Some(key)
}

fn mesh_for(file: &str) -> Option<Arc<VoxelMesh>> {
    if let Some(found) = meshes().lock().ok()?.get(file) {
        return found.clone();
    }
    let bytes = file_bytes(file)?;
    let meshed = mesh_kvx(bytes).map(Arc::new);
    if let Ok(mut cache) = meshes().lock() {
        cache.insert(file.to_string(), meshed.clone());
    }
    meshed
}

fn meshes() -> &'static Mutex<HashMap<String, Option<Arc<VoxelMesh>>>> {
    static MESHES: OnceLock<Mutex<HashMap<String, Option<Arc<VoxelMesh>>>>> = OnceLock::new();
    MESHES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn defs() -> &'static HashMap<String, Def> {
    static DEFS: OnceLock<HashMap<String, Def>> = OnceLock::new();
    DEFS.get_or_init(parse_voxeldef)
}

fn files() -> &'static HashMap<String, (usize, usize)> {
    static FILES: OnceLock<HashMap<String, (usize, usize)>> = OnceLock::new();
    FILES.get_or_init(|| {
        let blob = blob();
        let mut map = HashMap::new();
        let mut i = 0;
        while i + 2 <= blob.len() {
            let nlen = u16::from_le_bytes(blob[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            if i + nlen + 4 > blob.len() {
                break;
            }
            let name = String::from_utf8_lossy(&blob[i..i + nlen]).into_owned();
            i += nlen;
            let dlen = u32::from_le_bytes(blob[i..i + 4].try_into().unwrap()) as usize;
            i += 4;
            if i + dlen > blob.len() {
                break;
            }
            map.insert(name, (i, dlen));
            i += dlen;
        }
        map
    })
}

fn blob() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/voxels.bin"))
}

fn file_bytes(stem: &str) -> Option<&'static [u8]> {
    let (at, len) = *files().get(stem)?;
    Some(&blob()[at..at + len])
}

fn parse_voxeldef() -> HashMap<String, Def> {
    let text = include_str!("../assets/cheello/VOXELDEF.txt");
    let mut map = HashMap::new();
    for raw in text.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let rest = rest.trim();
        let Some(q0) = rest.find('"') else {
            continue;
        };
        let Some(q1) = rest[q0 + 1..].find('"') else {
            continue;
        };
        let quoted = &rest[q0 + 1..q0 + 1 + q1];
        let file = std::path::Path::new(quoted)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(quoted)
            .to_ascii_lowercase();
        let angle = field(rest, "AngleOffset");
        let spin = field(rest, "PlacedSpin");
        map.insert(
            key,
            Def {
                file,
                angle: angle * TAU / 360.0,
                spin: spin * TAU / 360.0,
            },
        );
    }
    map
}

fn field(body: &str, name: &str) -> f32 {
    let Some(at) = body.find(name) else {
        return 0.0;
    };
    let after = body[at + name.len()..].trim_start();
    let after = after.strip_prefix('=').unwrap_or(after).trim_start();
    let num: String = after
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
        .collect();
    num.parse().unwrap_or(0.0)
}

fn mesh_kvx(data: &[u8]) -> Option<VoxelMesh> {
    if data.len() < 28 {
        return None;
    }
    let numbytes = i32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let xs = i32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
    let ys = i32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
    let zs = i32::from_le_bytes(data[12..16].try_into().ok()?) as usize;
    let xp = i32::from_le_bytes(data[16..20].try_into().ok()?) as f32 / 256.0;
    let yp = i32::from_le_bytes(data[20..24].try_into().ok()?) as f32 / 256.0;
    let zp = i32::from_le_bytes(data[24..28].try_into().ok()?) as f32 / 256.0;
    if xs == 0 || ys == 0 || zs == 0 || xs > 512 || ys > 512 || zs > 512 {
        return None;
    }
    let declared = 4 + numbytes;
    if declared > data.len() {
        return None;
    }
    let base = 28usize;
    let mut xoff = Vec::with_capacity(xs + 1);
    for i in 0..=xs {
        let at = base + i * 4;
        if at + 4 > declared {
            return None;
        }
        xoff.push(i32::from_le_bytes(data[at..at + 4].try_into().ok()?) as usize);
    }
    let stride = ys + 1;
    let xy_at = base + (xs + 1) * 4;
    if xy_at + xs * stride * 2 > declared {
        return None;
    }
    let mut grid = vec![0u16; xs * ys * zs];
    for x in 0..xs {
        let sl = xoff[x];
        for y in 0..ys {
            let row = xy_at + (x * stride + y) * 2;
            let a = u16::from_le_bytes(data[row..row + 2].try_into().ok()?) as usize;
            let b = u16::from_le_bytes(data[row + 2..row + 4].try_into().ok()?) as usize;
            if b < a {
                return None;
            }
            let start = base + sl + a;
            let end = base + sl + b;
            if end > declared {
                return None;
            }
            let mut p = start;
            while p + 3 <= end {
                let zt = data[p] as usize;
                let zl = data[p + 1] as usize;
                if zl == 0 || p + 3 + zl > end || zt + zl > zs {
                    return None;
                }
                for i in 0..zl {
                    let z = zt + i;
                    grid[(z * ys + y) * xs + x] = data[p + 3 + i] as u16 + 1;
                }
                p += 3 + zl;
            }
            if p != end {
                return None;
            }
        }
    }
    let mut mesh = greedy(xs, ys, zs, &grid, xp, yp, zp);
    mesh.palette = kvx_palette(data);
    Some(mesh)
}

/// The last 768 bytes are a 256-color VGA palette, components 0–63.
fn kvx_palette(data: &[u8]) -> [[u8; 3]; 256] {
    let mut pal = [[0u8; 3]; 256];
    if data.len() < 768 {
        return pal;
    }
    let raw = &data[data.len() - 768..];
    for i in 0..256 {
        for c in 0..3 {
            let v = raw[i * 3 + c];
            // GZDoom expands the 6-bit channel the same way: `(v << 2) | (v >> 4)`.
            pal[i][c] = (v << 2) | (v >> 4);
        }
    }
    pal
}

fn cell(grid: &[u16], xs: usize, ys: usize, x: usize, y: usize, z: usize) -> u16 {
    grid[(z * ys + y) * xs + x]
}

fn greedy(xs: usize, ys: usize, zs: usize, grid: &[u16], xp: f32, yp: f32, zp: f32) -> VoxelMesh {
    let mut quads = Vec::new();
    // +X and -X, merged across Y and Z.
    for x in 0..xs {
        grow_x(&mut quads, grid, xs, ys, zs, x, true, xp, yp, zp);
        grow_x(&mut quads, grid, xs, ys, zs, x, false, xp, yp, zp);
    }
    for y in 0..ys {
        grow_y(&mut quads, grid, xs, ys, zs, y, true, xp, yp, zp);
        grow_y(&mut quads, grid, xs, ys, zs, y, false, xp, yp, zp);
    }
    for z in 0..zs {
        grow_z(&mut quads, grid, xs, ys, zs, z, true, xp, yp, zp);
        grow_z(&mut quads, grid, xs, ys, zs, z, false, xp, yp, zp);
    }
    VoxelMesh {
        quads,
        palette: [[0; 3]; 256],
    }
}

fn grow_x(
    quads: &mut Vec<VoxelQuad>,
    grid: &[u16],
    xs: usize,
    ys: usize,
    zs: usize,
    x: usize,
    positive: bool,
    xp: f32,
    yp: f32,
    zp: f32,
) {
    let mut used = vec![false; ys * zs];
    for z in 0..zs {
        for y in 0..ys {
            if used[z * ys + y] || !exposed_x(grid, xs, ys, x, y, z, positive) {
                continue;
            }
            let color = cell(grid, xs, ys, x, y, z);
            let mut y1 = y + 1;
            while y1 < ys
                && !used[z * ys + y1]
                && exposed_x(grid, xs, ys, x, y1, z, positive)
                && cell(grid, xs, ys, x, y1, z) == color
            {
                y1 += 1;
            }
            let mut z1 = z + 1;
            'up: while z1 < zs {
                for yy in y..y1 {
                    if used[z1 * ys + yy]
                        || !exposed_x(grid, xs, ys, x, yy, z1, positive)
                        || cell(grid, xs, ys, x, yy, z1) != color
                    {
                        break 'up;
                    }
                }
                z1 += 1;
            }
            for zz in z..z1 {
                for yy in y..y1 {
                    used[zz * ys + yy] = true;
                }
            }
            let face = if positive { (x + 1) as f32 } else { x as f32 };
            quads.push(quad(
                [
                    [face, y as f32, z as f32],
                    [face, y1 as f32, z as f32],
                    [face, y1 as f32, z1 as f32],
                    [face, y as f32, z1 as f32],
                ],
                (color - 1) as u8,
                xp,
                yp,
                zp,
            ));
        }
    }
}

fn exposed_x(
    grid: &[u16],
    xs: usize,
    ys: usize,
    x: usize,
    y: usize,
    z: usize,
    positive: bool,
) -> bool {
    if cell(grid, xs, ys, x, y, z) == 0 {
        return false;
    }
    if positive {
        x + 1 >= xs || cell(grid, xs, ys, x + 1, y, z) == 0
    } else {
        x == 0 || cell(grid, xs, ys, x - 1, y, z) == 0
    }
}

fn grow_y(
    quads: &mut Vec<VoxelQuad>,
    grid: &[u16],
    xs: usize,
    ys: usize,
    zs: usize,
    y: usize,
    positive: bool,
    xp: f32,
    yp: f32,
    zp: f32,
) {
    let mut used = vec![false; xs * zs];
    for z in 0..zs {
        for x in 0..xs {
            if used[z * xs + x] || !exposed_y(grid, xs, ys, x, y, z, positive) {
                continue;
            }
            let color = cell(grid, xs, ys, x, y, z);
            let mut x1 = x + 1;
            while x1 < xs
                && !used[z * xs + x1]
                && exposed_y(grid, xs, ys, x1, y, z, positive)
                && cell(grid, xs, ys, x1, y, z) == color
            {
                x1 += 1;
            }
            let mut z1 = z + 1;
            'up: while z1 < zs {
                for xx in x..x1 {
                    if used[z1 * xs + xx]
                        || !exposed_y(grid, xs, ys, xx, y, z1, positive)
                        || cell(grid, xs, ys, xx, y, z1) != color
                    {
                        break 'up;
                    }
                }
                z1 += 1;
            }
            for zz in z..z1 {
                for xx in x..x1 {
                    used[zz * xs + xx] = true;
                }
            }
            let face = if positive { (y + 1) as f32 } else { y as f32 };
            quads.push(quad(
                [
                    [x as f32, face, z as f32],
                    [x1 as f32, face, z as f32],
                    [x1 as f32, face, z1 as f32],
                    [x as f32, face, z1 as f32],
                ],
                (color - 1) as u8,
                xp,
                yp,
                zp,
            ));
        }
    }
}

fn exposed_y(
    grid: &[u16],
    xs: usize,
    ys: usize,
    x: usize,
    y: usize,
    z: usize,
    positive: bool,
) -> bool {
    if cell(grid, xs, ys, x, y, z) == 0 {
        return false;
    }
    if positive {
        y + 1 >= ys || cell(grid, xs, ys, x, y + 1, z) == 0
    } else {
        y == 0 || cell(grid, xs, ys, x, y - 1, z) == 0
    }
}

fn grow_z(
    quads: &mut Vec<VoxelQuad>,
    grid: &[u16],
    xs: usize,
    ys: usize,
    zs: usize,
    z: usize,
    positive: bool,
    xp: f32,
    yp: f32,
    zp: f32,
) {
    let mut used = vec![false; xs * ys];
    for y in 0..ys {
        for x in 0..xs {
            if used[y * xs + x] || !exposed_z(grid, xs, ys, zs, x, y, z, positive) {
                continue;
            }
            let color = cell(grid, xs, ys, x, y, z);
            let mut x1 = x + 1;
            while x1 < xs
                && !used[y * xs + x1]
                && exposed_z(grid, xs, ys, zs, x1, y, z, positive)
                && cell(grid, xs, ys, x1, y, z) == color
            {
                x1 += 1;
            }
            let mut y1 = y + 1;
            'up: while y1 < ys {
                for xx in x..x1 {
                    if used[y1 * xs + xx]
                        || !exposed_z(grid, xs, ys, zs, xx, y1, z, positive)
                        || cell(grid, xs, ys, xx, y1, z) != color
                    {
                        break 'up;
                    }
                }
                y1 += 1;
            }
            for yy in y..y1 {
                for xx in x..x1 {
                    used[yy * xs + xx] = true;
                }
            }
            let face = if positive { z as f32 } else { (z + 1) as f32 };
            // File +Z is model down, so the upward face is the low file-z side.
            quads.push(quad(
                [
                    [x as f32, y as f32, face],
                    [x1 as f32, y as f32, face],
                    [x1 as f32, y1 as f32, face],
                    [x as f32, y1 as f32, face],
                ],
                (color - 1) as u8,
                xp,
                yp,
                zp,
            ));
        }
    }
}

fn exposed_z(
    grid: &[u16],
    xs: usize,
    ys: usize,
    zs: usize,
    x: usize,
    y: usize,
    z: usize,
    positive: bool,
) -> bool {
    if cell(grid, xs, ys, x, y, z) == 0 {
        return false;
    }
    // `positive` here means the upward model face, which is the low-Z side of the cell.
    if positive {
        z == 0 || cell(grid, xs, ys, x, y, z - 1) == 0
    } else {
        z + 1 >= zs || cell(grid, xs, ys, x, y, z + 1) == 0
    }
}

fn quad(file: [[f32; 3]; 4], color: u8, xp: f32, yp: f32, zp: f32) -> VoxelQuad {
    let mut p = [[0.0; 3]; 4];
    for i in 0..4 {
        p[i] = [file[i][0] - xp, file[i][1] - yp, zp - file[i][2]];
    }
    VoxelQuad { p, color }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrel_sits_on_its_pivot() {
        let bytes = include_bytes!("../assets/cheello/voxels/bar1a.kvx");
        let mesh = mesh_kvx(bytes).expect("barrel");
        assert!(mesh.quads.len() > 50, "quads {}", mesh.quads.len());
        assert!(mesh.quads.len() < 2721 * 6, "quads {}", mesh.quads.len());
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for q in &mesh.quads {
            for p in q.p {
                min_z = min_z.min(p[2]);
                max_z = max_z.max(p[2]);
            }
        }
        assert!(min_z.abs() < 0.01, "floor {min_z}");
        assert!((max_z - 33.0).abs() < 0.01, "top {max_z}");
        // Index 15 is the barrel's most used color: black in the file palette,
        // a brown in PLAYPAL. Sampling PLAYPAL is what made the models look off.
        assert_eq!(mesh.palette[15], [0, 0, 0]);
    }

    #[test]
    fn voxeldef_names_a_zombie_and_a_shell() {
        let defs = parse_voxeldef();
        assert_eq!(defs.get("possa").map(|d| d.file.as_str()), Some("possa"));
        assert_eq!(
            defs.get("pinve").map(|d| d.file.as_str()),
            Some("pinv_shell")
        );
        assert!(defs.get("bar1a").unwrap().angle.abs() > 1.0);
    }

    #[test]
    fn every_packed_kvx_meshes() {
        for (name, (at, len)) in files() {
            let mesh = mesh_kvx(&blob()[*at..*at + *len]).unwrap_or_else(|| panic!("{name}"));
            assert!(!mesh.quads.is_empty(), "{name}");
        }
    }
}
