//! Triangle scene built from the loaded Doom map.
//!
//! The software renderer still draws the status bar, menus, the automap, and
//! the title sequence. While a level is being played, [`active`] is true:
//! `R_RenderPlayerView` only sets up the camera, and [`build_scene`] turns
//! sectors, segs, and things into a mesh.
//!
//! Map coordinates become a right-handed Y-up world: Doom X stays X, Doom Z
//! (height) becomes Y, and Doom Y becomes world −Z. Angle 0 still faces east.

use crate::doom::am_map::automapactive;
use crate::doom::c_ffi::{sector_t, seg_t, side_t, subsector_t, vertex_t, LinedefFlag};
use crate::doom::d_player::{consoleplayer, players, PspdefT, MAXPLAYERS, NUMPSPRITES};
use crate::doom::i_video::{hw_palette_rgb, SCREENHEIGHT, SCREENWIDTH};
use crate::doom::info::State;
use crate::doom::m_menu::menuactive;
use crate::doom::p_setup::{
    nodes, numnodes, numsectors, numsegs, numsubsectors, numvertexes, sectors, segs, subsectors,
    vertexes,
};
use crate::doom::p_telept::mobj_t;
use crate::doom::p_tick::leveltime;
use crate::doom::r_data::{
    firstflat, firstspritelump, flattranslation, hw_texture_size, numflats, numspritelumps,
    numtextures, spriteoffset, spritetopoffset, spritewidth, textureheight, texturetranslation,
    R_GetColumn,
};
use crate::doom::r_draw::{viewheight, viewwidth, viewwindowx, viewwindowy};
use crate::doom::r_main::{viewangle, viewx, viewy, viewz};
use crate::doom::r_sky::{skyflatnum, skytexture};
use crate::doom::r_things::{hw_psprite_lump, hw_sprite_lump};
use crate::doom::w_wad::{W_CacheLumpNum, W_LumpLength};
use crate::doom::z_zone::PU_CACHE;
use crate::voxel::{self, VoxelMesh};
use std::collections::HashSet;
use std::sync::Arc;

const FRAC: f32 = 65536.0;
const FF_FULLBRIGHT: i32 = 0x8000;
const MF_SHADOW: i32 = 0x00040000;
/// `GS_LEVEL` in `g_game.rs`. The constant itself is private to that module.
const GS_LEVEL: i32 = 0;

/// One mesh vertex. `atlas` is the texture rectangle in the shared atlas
/// (u0, v0, u1, v1). `uv` is in tiles; the shader wraps it inside that rectangle.
/// `tint` is multiplied with the texel. Walls and sprites leave it white.
/// Voxels put their own palette color here and sample a white texel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub atlas: [f32; 4],
    pub light: f32,
    pub tint: [f32; 3],
    pub _pad: f32,
}

/// One frame of world geometry plus the camera and the HUD hole.
pub struct Scene {
    pub verts: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Index count of sky, walls, and flats. Hardware ray tracing traces this
    /// prefix. Sprites follow, through `world_indices`, and stay rasterized
    /// so their transparent texels still discard.
    pub opaque_indices: u32,
    /// Index count of sky, walls, flats, and sprites. The rest are the weapon,
    /// drawn with depth testing disabled so nearby walls do not clip it.
    pub world_indices: u32,
    pub view_proj: [[f32; 4]; 4],
    /// Camera basis for primary rays. `right`, `up`, and `forward` are unit
    /// vectors. `aspect` is the projection aspect used by [`view_proj`].
    pub eye: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub aspect: f32,
    /// Fraction of the framebuffer covered by the 3D view: x, y, width, height.
    /// The HUD pass discards this rectangle and keeps the status bar.
    pub view_rect: [f32; 4],
    /// Fresh atlas pixels, present only on the frame the atlas is rebuilt.
    /// The third buffer is the brightmap emission, one byte per texel.
    pub atlas: Option<(u32, Vec<u8>, Vec<u8>)>,
}

struct Atlas {
    /// `vertexes` pointer identity, so a new map rebuilds the atlas.
    key: usize,
    tex_rects: Vec<[f32; 4]>,
    flat_rects: Vec<[f32; 4]>,
    sprite_rects: Vec<[f32; 4]>,
    sprite_size: Vec<(i32, i32)>,
    /// A solid white texel. Voxel faces sample this and carry their color in `tint`.
    white: [f32; 4],
    pending: Option<(u32, Vec<u8>, Vec<u8>)>,
}

struct Scratch {
    verts: Vec<Vertex>,
    indices: Vec<u32>,
    atlas: Option<Atlas>,
    /// Closed floor outlines in Doom map units, one per subsector.
    /// Rebuilt when a new map is loaded. Segs alone do not form these
    /// outlines: the BSP splits are not stored as edges.
    flats: Vec<Vec<[f32; 2]>>,
    flats_key: usize,
    logged: bool,
}

static mut SCRATCH: Scratch = Scratch {
    verts: Vec::new(),
    indices: Vec::new(),
    atlas: None,
    flats: Vec::new(),
    flats_key: 0,
    logged: false,
};

/// True when the frame should be a triangle scene instead of the software view.
pub fn active() -> bool {
    unsafe {
        crate::doom::g_game::gamestate == GS_LEVEL
            && numsubsectors > 0
            && numtextures > 0
            && menuactive == 0
            && automapactive == 0
            && viewwidth > 1
            && viewheight > 1
    }
}

/// Build the mesh for the camera `R_SetupFrame` just installed.
///
/// # Safety
/// Call only from the game thread, after map data and texture tables exist.
pub unsafe fn build_scene() -> Scene {
    let scratch = &mut *std::ptr::addr_of_mut!(SCRATCH);
    let atlas_upload = ensure_atlas(scratch);
    scratch.verts.clear();
    scratch.indices.clear();

    let eye = [fx(viewx), fx(viewz), -fx(viewy)];
    let ang = viewangle as f32 * std::f32::consts::TAU / 4294967296.0;
    let pitch = crate::platform::look_pitch();
    // Angle 0 faces east. North is world −Z, so the camera's right is south.
    // Pitch raises that heading without tilting the horizon.
    let yaw_forward = [ang.cos(), 0.0, -ang.sin()];
    let right = [ang.sin(), 0.0, ang.cos()];
    let (sp, cp) = pitch.sin_cos();
    let forward = [yaw_forward[0] * cp, sp, yaw_forward[2] * cp];
    let cam_up = cross(right, forward);

    ensure_flats(scratch);
    if let Some(atlas) = scratch.atlas.as_ref() {
        push_sky(&mut scratch.verts, &mut scratch.indices, atlas, eye, ang);
        push_world(
            &mut scratch.verts,
            &mut scratch.indices,
            atlas,
            &scratch.flats,
        );
    }
    let opaque_indices = scratch.indices.len() as u32;
    if let Some(atlas) = scratch.atlas.as_ref() {
        push_sprites(&mut scratch.verts, &mut scratch.indices, atlas, right);
    }
    let world_indices = scratch.indices.len() as u32;
    if let Some(atlas) = scratch.atlas.as_ref() {
        push_weapon(
            &mut scratch.verts,
            &mut scratch.indices,
            atlas,
            eye,
            forward,
            right,
            cam_up,
        );
    }

    let aspect = viewwidth as f32 / viewheight as f32;
    let view_proj = mul(perspective(aspect), look_at(eye, forward));
    let view_rect = [
        viewwindowx as f32 / SCREENWIDTH as f32,
        viewwindowy as f32 / SCREENHEIGHT as f32,
        viewwidth as f32 / SCREENWIDTH as f32,
        viewheight as f32 / SCREENHEIGHT as f32,
    ];

    if !scratch.logged {
        scratch.logged = true;
        log::info!(
            "triangle view: {} vertices, {} indices",
            scratch.verts.len(),
            scratch.indices.len()
        );
    }

    let verts = std::mem::take(&mut scratch.verts);
    let indices = std::mem::take(&mut scratch.indices);
    Scene {
        verts,
        indices,
        opaque_indices,
        world_indices,
        view_proj,
        eye,
        right,
        up: cam_up,
        forward,
        aspect,
        view_rect,
        atlas: atlas_upload,
    }
}

/// Put the mesh buffers back so the next frame can reuse their allocation.
pub fn recycle(scene: Scene) {
    unsafe {
        let scratch = &mut *std::ptr::addr_of_mut!(SCRATCH);
        scratch.verts = scene.verts;
        scratch.indices = scene.indices;
    }
}

fn fx(v: i32) -> f32 {
    v as f32 / FRAC
}

fn ensure_atlas(scratch: &mut Scratch) -> Option<(u32, Vec<u8>, Vec<u8>)> {
    let key = unsafe { crate::doom::p_setup::vertexes as usize };
    let ready = scratch
        .atlas
        .as_ref()
        .is_some_and(|a| a.key == key && a.pending.is_none());
    if ready {
        return None;
    }
    if scratch.atlas.as_ref().is_some_and(|a| a.key == key) {
        return scratch.atlas.as_mut().and_then(|a| a.pending.take());
    }
    let mut built = unsafe { build_atlas(key) };
    let pending = built.pending.take();
    scratch.atlas = Some(built);
    pending
}

unsafe fn build_atlas(key: usize) -> Atlas {
    let pal = palette();
    let mut atlas = Atlas {
        key,
        tex_rects: vec![[0.0; 4]; numtextures.max(0) as usize],
        flat_rects: vec![[0.0; 4]; numflats.max(0) as usize],
        sprite_rects: vec![[0.0; 4]; numspritelumps.max(0) as usize],
        sprite_size: vec![(0, 0); numspritelumps.max(0) as usize],
        white: [0.0; 4],
        pending: None,
    };
    for size in [2048i32, 4096, 8192] {
        if pack_into(&mut atlas, size, &pal) {
            log::info!("texture atlas {size}x{size}");
            return atlas;
        }
    }
    log::warn!("texture atlas ran out of room; some surfaces will be missing");
    let _ = pack_into(&mut atlas, 8192, &pal);
    atlas
}

unsafe fn pack_into(atlas: &mut Atlas, size: i32, pal: &[(u8, u8, u8); 256]) -> bool {
    atlas.tex_rects.fill([0.0; 4]);
    atlas.flat_rects.fill([0.0; 4]);
    atlas.sprite_rects.fill([0.0; 4]);
    atlas.sprite_size.fill((0, 0));
    atlas.white = [0.0; 4];
    let mut pack = Packer::new(size);
    let mut ok = true;
    if let Some((wx, wy)) = pack.alloc(1, 1) {
        pack.put(wx, wy, [255, 255, 255, 255]);
        atlas.white = pack.rect(wx, wy, 1, 1);
    } else {
        ok = false;
    }
    for tex in 0..numtextures {
        let (w, h) = hw_texture_size(tex);
        if w <= 0 || h <= 0 {
            atlas.tex_rects[tex as usize] = [0.0; 4];
            continue;
        }
        let Some((x, y)) = pack.alloc(w, h) else {
            ok = false;
            break;
        };
        blit_wall(&mut pack, x, y, tex, w, h, pal);
        blit_wall_emission(&mut pack, x, y, tex, w, h);
        atlas.tex_rects[tex as usize] = pack.rect(x, y, w, h);
    }
    if ok {
        for flat in 0..numflats {
            let Some((x, y)) = pack.alloc(64, 64) else {
                ok = false;
                break;
            };
            blit_flat(&mut pack, x, y, flat, pal);
            blit_flat_emission(&mut pack, x, y, flat);
            atlas.flat_rects[flat as usize] = pack.rect(x, y, 64, 64);
        }
    }
    if ok {
        for lump in 0..numspritelumps {
            let Some((w, h, pixels)) = raster_patch(firstspritelump + lump, pal) else {
                atlas.sprite_rects[lump as usize] = [0.0; 4];
                atlas.sprite_size[lump as usize] = (0, 0);
                continue;
            };
            let Some((x, y)) = pack.alloc(w, h) else {
                ok = false;
                break;
            };
            pack.blit(x, y, w, h, &pixels);
            atlas.sprite_rects[lump as usize] = pack.rect(x, y, w, h);
            atlas.sprite_size[lump as usize] = (w, h);
        }
    }
    atlas.pending = Some((size as u32, pack.rgba, pack.emit));
    ok
}

unsafe fn palette() -> [(u8, u8, u8); 256] {
    let rgb = hw_palette_rgb();
    if rgb.iter().any(|c| (c.0 | c.1 | c.2) != 0) {
        return rgb;
    }
    let lump = W_CacheLumpNum(
        crate::doom::w_wad::W_GetNumForName(c"PLAYPAL".as_ptr()),
        PU_CACHE,
    ) as *const u8;
    let mut rgb = [(0u8, 0u8, 0u8); 256];
    if !lump.is_null() {
        for i in 0..256 {
            rgb[i] = (*lump.add(i * 3), *lump.add(i * 3 + 1), *lump.add(i * 3 + 2));
        }
    }
    rgb
}

struct Packer {
    size: i32,
    x: i32,
    y: i32,
    row_h: i32,
    rgba: Vec<u8>,
    /// One byte per texel. 0 is dark, 128 is half, 255 is a full brightmap.
    emit: Vec<u8>,
}

impl Packer {
    fn new(size: i32) -> Self {
        let n = (size as usize) * (size as usize);
        Self {
            size,
            x: 0,
            y: 0,
            row_h: 0,
            rgba: vec![0u8; n * 4],
            emit: vec![0u8; n],
        }
    }

    fn alloc(&mut self, w: i32, h: i32) -> Option<(i32, i32)> {
        if w > self.size || h > self.size {
            return None;
        }
        if self.x + w > self.size {
            self.y += self.row_h;
            self.x = 0;
            self.row_h = 0;
        }
        if self.y + h > self.size {
            return None;
        }
        let at = (self.x, self.y);
        self.x += w;
        self.row_h = self.row_h.max(h);
        Some(at)
    }

    fn rect(&self, x: i32, y: i32, w: i32, h: i32) -> [f32; 4] {
        let s = self.size as f32;
        let inset_x = 0.5 / s;
        let inset_y = 0.5 / s;
        [
            x as f32 / s + inset_x,
            y as f32 / s + inset_y,
            (x + w) as f32 / s - inset_x,
            (y + h) as f32 / s - inset_y,
        ]
    }

    fn put(&mut self, x: i32, y: i32, rgba: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.size || y >= self.size {
            return;
        }
        let dst = (y as usize * self.size as usize + x as usize) * 4;
        self.rgba[dst..dst + 4].copy_from_slice(&rgba);
    }

    fn put_emit(&mut self, x: i32, y: i32, level: u8) {
        if x < 0 || y < 0 || x >= self.size || y >= self.size || level == 0 {
            return;
        }
        let dst = y as usize * self.size as usize + x as usize;
        self.emit[dst] = level;
    }

    fn blit(&mut self, x: i32, y: i32, w: i32, h: i32, src: &[u8]) {
        for row in 0..h {
            let dst = ((y + row) as usize * self.size as usize + x as usize) * 4;
            let src_off = (row as usize) * (w as usize) * 4;
            let n = (w as usize) * 4;
            self.rgba[dst..dst + n].copy_from_slice(&src[src_off..src_off + n]);
        }
    }
}

unsafe fn blit_wall(
    pack: &mut Packer,
    x: i32,
    y: i32,
    tex: i32,
    w: i32,
    h: i32,
    pal: &[(u8, u8, u8); 256],
) {
    for col in 0..w {
        let src = R_GetColumn(tex, col);
        if src.is_null() {
            continue;
        }
        for row in 0..h {
            let idx = *src.add(row as usize) as usize;
            let (r, g, b) = pal[idx.min(255)];
            pack.put(x + col, y + row, [r, g, b, 255]);
        }
    }
}

unsafe fn blit_wall_emission(pack: &mut Packer, x: i32, y: i32, tex: i32, w: i32, h: i32) {
    let name = crate::doom::r_data::hw_texture_name(tex);
    if name[0] == 0 {
        return;
    }
    for row in 0..h {
        for col in 0..w {
            let level = crate::brightmaps::wall_level(&name, col, row, w, h);
            pack.put_emit(x + col, y + row, level);
        }
    }
}

unsafe fn blit_flat(pack: &mut Packer, x: i32, y: i32, flat: i32, pal: &[(u8, u8, u8); 256]) {
    let src = W_CacheLumpNum(firstflat + flat, PU_CACHE) as *const u8;
    if src.is_null() || W_LumpLength((firstflat + flat) as u32) < 4096 {
        return;
    }
    for row in 0..64 {
        for col in 0..64 {
            let idx = *src.add((row * 64 + col) as usize) as usize;
            let (r, g, b) = pal[idx.min(255)];
            pack.put(x + col, y + row, [r, g, b, 255]);
        }
    }
}

unsafe fn blit_flat_emission(pack: &mut Packer, x: i32, y: i32, flat: i32) {
    let name = flat_name(flat);
    if name[0] == 0 {
        return;
    }
    for row in 0..64 {
        for col in 0..64 {
            let level = crate::brightmaps::flat_level(&name, col, row);
            pack.put_emit(x + col, y + row, level);
        }
    }
}

unsafe fn flat_name(flat: i32) -> [u8; 8] {
    use crate::doom::w_wad::{lumpinfo, numlumps};
    let mut out = [0u8; 8];
    let lump = firstflat + flat;
    if lump < 0 || lumpinfo.is_null() || (lump as u32) >= numlumps {
        return out;
    }
    let name = &(*lumpinfo.add(lump as usize)).name;
    for i in 0..8 {
        let c = name[i] as u8;
        if c == 0 {
            break;
        }
        out[i] = c.to_ascii_uppercase();
    }
    out
}

/// Rasterise a Doom patch lump. Palette index 0 is transparent.
unsafe fn raster_patch(lump: i32, pal: &[(u8, u8, u8); 256]) -> Option<(i32, i32, Vec<u8>)> {
    if lump < 0 {
        return None;
    }
    let len = W_LumpLength(lump as u32);
    if len < 8 {
        return None;
    }
    let data = W_CacheLumpNum(lump, PU_CACHE) as *const u8;
    if data.is_null() {
        return None;
    }
    let bytes = std::slice::from_raw_parts(data, len as usize);
    let w = i16::from_le_bytes([bytes[0], bytes[1]]) as i32;
    let h = i16::from_le_bytes([bytes[2], bytes[3]]) as i32;
    if w <= 0 || h <= 0 || w > 2048 || h > 2048 {
        return None;
    }
    let header = 8usize;
    if bytes.len() < header + (w as usize) * 4 {
        return None;
    }
    let mut px = vec![0u8; (w as usize) * (h as usize) * 4];
    for col in 0..w as usize {
        let ofs = u32::from_le_bytes(
            bytes[header + col * 4..header + col * 4 + 4]
                .try_into()
                .ok()?,
        ) as usize;
        if ofs >= bytes.len() {
            continue;
        }
        let mut p = ofs;
        loop {
            if p >= bytes.len() {
                break;
            }
            let top = bytes[p];
            if top == 0xff {
                break;
            }
            if p + 1 >= bytes.len() {
                break;
            }
            let n = bytes[p + 1] as usize;
            let start = p + 3;
            if start + n > bytes.len() {
                break;
            }
            for i in 0..n {
                let row = top as usize + i;
                if row >= h as usize {
                    break;
                }
                let idx = bytes[start + i] as usize;
                let (r, g, b) = pal[idx.min(255)];
                // A posted pixel is opaque, including palette index 0 (black).
                // Transparency is the texels this loop never writes.
                let di = (row * w as usize + col) * 4;
                px[di..di + 4].copy_from_slice(&[r, g, b, 255]);
            }
            p = start + n + 1;
        }
    }
    Some((w, h, px))
}

unsafe fn push_sky(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    eye: [f32; 3],
    _ang: f32,
) {
    let tex = skytexture;
    let rect = tex_rect(atlas, tex);
    if rect == [0.0; 4] {
        return;
    }
    let (tw, th) = hw_texture_size(tex);
    if tw <= 0 || th <= 0 {
        return;
    }
    // A sphere around the eye, so looking straight up or down still hits sky.
    // Row 100 sits on the horizon (`skytexturemid`). The top of the texture
    // meets the pole. A full turn maps to 1024 sky columns.
    let radius = 6000.0;
    let repeats = 1024.0 / tw as f32;
    let sky_start = verts.len();
    let segs_n = 48i32;
    let lats = 16i32;
    let v_of = |el: f32| (100.0 - el.to_degrees() * (100.0 / 90.0)) / th as f32;
    for lat in 0..lats {
        let el0 =
            -std::f32::consts::FRAC_PI_2 + (lat as f32) / (lats as f32) * std::f32::consts::PI;
        let el1 = -std::f32::consts::FRAC_PI_2
            + ((lat + 1) as f32) / (lats as f32) * std::f32::consts::PI;
        for i in 0..segs_n {
            let a0 = (i as f32) / (segs_n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32) / (segs_n as f32) * std::f32::consts::TAU;
            let p = |a: f32, el: f32| {
                let c = el.cos() * radius;
                [
                    eye[0] + a.cos() * c,
                    eye[1] + el.sin() * radius,
                    eye[2] - a.sin() * c,
                ]
            };
            let u0 = (i as f32) / (segs_n as f32) * repeats;
            let u1 = ((i + 1) as f32) / (segs_n as f32) * repeats;
            push_quad(
                verts,
                indices,
                [p(a0, el0), p(a0, el1), p(a1, el1), p(a1, el0)],
                [
                    [u0, v_of(el0)],
                    [u0, v_of(el1)],
                    [u1, v_of(el1)],
                    [u1, v_of(el0)],
                ],
                rect,
                1.0,
                [1.0; 3],
            );
        }
    }
    // The path tracer treats this sphere as the sky light. `_pad` is otherwise 0.
    for v in verts.iter_mut().skip(sky_start) {
        v._pad = 1.0;
    }
}

unsafe fn push_world(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    flats: &[Vec<[f32; 2]>],
) {
    if segs.is_null() || subsectors.is_null() {
        return;
    }
    // One quad per solid surface. The partner seg of a two-sided line faces
    // the other way and would otherwise sit on the same plane.
    let mut drawn: HashSet<WallKey> = HashSet::new();
    for i in 0..numsegs {
        let seg = &*segs.add(i as usize);
        push_seg(verts, indices, atlas, seg, &mut drawn);
    }
    for i in 0..numsubsectors {
        let sub = &*subsectors.add(i as usize);
        let outline = flats.get(i as usize).map(Vec::as_slice).unwrap_or(&[]);
        push_flat(verts, indices, atlas, sub, outline);
    }
}

unsafe fn push_seg(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    seg: &seg_t,
    drawn: &mut HashSet<WallKey>,
) {
    if seg.v1.is_null()
        || seg.v2.is_null()
        || seg.linedef.is_null()
        || seg.sidedef.is_null()
        || seg.frontsector.is_null()
    {
        return;
    }
    let v1 = &*seg.v1;
    let v2 = &*seg.v2;
    let side = &*seg.sidedef;
    let line = &*seg.linedef;
    let front = &*seg.frontsector;
    let back = if seg.backsector.is_null() {
        None
    } else {
        Some(&*seg.backsector)
    };
    let light = wall_light(front.lightlevel, v1, v2);
    let flags = line.flags as u16;

    if back.is_none() {
        let z0 = fx(front.floorheight);
        let z1 = fx(front.ceilingheight);
        let tex = translated_tex(side.midtexture);
        let origin = if flags & LinedefFlag::DONTPEGBOTTOM != 0 {
            z0 + tex_pixels(tex)
        } else {
            z1
        } + fx(side.rowoffset);
        push_wall(
            verts, indices, atlas, v1, v2, z0, z1, side, seg, tex, origin, light, drawn,
        );
        return;
    }
    let back = back.unwrap();
    let sky = front.ceilingpic as i32 == skyflatnum && back.ceilingpic as i32 == skyflatnum;

    if !sky && back.ceilingheight < front.ceilingheight {
        let z0 = fx(back.ceilingheight);
        let z1 = fx(front.ceilingheight);
        let tex = translated_tex(side.toptexture);
        let origin = if flags & LinedefFlag::DONTPEGTOP != 0 {
            z1
        } else {
            z0 + tex_pixels(tex)
        } + fx(side.rowoffset);
        push_wall(
            verts, indices, atlas, v1, v2, z0, z1, side, seg, tex, origin, light, drawn,
        );
    }
    if back.floorheight > front.floorheight {
        let z0 = fx(front.floorheight);
        let z1 = fx(back.floorheight);
        let tex = translated_tex(side.bottomtexture);
        let origin = if flags & LinedefFlag::DONTPEGBOTTOM != 0 {
            fx(front.ceilingheight)
        } else {
            z1
        } + fx(side.rowoffset);
        push_wall(
            verts, indices, atlas, v1, v2, z0, z1, side, seg, tex, origin, light, drawn,
        );
    }
    if side.midtexture != 0 {
        let z0 = fx(front.floorheight.max(back.floorheight));
        let z1 = fx(front.ceilingheight.min(back.ceilingheight));
        let tex = translated_tex(side.midtexture);
        let origin = if flags & LinedefFlag::DONTPEGBOTTOM != 0 {
            z0 + tex_pixels(tex)
        } else {
            fx(front.ceilingheight.min(back.ceilingheight))
        } + fx(side.rowoffset);
        push_wall(
            verts, indices, atlas, v1, v2, z0, z1, side, seg, tex, origin, light, drawn,
        );
    }
}

/// True when the camera is on the front side of the directed seg.
///
/// The front side is to the right of v1→v2. Kept for the facing test; the
/// mesh itself keeps every side so lamps still cast after the camera turns.
#[cfg(test)]
fn seg_faces_view(x1: i32, y1: i32, x2: i32, y2: i32, cam_x: i32, cam_y: i32) -> bool {
    let dx = x2 as i64 - x1 as i64;
    let dy = y2 as i64 - y1 as i64;
    let side = dy * (cam_x as i64 - x1 as i64) + dx * (y1 as i64 - cam_y as i64);
    side >= 0
}

/// One quad per directed side. Opposite segs of a two-sided line do not share
/// a key: each texture has to stay in the scene so its brightmap can light
/// the sector it faces, even after the camera walks around the corner.
type WallKey = (i32, i32, i32, i32, i32, i32, i32);

fn wall_key(v1: &vertex_t, v2: &vertex_t, z0: f32, z1: f32) -> WallKey {
    let swapped = (v2.x, v2.y) < (v1.x, v1.y);
    let (ax, ay, bx, by) = if swapped {
        (v2.x, v2.y, v1.x, v1.y)
    } else {
        (v1.x, v1.y, v2.x, v2.y)
    };
    (
        ax,
        ay,
        bx,
        by,
        z0.to_bits() as i32,
        z1.to_bits() as i32,
        i32::from(swapped),
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn push_wall(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    v1: &vertex_t,
    v2: &vertex_t,
    z0: f32,
    z1: f32,
    side: &side_t,
    seg: &seg_t,
    tex: i32,
    origin: f32,
    light: f32,
    drawn: &mut HashSet<WallKey>,
) {
    if z1 <= z0 + 0.01 {
        return;
    }
    let rect = tex_rect(atlas, tex);
    if rect == [0.0; 4] {
        return;
    }
    let (tw, th) = hw_texture_size(tex);
    if tw <= 0 || th <= 0 {
        return;
    }
    if !drawn.insert(wall_key(v1, v2, z0, z1)) {
        return;
    }
    let x1 = fx(v1.x);
    let y1 = fx(v1.y);
    let x2 = fx(v2.x);
    let y2 = fx(v2.y);
    let len = (x2 - x1).hypot(y2 - y1);
    let u0 = (fx(side.textureoffset) + fx(seg.offset)) / tw as f32;
    let u1 = u0 + len / tw as f32;
    let v_of = |z: f32| (origin - z) / th as f32;
    push_quad(
        verts,
        indices,
        [[x1, z0, -y1], [x1, z1, -y1], [x2, z1, -y2], [x2, z0, -y2]],
        [
            [u0, v_of(z0)],
            [u0, v_of(z1)],
            [u1, v_of(z1)],
            [u1, v_of(z0)],
        ],
        rect,
        light,
        [1.0; 3],
    );
}

unsafe fn push_flat(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    sub: &subsector_t,
    outline: &[[f32; 2]],
) {
    if sub.sector.is_null() || outline.len() < 3 {
        return;
    }
    let sector = &*(sub.sector as *const sector_t);
    let light = (sector.lightlevel as f32 / 255.0).clamp(0.04, 1.0);
    let y_floor = fx(sector.floorheight);
    let grown = inflate_poly(outline, 3.0);
    let floor_rect = flat_rect(atlas, sector.floorpic as i32);
    if floor_rect != [0.0; 4] {
        fan(verts, indices, &grown, y_floor, floor_rect, light);
    }
    if sector.ceilingpic as i32 != skyflatnum {
        let ceil_rect = flat_rect(atlas, sector.ceilingpic as i32);
        if ceil_rect != [0.0; 4] {
            fan(
                verts,
                indices,
                &grown,
                fx(sector.ceilingheight),
                ceil_rect,
                light,
            );
        }
    }
}

/// Push every edge outward so neighboring floors overlap. A shared BSP edge
/// is one line, and a ray can slip between the two triangles there.
fn inflate_poly(poly: &[[f32; 2]], amount: f32) -> Vec<[f32; 2]> {
    let n = poly.len();
    if n < 3 || amount <= 0.0 {
        return poly.to_vec();
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    let side = if area >= 0.0 { 1.0 } else { -1.0 };
    let mut lines = Vec::with_capacity(n);
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = dx.hypot(dy).max(1e-6);
        lines.push((
            a[0] + side * dy / len * amount,
            a[1] - side * dx / len * amount,
            dx,
            dy,
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let (x1, y1, dx1, dy1) = lines[prev];
        let (x2, y2, dx2, dy2) = lines[i];
        let det = dx1 * dy2 - dy1 * dx2;
        let cur = poly[i];
        if det.abs() < 1e-6 {
            out.push([x2, y2]);
            continue;
        }
        let t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / det;
        let p = [x1 + dx1 * t, y1 + dy1 * t];
        let drift = (p[0] - cur[0]).hypot(p[1] - cur[1]);
        if !drift.is_finite() || drift > amount * 4.0 {
            let e0x = cur[0] - poly[prev][0];
            let e0y = cur[1] - poly[prev][1];
            let next = poly[(i + 1) % n];
            let e1x = next[0] - cur[0];
            let e1y = next[1] - cur[1];
            let l0 = e0x.hypot(e0y).max(1e-6);
            let l1 = e1x.hypot(e1y).max(1e-6);
            let ox = side * (e0y / l0 + e1y / l1);
            let oy = -side * (e0x / l0 + e1x / l1);
            let ol = ox.hypot(oy).max(1e-6);
            out.push([cur[0] + ox / ol * amount, cur[1] + oy / ol * amount]);
        } else {
            out.push(p);
        }
    }
    out
}

fn fan(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    poly: &[[f32; 2]],
    y: f32,
    rect: [f32; 4],
    light: f32,
) {
    if poly.len() < 3 || verts.len() + poly.len() > 2_000_000 {
        return;
    }
    let base = verts.len() as u32;
    for p in poly {
        verts.push(Vertex {
            pos: [p[0], y, -p[1]],
            uv: [p[0] / 64.0, p[1] / 64.0],
            atlas: rect,
            light,
            tint: [1.0; 3],
            _pad: 0.0,
        });
    }
    for i in 1..poly.len() as u32 - 1 {
        indices.extend_from_slice(&[base, base + i, base + i + 1]);
    }
}

const NF_SUBSECTOR: u32 = 0x8000;

struct Plane {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    keep_right: bool,
}

fn ensure_flats(scratch: &mut Scratch) {
    let key = unsafe { vertexes as usize };
    if scratch.flats_key == key && scratch.flats.len() == unsafe { numsubsectors.max(0) as usize } {
        return;
    }
    scratch.flats = unsafe { build_flat_polys() };
    scratch.flats_key = key;
}

unsafe fn build_flat_polys() -> Vec<Vec<[f32; 2]>> {
    let n = numsubsectors.max(0) as usize;
    let mut out = vec![Vec::new(); n];
    if n == 0 || vertexes.is_null() || numvertexes < 1 {
        return out;
    }
    let mut minx = f32::MAX;
    let mut miny = f32::MAX;
    let mut maxx = f32::MIN;
    let mut maxy = f32::MIN;
    for i in 0..numvertexes as usize {
        let v = &*vertexes.add(i);
        let x = fx(v.x);
        let y = fx(v.y);
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    minx -= 64.0;
    miny -= 64.0;
    maxx += 64.0;
    maxy += 64.0;
    if numnodes <= 0 || nodes.is_null() {
        if n == 1 {
            out[0] = vec![[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]];
        }
        return out;
    }
    let mut cons = Vec::new();
    walk_flats(numnodes - 1, &mut cons, &mut out, minx, miny, maxx, maxy);
    out
}

unsafe fn walk_flats(
    nodenum: i32,
    cons: &mut Vec<Plane>,
    out: &mut [Vec<[f32; 2]>],
    minx: f32,
    miny: f32,
    maxx: f32,
    maxy: f32,
) {
    if nodenum as u32 & NF_SUBSECTOR != 0 {
        let ss = (nodenum as u32 & !NF_SUBSECTOR) as usize;
        if ss < out.len() {
            let mut poly = vec![[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]];
            for plane in cons.iter() {
                poly = clip_poly(&poly, plane);
                if poly.len() < 3 {
                    return;
                }
            }
            out[ss] = poly;
        }
        return;
    }
    if nodenum < 0 || nodenum >= numnodes {
        return;
    }
    let node = &*nodes.add(nodenum as usize);
    let x = fx(node.x);
    let y = fx(node.y);
    let dx = fx(node.dx);
    let dy = fx(node.dy);
    for (child, keep_right) in [(0, true), (1, false)] {
        cons.push(Plane {
            x,
            y,
            dx,
            dy,
            keep_right,
        });
        walk_flats(
            node.children[child] as i32,
            cons,
            out,
            minx,
            miny,
            maxx,
            maxy,
        );
        cons.pop();
    }
}

fn clip_poly(poly: &[[f32; 2]], plane: &Plane) -> Vec<[f32; 2]> {
    if poly.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 1);
    let mut prev = poly[poly.len() - 1];
    let mut prev_in = plane_inside(plane, prev);
    for &cur in poly {
        let cur_in = plane_inside(plane, cur);
        if cur_in {
            if !prev_in {
                out.push(plane_hit(plane, prev, cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(plane_hit(plane, prev, cur));
        }
        prev = cur;
        prev_in = cur_in;
    }
    dedup_poly(&mut out);
    out
}

fn plane_dist(plane: &Plane, p: [f32; 2]) -> f32 {
    let len = plane.dx.hypot(plane.dy).max(1e-6);
    (plane.dx * (p[1] - plane.y) - plane.dy * (p[0] - plane.x)) / len
}

fn plane_inside(plane: &Plane, p: [f32; 2]) -> bool {
    // Child 0 is the right half-space. The boundary belongs to both sides so
    // the shared edge is the same line, without an overlapping strip that
    // shimmers.
    let d = plane_dist(plane, p);
    if plane.keep_right {
        d <= 1.0e-3
    } else {
        d >= -1.0e-3
    }
}

fn plane_hit(plane: &Plane, a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let ca = plane.dx * (a[1] - plane.y) - plane.dy * (a[0] - plane.x);
    let cb = plane.dx * (b[1] - plane.y) - plane.dy * (b[0] - plane.x);
    let denom = ca - cb;
    if denom.abs() < 1e-8 {
        return a;
    }
    let u = ca / denom;
    [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])]
}

fn dedup_poly(poly: &mut Vec<[f32; 2]>) {
    poly.retain(|p| p[0].is_finite() && p[1].is_finite());
    if poly.len() < 2 {
        return;
    }
    let mut i = 0;
    while i < poly.len() {
        let j = (i + 1) % poly.len();
        let d = (poly[i][0] - poly[j][0]).hypot(poly[i][1] - poly[j][1]);
        if d < 0.05 && poly.len() > 3 {
            poly.remove(j);
            if j == 0 {
                break;
            }
        } else {
            i += 1;
        }
    }
}

unsafe fn push_sprites(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    right: [f32; 3],
) {
    if sectors.is_null()
        || spritewidth.is_null()
        || spriteoffset.is_null()
        || spritetopoffset.is_null()
    {
        return;
    }
    if consoleplayer < 0 || consoleplayer as usize >= MAXPLAYERS {
        return;
    }
    let player = std::ptr::addr_of_mut!(players[consoleplayer as usize]) as *mut std::ffi::c_void;
    for i in 0..numsectors.max(0) as usize {
        let sector = &*sectors.add(i);
        let mut thing = sector.thinglist as *mut mobj_t;
        let mut guard = 0;
        while !thing.is_null() && guard < 4096 {
            guard += 1;
            let th = &*thing;
            thing = th.snext;
            if !th.player.is_null() && th.player as *mut std::ffi::c_void == player {
                continue;
            }
            let mut light = (sector.lightlevel as f32 / 255.0).clamp(0.04, 1.0);
            if th.frame & FF_FULLBRIGHT != 0 {
                light = 1.0;
            } else if th.flags & MF_SHADOW != 0 {
                light = 0.18;
            }
            if let Some((mesh, extra)) = voxel::model_for(th.sprite, th.frame, leveltime) {
                push_voxel(verts, indices, atlas, th, &mesh, extra, light);
                continue;
            }
            let Some((lump, flip)) = hw_sprite_lump(th.sprite, th.frame, th.x, th.y, th.angle)
            else {
                continue;
            };
            push_billboard(verts, indices, atlas, th, lump, flip, right, light);
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn push_billboard(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    th: &mobj_t,
    lump: i32,
    flip: bool,
    right: [f32; 3],
    light: f32,
) {
    if lump < 0 || lump as usize >= atlas.sprite_rects.len() {
        return;
    }
    let rect = atlas.sprite_rects[lump as usize];
    let (pw, ph) = atlas.sprite_size[lump as usize];
    if rect == [0.0; 4] || pw <= 0 || ph <= 0 {
        return;
    }
    let width = fx(*spritewidth.add(lump as usize)).max(1.0);
    let offset = fx(*spriteoffset.add(lump as usize));
    let top = fx(th.z) + fx(*spritetopoffset.add(lump as usize));
    let cx = fx(th.x);
    let cz = -fx(th.y);
    let left = [cx - right[0] * offset, top, cz - right[2] * offset];
    let across = [right[0] * width, 0.0, right[2] * width];
    let down = ph as f32;
    let (u0, u1) = if flip { (0.999, 0.0) } else { (0.0, 0.999) };
    push_quad(
        verts,
        indices,
        [
            left,
            [left[0] + across[0], left[1], left[2] + across[2]],
            [left[0] + across[0], left[1] - down, left[2] + across[2]],
            [left[0], left[1] - down, left[2]],
        ],
        [[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]],
        rect,
        light,
        [1.0; 3],
    );
}

unsafe fn push_weapon(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    eye: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    cam_up: [f32; 3],
) {
    if consoleplayer < 0
        || consoleplayer as usize >= MAXPLAYERS
        || spriteoffset.is_null()
        || spritetopoffset.is_null()
    {
        return;
    }
    let player = &players[consoleplayer as usize];
    let distance = 48.0;
    let world_per_px = distance / 160.0;
    for slot in 0..NUMPSPRITES {
        let psp: &PspdefT = &player.psprites[slot];
        if psp.state.is_null() {
            continue;
        }
        let state = &*(psp.state as *const State);
        let Some((lump, flip)) = hw_psprite_lump(state.sprite, state.frame) else {
            continue;
        };
        if lump < 0 || lump as usize >= atlas.sprite_size.len() {
            continue;
        }
        let rect = atlas.sprite_rects[lump as usize];
        let (pw, ph) = atlas.sprite_size[lump as usize];
        if rect == [0.0; 4] || pw <= 0 || ph <= 0 {
            continue;
        }
        let offset = fx(*spriteoffset.add(lump as usize));
        let topoff = fx(*spritetopoffset.add(lump as usize));
        let sx = fx(psp.sx);
        let sy = fx(psp.sy);
        let left_px = sx - offset - 160.0;
        let top_px = sy - topoff - 0.5;
        let top_up = 100.0 - top_px;
        let yoff = top_up * world_per_px;
        let mut origin = eye;
        for k in 0..3 {
            origin[k] +=
                forward[k] * distance + right[k] * (left_px * world_per_px) + cam_up[k] * yoff;
        }
        let width = pw as f32 * world_per_px;
        let height = ph as f32 * world_per_px;
        let right_w = [right[0] * width, right[1] * width, right[2] * width];
        let down = [
            -cam_up[0] * height,
            -cam_up[1] * height,
            -cam_up[2] * height,
        ];
        let (u0, u1) = if flip { (0.999, 0.0) } else { (0.0, 0.999) };
        let corner = |a: [f32; 3], b: [f32; 3]| {
            [
                origin[0] + a[0] + b[0],
                origin[1] + a[1] + b[1],
                origin[2] + a[2] + b[2],
            ]
        };
        push_quad(
            verts,
            indices,
            [
                origin,
                corner(right_w, [0.0; 3]),
                corner(right_w, down),
                corner([0.0; 3], down),
            ],
            [[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]],
            rect,
            1.0,
            [1.0; 3],
        );
    }
}

fn push_voxel(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    atlas: &Atlas,
    th: &mobj_t,
    mesh: &Arc<VoxelMesh>,
    extra_yaw: f32,
    light: f32,
) {
    let yaw = (th.angle as f32) * std::f32::consts::TAU / 4294967296.0 + extra_yaw;
    let (s, c) = yaw.sin_cos();
    let ox = fx(th.x);
    let oy = fx(th.z);
    let oz = -fx(th.y);
    let uv = [[0.5, 0.5]; 4];
    for quad in &mesh.quads {
        let mut p = [[0.0; 3]; 4];
        for i in 0..4 {
            let m = quad.p[i];
            p[i] = [
                ox + c * m[1] + s * m[0],
                oy + m[2],
                oz - s * m[1] + c * m[0],
            ];
        }
        let rgb = mesh.palette[quad.color as usize];
        push_quad(
            verts,
            indices,
            p,
            uv,
            atlas.white,
            light,
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ],
        );
    }
}

fn push_quad(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    p: [[f32; 3]; 4],
    uv: [[f32; 2]; 4],
    atlas: [f32; 4],
    light: f32,
    tint: [f32; 3],
) {
    if verts.len() + 4 > 2_000_000 {
        return;
    }
    let base = verts.len() as u32;
    for i in 0..4 {
        verts.push(Vertex {
            pos: p[i],
            uv: uv[i],
            atlas,
            light,
            tint,
            _pad: 0.0,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn tex_rect(atlas: &Atlas, tex: i32) -> [f32; 4] {
    if tex < 0 || tex as usize >= atlas.tex_rects.len() {
        [0.0; 4]
    } else {
        atlas.tex_rects[tex as usize]
    }
}

fn flat_rect(atlas: &Atlas, flat: i32) -> [f32; 4] {
    let flat = unsafe {
        if flattranslation.is_null() || flat < 0 || flat >= numflats {
            -1
        } else {
            *flattranslation.add(flat as usize)
        }
    };
    if flat < 0 || flat as usize >= atlas.flat_rects.len() {
        [0.0; 4]
    } else {
        atlas.flat_rects[flat as usize]
    }
}

unsafe fn translated_tex(raw: i16) -> i32 {
    let raw = raw as i32;
    if raw <= 0 || texturetranslation.is_null() || raw >= numtextures {
        return raw;
    }
    *texturetranslation.add(raw as usize)
}

unsafe fn tex_pixels(tex: i32) -> f32 {
    if tex < 0 || textureheight.is_null() || tex >= numtextures {
        return 0.0;
    }
    (*textureheight.add(tex as usize) >> 16) as f32
}

fn wall_light(level: i16, v1: &vertex_t, v2: &vertex_t) -> f32 {
    let mut light = level as f32 / 255.0;
    if v1.y == v2.y {
        light -= 0.1;
    } else if v1.x == v2.x {
        light += 0.1;
    }
    light.clamp(0.04, 1.0)
}

fn perspective(aspect: f32) -> [[f32; 4]; 4] {
    let near = 1.0;
    let far = 16000.0;
    let f = aspect;
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * nf, -1.0],
        [0.0, 0.0, far * near * nf, 0.0],
    ]
}

fn look_at(eye: [f32; 3], forward: [f32; 3]) -> [[f32; 4]; 4] {
    let f = norm(forward);
    let up = [0.0, 1.0, 0.0];
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut o = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            o[c][r] = a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    o
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let n = dot(v, v).sqrt().max(1e-8);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[cfg(test)]
mod tests {
    use super::seg_faces_view;

    #[test]
    fn wall_is_drawn_from_the_front_only() {
        let x1 = 0;
        let y1 = 0;
        let x2 = 64 << 16;
        let y2 = 0;
        let south = -100 << 16;
        let north = 100 << 16;
        assert!(seg_faces_view(x1, y1, x2, y2, 0, south));
        assert!(!seg_faces_view(x1, y1, x2, y2, 0, north));
        assert!(!seg_faces_view(x2, y2, x1, y1, 0, south));
        assert!(seg_faces_view(x2, y2, x1, y1, 0, north));
    }

    #[test]
    fn inflated_square_overlaps_its_neighbors() {
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let grown = super::inflate_poly(&square, 1.0);
        assert_eq!(grown.len(), 4);
        assert!(grown.iter().any(|p| p[0] < -0.5 && p[1] < -0.5));
        assert!(grown.iter().any(|p| p[0] > 1.5 && p[1] > 1.5));
    }
}
