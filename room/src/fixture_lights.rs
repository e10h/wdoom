//! Dedicated stationary-fixture lights. Bake traces the empty map as before;
//! these compact records add illumination without any fixture geometry.
use crate::doom::info::*;
use bytemuck::{Pod, Zeroable};

pub const MAX_FIXTURE_LIGHTS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FixtureLight {
    /// World position and soft-source radius, in Doom map units.
    pub position_radius: [f32; 4],
    /// Linear RGB and intensity (roughly unit incoming light at 64 units).
    pub color_power: [f32; 4],
    /// Area sources reuse existing ceiling triangles and atlas masks.
    pub triangle_c: [f32; 4],
    pub atlas: [f32; 4],
    pub uv_ab: [f32; 4],
    pub uv_c: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FixtureLights {
    pub meta: [u32; 4],
    pub lights: [FixtureLight; MAX_FIXTURE_LIGHTS],
}
impl Default for FixtureLights {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// RGB, intensity, radius, and distance down from the sprite's top.
fn profile(sprite: i32) -> Option<([f32; 3], f32, f32, f32)> {
    Some(match sprite {
        SPR_TRED => ([1.0, 0.16, 0.035], 4096.0, 6.0, 18.0),
        SPR_TGRN => ([0.12, 1.0, 0.16], 4096.0, 6.0, 18.0),
        SPR_TBLU => ([0.10, 0.28, 1.0], 4096.0, 6.0, 18.0),
        SPR_SMRT => ([1.0, 0.16, 0.035], 3072.0, 4.0, 12.0),
        SPR_SMGT => ([0.12, 1.0, 0.16], 3072.0, 4.0, 12.0),
        SPR_SMBT => ([0.10, 0.28, 1.0], 3072.0, 4.0, 12.0),
        SPR_CAND => ([1.0, 0.58, 0.22], 512.0, 1.0, 3.0),
        SPR_CBRA => ([1.0, 0.58, 0.22], 1536.0, 3.0, 5.0),
        SPR_COLU => ([1.0, 0.78, 0.30], 1024.0, 1.5, 5.0),
        SPR_TLMP => ([0.82, 0.92, 1.0], 1536.0, 3.0, 12.0),
        SPR_TLP2 => ([0.82, 0.92, 1.0], 1024.0, 2.5, 10.0),
        SPR_FCAN => ([1.0, 0.42, 0.08], 3072.0, 6.0, 10.0),
        _ => return None,
    })
}

/// Snapshot light fixtures on the game/render thread. Frame A makes source
/// height and intensity stable while the fixture's visible flame animates.
/// # Safety
/// Doom's sector lists and sprite tables must not be mutated concurrently.
pub unsafe fn collect() -> FixtureLights {
    use crate::doom::p_setup::{numsectors, sectors};
    use crate::doom::p_telept::mobj_t;
    use crate::doom::r_data::spritetopoffset;
    use crate::doom::r_things::hw_sprite_lump;
    let mut result = FixtureLights::default();
    if sectors.is_null() || spritetopoffset.is_null() {
        return result;
    }
    for i in 0..numsectors.max(0) as usize {
        let mut thing = (*sectors.add(i)).thinglist as *mut mobj_t;
        let mut guard = 0;
        while !thing.is_null() && guard < 4096 {
            let th = &*thing;
            thing = th.snext;
            guard += 1;
            let Some((color, power, radius, depth)) = profile(th.sprite) else {
                continue;
            };
            let Some((lump, _)) = hw_sprite_lump(th.sprite, 0, th.x, th.y, th.angle) else {
                continue;
            };
            let index = result.meta[0] as usize;
            if index == MAX_FIXTURE_LIGHTS {
                return result;
            }
            let top = crate::voxel::model_top(th.sprite, 0)
                .unwrap_or(*spritetopoffset.add(lump as usize) as f32 / 65536.0);
            result.lights[index] = FixtureLight {
                position_radius: [
                    th.x as f32 / 65536.0,
                    th.z as f32 / 65536.0 + (top - depth).max(radius),
                    -th.y as f32 / 65536.0,
                    radius,
                ],
                color_power: [color[0], color[1], color[2], power],
                ..Default::default()
            };
            result.meta[0] += 1;
            result.meta[2] += 1;
        }
    }
    result
}

/// Ceiling-panel materials. Their original mask still selects the lit pixels.
pub(crate) fn is_ceiling_panel(name: &[u8; 8]) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<[u8; 8], bool>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    *cache.entry(*name).or_insert_with(|| {
        (0..64).any(|y| (0..64).any(|x| crate::brightmaps::flat_level(name, x, y) > 0))
    })
}

/// Add direct area sampling for existing ceiling geometry, with no new mesh.
/// If a map exceeds the source budget, rotate a normalized subset each frame.
/// Every ceiling remains represented over time; lamps keep their own samples.
/// # Safety
/// Same game-thread requirements as `collect`.
pub unsafe fn collect_for_scene(scene: &crate::hw::Scene) -> FixtureLights {
    use std::sync::atomic::{AtomicU32, Ordering};
    static PHASE: AtomicU32 = AtomicU32::new(1);
    let mut phase = PHASE.fetch_add(1, Ordering::Relaxed);
    phase ^= phase >> 16;
    phase = phase.wrapping_mul(0x7feb352d);
    phase ^= phase >> 15;
    let mut result = collect();
    append_panels(
        &mut result,
        &scene.verts,
        &scene.indices[..scene.opaque_indices as usize],
        phase,
    );
    result
}

fn append_panels(
    result: &mut FixtureLights,
    verts: &[crate::hw::Vertex],
    triangles: &[u32],
    phase: u32,
) {
    let panels = triangles
        .chunks_exact(3)
        .filter(|tri| verts[tri[0] as usize]._pad == -0.25);
    let total = panels.clone().count();
    let budget = (MAX_FIXTURE_LIGHTS - result.meta[0] as usize).min(total);
    if budget == 0 {
        return;
    }
    let start = phase as usize % total;
    for (number, tri) in panels.enumerate() {
        if (number + total - start) % total >= budget {
            continue;
        }
        let [a, b, c] = [0, 1, 2].map(|i| verts[tri[i] as usize]);
        let ab = [0, 1, 2].map(|i| b.pos[i] - a.pos[i]);
        let ac = [0, 1, 2].map(|i| c.pos[i] - a.pos[i]);
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let area = cross.iter().map(|v| v * v).sum::<f32>().sqrt() * 0.5;
        result.lights[result.meta[0] as usize] = FixtureLight {
            position_radius: [a.pos[0], a.pos[1], a.pos[2], -1.0],
            color_power: [
                b.pos[0],
                b.pos[1],
                b.pos[2],
                area * total as f32 / budget as f32,
            ],
            triangle_c: [c.pos[0], c.pos[1], c.pos[2], 0.0],
            atlas: a.atlas,
            uv_ab: [a.uv[0], a.uv[1], b.uv[0], b.uv[1]],
            uv_c: [c.uv[0], c.uv[1], 0.0, 0.0],
        };
        result.meta[0] += 1;
    }
    // All marked panels are now covered. Disable their random-hit emission
    // term in the shader so direct sampling doesn't double their energy.
    result.meta[1] = 1;
    result.meta[3] = total as u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ceiling_sources_over_budget_rotate_without_losing_energy() {
        use crate::hw::Vertex;
        let mut verts = Vec::new();
        for i in 0..300 {
            for pos in [
                [i as f32, 0.0, 0.0],
                [i as f32 + 1.0, 0.0, 0.0],
                [i as f32, 0.0, 1.0],
            ] {
                verts.push(Vertex {
                    pos,
                    _pad: -0.25,
                    ..Zeroable::zeroed()
                });
            }
        }
        let indices: Vec<u32> = (0..verts.len() as u32).collect();
        let mut seen = [false; 300];
        for phase in [0, 128, 256] {
            let mut sources = FixtureLights::default();
            append_panels(&mut sources, &verts, &indices, phase);
            assert_eq!(sources.meta, [128, 1, 0, 300]);
            let energy: f32 = sources.lights.iter().map(|l| l.color_power[3]).sum();
            assert!(
                (energy - 150.0).abs() < 0.001,
                "preserve the total emitting area"
            );
            for light in sources.lights {
                seen[light.position_radius[0] as usize] = true;
            }
        }
        assert!(
            seen.into_iter().all(|v| v),
            "no ceiling may be permanently dropped"
        );
    }

    #[test]
    fn lamp_pool_survives_the_default_floor_without_clipping() {
        let (_, power, _, drop) = profile(SPR_COLU).unwrap();
        let height = crate::voxel::model_top(SPR_COLU, 0).unwrap() - drop;
        assert_eq!(height, 43.0, "keep the source inside the actual bulb");
        let incoming_below = power / (height * height);
        assert!(
            incoming_below > 0.4 && incoming_below < 1.0,
            "the nearby pool must exceed the minimum light without blowing out: {incoming_below}"
        );
        assert!(
            is_ceiling_panel(b"CEIL1_2\0"),
            "include authored lamps beyond TLITE"
        );
        assert!(
            !is_ceiling_panel(b"CEIL5_1\0"),
            "ordinary ceiling stays non-emissive"
        );
    }

    #[test]
    fn only_stationary_fixtures_get_sources() {
        assert!(profile(SPR_TRED).is_some());
        assert!(profile(SPR_TLMP).is_some());
        assert!(profile(SPR_CAND).is_some());
        assert!(profile(SPR_POSS).is_none());
        assert!(profile(SPR_BON1).is_none());
        assert_eq!(std::mem::size_of::<FixtureLight>(), 96);
        assert_eq!(std::mem::size_of::<FixtureLights>(), 16 + 128 * 96);
    }
}
