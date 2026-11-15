//! Real-map regression for light-map identity across changing door geometry.
use std::collections::HashMap;
use std::ffi::CString;

fn wall_samples(scene: &room::hw::Scene) -> HashMap<[u32; 9], [u32; 2]> {
    scene.indices[..scene.world_indices as usize]
        .iter()
        .filter_map(|&index| {
            let v = scene.verts[index as usize];
            if v._pad > -0.5 {
                return None;
            }
            let key = [
                v.pos[0].to_bits(),
                v.pos[1].to_bits(),
                v.pos[2].to_bits(),
                v.uv[0].to_bits(),
                v.uv[1].to_bits(),
                v.atlas[0].to_bits(),
                v.atlas[1].to_bits(),
                v.atlas[2].to_bits(),
                v.atlas[3].to_bits(),
            ];
            Some((key, v.light_uv.map(f32::to_bits)))
        })
        .collect()
}

#[test]
fn opening_a_real_e1m1_door_keeps_other_walls_on_their_saved_texels() {
    let wad = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../doom1.wad");
    let args: Vec<CString> = [
        "room",
        "-iwad",
        wad.to_str().unwrap(),
        "-warp",
        "1",
        "1",
        "-nomonsters",
        "-nomusic",
        "-nosound",
        "-nomouse",
        "-nojoy",
        "-config",
        "/tmp/wdoom-lightmap-identity.cfg",
    ]
    .iter()
    .map(|s| CString::new(*s).unwrap())
    .collect();
    let mut argv: Vec<_> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
    argv.push(std::ptr::null_mut());
    room::platform::init_start_time();
    unsafe {
        room::doom::d_loop::singletics = 1;
        room::doom::doomgeneric::doomgeneric_Create(args.len() as i32, argv.as_mut_ptr());
        let before = room::hw::build_scene(false);
        assert!(before.light_reset);
        let key = before.light_bake;
        let walls = wall_samples(&before);
        assert!(walls.len() > 200, "load the actual E1M1 wall geometry");
        room::hw::recycle(before);
        let camera = (
            room::doom::r_main::viewx,
            room::doom::r_main::viewy,
            room::doom::r_main::viewangle,
        );
        for turn in [0, 0x4000_0000, 0x8000_0000] {
            room::doom::r_main::viewx = camera.0 + (128 << 16);
            room::doom::r_main::viewy = camera.1 - (256 << 16);
            room::doom::r_main::viewangle = turn;
            let scene = room::hw::build_scene(false);
            assert_eq!(
                wall_samples(&scene),
                walls,
                "walking/turning must not change wall lighting identity"
            );
            room::hw::recycle(scene);
        }
        room::doom::r_main::viewx = camera.0;
        room::doom::r_main::viewy = camera.1;
        room::doom::r_main::viewangle = camera.2;
        let doors: Vec<_> = (0..room::doom::p_setup::numsectors as usize)
            .filter_map(|index| {
                let sector = room::doom::p_setup::sectors.add(index);
                ((*sector).ceilingheight == (*sector).floorheight).then_some(sector)
            })
            .collect();
        assert!(!doors.is_empty(), "E1M1 has closed doors");
        for door in doors {
            let closed = (*door).ceilingheight;
            for open in [64, 128, 0] {
                (*door).ceilingheight = closed + (open << 16);
                let after = room::hw::build_scene(false);
                assert!(!after.light_reset, "a door must never reset the map");
                assert_eq!(
                    after.light_bake, key,
                    "a door must not change the saved bake identity"
                );
                let changed = wall_samples(&after);
                assert!(
                    changed.values().all(|uv| uv.iter().all(|v| {
                        let value = f32::from_bits(*v);
                        (0.0..1.0).contains(&value)
                    })),
                    "newly exposed door faces must have allocated light texels"
                );
                let mut matched = 0;
                for (wall, light_uv) in &walls {
                    if let Some(actual) = changed.get(wall) {
                        assert_eq!(
                            actual, light_uv,
                            "unchanged wall moved to another light patch at door opening {open}"
                        );
                        matched += 1;
                    }
                }
                assert!(matched > 200, "compare real unchanged walls around the map");
                room::hw::recycle(after);
            }
            (*door).ceilingheight = closed;
        }
    }
}
