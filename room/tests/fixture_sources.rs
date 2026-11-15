//! Dedicated fixture records must not increase Bake's mesh or change its key.
use std::ffi::CString;

#[test]
fn torch_adds_a_light_record_and_zero_bake_triangles() {
    let wad = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../doom1.wad");
    let args: Vec<_> = [
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
        "/tmp/wdoom-fixture-sources.cfg",
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
        let baseline = room::fixture_lights::collect();
        let panel_sources = room::fixture_lights::collect_for_scene(&before);
        assert_eq!(panel_sources.meta[1], 1, "all ceiling panels must fit the direct-sampling budget");
        assert!(panel_sources.meta[0] > baseline.meta[0], "discover the ceiling panel triangles");
        eprintln!("{} point fixtures, {} ceiling panel triangles", baseline.meta[0], panel_sources.meta[0] - baseline.meta[0]);
        for lamp in baseline.lights[..baseline.meta[0] as usize].iter().filter(|l| l.color_power[3] == 1024.0) {
            assert!(lamp.color_power[0] > lamp.color_power[1], "COLU bulb is warm yellow, not green");
            assert!(lamp.position_radius[3] <= 2.0, "source stays inside the narrow bulb band");
        }

        assert!(baseline.meta[0] > 0, "E1M1's lamps must be discovered");
        let counts = (
            before.verts.len(),
            before.opaque_indices,
            before.indices.len(),
        );
        let key = before.light_bake;
        room::hw::recycle(before);
        let x = room::doom::r_main::viewx;
        let y = room::doom::r_main::viewy;
        let type_ = room::doom::info::mobjinfo
            .iter()
            .position(|i| i.doomednum == 46)
            .unwrap();
        let torch = room::doom::p_mobj::P_SpawnMobj(x, y, i32::MIN, type_ as i32);
        assert!(!torch.is_null());
        let lights = room::fixture_lights::collect();
        assert_eq!(lights.meta[0], baseline.meta[0] + 1);
        let light = lights.lights[..lights.meta[0] as usize]
            .iter()
            .find(|light| {
                light.position_radius[0] == x as f32 / 65536.0
                    && light.position_radius[2] == -y as f32 / 65536.0
            })
            .expect("spawned torch's source");
        assert!(
            light.position_radius[1] > (*torch).z as f32 / 65536.0 + 60.0,
            "place the source in the flame rather than at the feet"
        );
        assert!(light.color_power[0] > light.color_power[1] * 4.0);
        // Animation must not make a static source move or flicker during baking.
        (*torch).frame = 3 | 0x8000;
        assert_eq!(
            bytemuck::bytes_of(&lights),
            bytemuck::bytes_of(&room::fixture_lights::collect())
        );
        let after = room::hw::build_scene(false);
        assert_eq!(
            (after.verts.len(), after.opaque_indices, after.indices.len()),
            counts,
            "fixtures must add ZERO bake geometry"
        );
        assert_eq!(
            after.light_bake, key,
            "keep the user's native bake identity"
        );
        eprintln!(
            "{} dedicated fixtures; {} bake triangles, unchanged after adding a torch",
            lights.meta[0],
            after.opaque_indices / 3
        );
        room::hw::recycle(after);
    }
}
