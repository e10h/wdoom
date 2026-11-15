//! Audit hardware wall pixels against independent WAD patch composition.
use std::collections::HashMap;
fn word(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap()) as usize
}
fn signed(b: &[u8], at: usize) -> i32 {
    word(b, at) as i16 as i32
}
fn long(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}
fn name(b: &[u8]) -> String {
    String::from_utf8_lossy(b)
        .trim_end_matches('\0')
        .to_ascii_uppercase()
}
#[test]
fn all_wall_images_match_patch_posts_and_offsets() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../doom1.wad");
    let wad = std::fs::read(&path).unwrap();
    let mut lumps = HashMap::new();
    for i in 0..long(&wad, 4) {
        let at = long(&wad, 8) + 16 * i;
        let start = long(&wad, at);
        let length = long(&wad, at + 4);
        lumps.insert(name(&wad[at + 8..at + 16]), &wad[start..start + length]);
    }
    let args: Vec<_> = [
        "room",
        "-iwad",
        path.to_str().unwrap(),
        "-warp",
        "1",
        "1",
        "-nosound",
        "-nomusic",
        "-config",
        "/tmp/wdoom-wall-posts.cfg",
    ]
    .into_iter()
    .map(|s| std::ffi::CString::new(s).unwrap())
    .collect();
    let mut argv: Vec<_> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
    argv.push(std::ptr::null_mut());
    room::platform::init_start_time();
    unsafe {
        room::doom::d_loop::singletics = 1;
        room::doom::doomgeneric::doomgeneric_Create(args.len() as i32, argv.as_mut_ptr());
    }
    let pnames = lumps["PNAMES"];
    let textures = lumps["TEXTURE1"];
    let mut differences = Vec::new();
    for texture in 0..long(textures, 0) {
        let t = &textures[long(textures, 4 + texture * 4)..];
        let (w, h) = unsafe { room::doom::r_data::hw_texture_size(texture as i32) };
        let (w, h) = (w as usize, h as usize);
        let mut expected = vec![None; w * h];
        for patch in 0..word(t, 20) {
            let at = 22 + patch * 10;
            let (ox, oy, number) = (signed(t, at), signed(t, at + 2), word(t, at + 4));
            let patch_name = name(&pnames[4 + number * 8..12 + number * 8]);
            let image = *lumps
                .get(&patch_name)
                .unwrap_or_else(|| panic!("missing {patch_name} for {}", name(&t[..8])));
            for column in 0..word(image, 0) {
                let x = column as i32 + ox;
                if x < 0 || x >= w as i32 {
                    continue;
                }
                let mut post = long(image, 8 + column * 4);
                while image[post] != 255 {
                    let top = image[post] as i32;
                    let count = image[post + 1] as usize;
                    for row in 0..count {
                        let y = oy + top + row as i32;
                        if y >= 0 && y < h as i32 {
                            expected[y as usize * w + x as usize] = Some(image[post + 3 + row]);
                        }
                    }
                    post += count + 4;
                }
            }
        }
        let (actual_w, actual_h, actual) =
            unsafe { room::doom::r_data::hw_texture_pixels(texture as i32).unwrap() };
        assert_eq!((actual_w as usize, actual_h as usize), (w, h));
        let wrong = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        if wrong != 0 {
            differences.push(format!("{}: {wrong} wrong pixels", name(&t[..8])));
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}
