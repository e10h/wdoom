# Web deployment

Vercel deploys the `main` branch to https://www.wdoom.com. Its build command is
`python3 tools/web_release.py build`; output is `out/vercel`.

The checked-in `web-dist` directory is a self-contained static release. Vercel
validates its source fingerprint and artifact hashes before publishing, so
changing Rust/assets without refreshing the release fails rather than serving
an old WASM binary.

## Refreshing the release

1. Build the browser package:
   `wasm-pack build room --target web --out-dir ../web/pkg --release --no-opt`.
2. Get the current map bake names from the native renderer. These names include
   the renderer's packing hash; do not substitute older bakes with different hashes.
   Put one `WEB_BAKE light_e1m1_HASH.rmlt` line per map in a text file.
3. Run `python3 tools/web_release.py package --keys /path/to/bake-keys.txt`.
4. Run `python3 tools/web_release.py build`, then commit the source changes and
   updated `web-dist` together. Test the branch's Vercel preview before updating main.

Native `.savegame` files are never modified by packaging. Available matching
bakes are copied losslessly with gzip encoding; the manifest records hashes of
both their compressed artifacts and original native bytes. Vercel's
`Content-Encoding: gzip` header lets normal browser fetches decode them.
Unbaked maps retain the renderer's minimum lighting until a native bake is added.

JavaScript/WASM live under a content-versioned asset directory, avoiding stale
browser caches when moving from the old renderer or publishing a new build.

## Existing water page

`/water/` keeps the previous Emscripten water experience. Its runtime and title
assets were copied from the existing public production deployment, source
commit `e74a2f406ce415ae4b7e1b50cf0f3bedb2c78ce5`; licenses are included. That
commit is retained in main's history. The main `/` route serves the Rust renderer.
