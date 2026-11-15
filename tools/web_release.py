#!/usr/bin/env python3
"""Package/validate a self-contained web release; Vercel serves the checked-in build."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'web-dist'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def source_hash():
    paths = []
    for name in ['Cargo.toml', 'Cargo.lock', 'room/Cargo.toml', 'room/build.rs',
                 'room/src', 'room/assets', 'doomgeneric-sys', 'vendor/doomgeneric', 'web/index.html', 'web/index.js']:
        path = ROOT / name
        paths.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()

def bake_valid(data, name):
    match = re.fullmatch(r'light_e(\d+)m(\d+)_([0-9a-f]{16})\.rmlt', name)
    if not match or len(data) < 32:
        raise ValueError(f'Invalid native bake {name}')
    magic, version, ep, level, key, size, count = struct.unpack('<4sIIIQII', data[:32])
    if (magic, version, ep, level, key) != (b'RMLT', 1, int(match[1]), int(match[2]), int(match[3], 16)):
        raise ValueError(f'Bake header mismatch: {name}')
    if size != 4096 or count > size * size or len(data) != 32 + count * 12:
        raise ValueError(f'Incomplete native bake: {name}')

def package(keys):
    pkg = ROOT / 'web/pkg'
    wasm = (pkg / 'room_bg.wasm').read_bytes()
    if wasm[:4] != b'\0asm':
        raise ValueError('Build web/pkg with wasm-pack first')
    version = sha(wasm + (pkg / 'room.js').read_bytes() + (ROOT / 'web/index.js').read_bytes())[:16]
    assets = DIST / 'assets'
    if assets.exists():
        shutil.rmtree(assets)
    target = assets / version
    (target / 'pkg').mkdir(parents=True)
    for name in ['room.js', 'room_bg.wasm']:
        shutil.copyfile(pkg / name, target / 'pkg' / name)
    shutil.copyfile(ROOT / 'web/index.js', target / 'index.js')
    html = (ROOT / 'web/index.html').read_text().replace('<title>room</title>', '<title>wDoom</title>')
    html = html.replace('src="index.js"', f'src="assets/{version}/index.js"')
    (DIST / 'index.html').write_text(html)
    light = DIST / 'light'
    light.mkdir(exist_ok=True)
    for path in light.glob('*.rmlt'):
        path.unlink()
    bakes = {}
    missing = []
    for name in keys:
        source = ROOT / '.savegame' / name
        if not source.exists():
            missing.append(name)
            continue
        data = source.read_bytes()
        bake_valid(data, name)
        (light / name).write_bytes(gzip.compress(data, compresslevel=6, mtime=0))
        bakes[name] = {'nativeSha256': sha(data), 'nativeBytes': len(data)}
    shutil.copyfile(ROOT / 'LICENSE-GPL-2.0', DIST / 'LICENSE')
    for name in ['LICENSE-DOOM1', 'NOTICE.md']:
        shutil.copyfile(DIST / 'water' / name, DIST / name)
    files = {str(p.relative_to(DIST)): sha(p.read_bytes()) for p in sorted(DIST.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    manifest = {'version': version, 'sourceSha256': source_hash(), 'bakes': bakes,
                'unbakedMaps': missing, 'files': files, 'legacySource': 'e74a2f406ce415ae4b7e1b50cf0f3bedb2c78ce5'}
    (DIST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged {len(files)} files, {len(bakes)} exact native bakes; build {version}')

def verify():
    manifest = json.loads((DIST / 'manifest.json').read_text())
    if source_hash() != manifest['sourceSha256']:
        raise ValueError('Rust/assets changed: rebuild WASM and run the package command before deploying')
    actual = {str(p.relative_to(DIST)) for p in DIST.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('Deployment file list differs from manifest')
    for name, expected in manifest['files'].items():
        path = DIST / name
        if path.is_symlink() or sha(path.read_bytes()) != expected:
            raise ValueError(f'Changed or linked deployment artifact: {name}')
        if path.stat().st_size > 100_000_000:
            raise ValueError(f'Deployment asset is too large: {name}')
    for name, info in manifest['bakes'].items():
        data = gzip.decompress((DIST / 'light' / name).read_bytes())
        bake_valid(data, name)
        if len(data) != info['nativeBytes'] or sha(data) != info['nativeSha256']:
            raise ValueError(f'Native bake bytes changed: {name}')
    print(f'Verified {len(actual)} artifacts and lossless native bakes')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['package', 'verify', 'build'])
    parser.add_argument('--keys', type=Path, help='File containing WEB_BAKE lines from the native renderer')
    args = parser.parse_args()
    if args.command == 'package':
        if not args.keys:
            parser.error('package requires --keys')
        package(re.findall(r'WEB_BAKE (light_\S+\.rmlt)', args.keys.read_text()))
    verify()
    if args.command == 'build':
        output = ROOT / 'out/vercel'
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(DIST, output)
        print('Static release ready in out/vercel')

if __name__ == '__main__':
    main()
