#!/usr/bin/env python3
"""Pack authored PBR normal/roughness/metallic maps for WDoom; needs Pillow."""
import argparse
import struct
import zlib
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/Users/ethan/Code/octree-tracer/files/dds_textures_pbr'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'room/assets/pbr/doom_pbr.bin')
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source {args.source} does not exist, skipping.")
        return

    base = args.source
    norm_dir = base / 'materials/normalmaps/auto'
    rough_dir = base / 'materials/roughness/auto'
    met_dir = base / 'materials/metallic/auto'

    records = []
    # Find all normal maps
    for norm_file in sorted(norm_dir.glob('*.dds')):
        name = norm_file.stem.upper()
        if len(name) > 8:
            continue
        try:
            norm_im = Image.open(norm_file).convert('RGBA')
        except Exception:
            continue

        w = min(norm_im.width, 128)
        h = min(norm_im.height, 128)
        norm_im = norm_im.resize((w, h), Image.Resampling.LANCZOS)

        # Roughness map
        rough_file = rough_dir / norm_file.name
        if not rough_file.exists():
            rough_file = next(rough_dir.glob(norm_file.stem + '.*'), None)
        if rough_file and rough_file.exists():
            rough_im = Image.open(rough_file).convert('L').resize((w, h))
        else:
            rough_im = Image.new('L', (w, h), 190)

        # Metallic map
        met_file = met_dir / norm_file.name
        if not met_file.exists():
            met_file = next(met_dir.glob(norm_file.stem + '.*'), None)
        if met_file and met_file.exists():
            met_im = Image.open(met_file).convert('L').resize((w, h))
        else:
            is_metal = any(name.startswith(p) for p in ['COMP', 'LITE', 'SHAWN', 'TEKW', 'DOOR', 'STARG', 'SUPPORT'])
            met_im = Image.new('L', (w, h), 200 if is_metal else 0)

        norm_bytes = norm_im.tobytes()
        rough_bytes = rough_im.tobytes()
        met_bytes = met_im.tobytes()

        pbr_bytes = bytearray(w * h * 4)
        for i in range(w * h):
            pbr_bytes[i * 4] = norm_bytes[i * 4]         # Normal X
            pbr_bytes[i * 4 + 1] = norm_bytes[i * 4 + 1] # Normal Y
            pbr_bytes[i * 4 + 2] = rough_bytes[i]        # Roughness
            pbr_bytes[i * 4 + 3] = met_bytes[i]          # Metallic

        records.append((name, w, h, pbr_bytes))

    header = bytearray(b"RPBR" + struct.pack('<I', len(records)))
    body = bytearray()
    for name, w, h, data in records:
        comp_data = zlib.compress(data, 9)
        body.extend(struct.pack('<8sHHII', name.encode().ljust(8, b'\0'), w, h, len(comp_data), len(data)))
        body.extend(comp_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + body)
    print(f"Packed {len(records)} PBR records into {args.output} ({len(header + body)} bytes, {len(header + body)/1024:.1f} KB)")

if __name__ == '__main__':
    main()
