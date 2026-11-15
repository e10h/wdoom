#!/usr/bin/env python3
"""Pack the native Brightmaps Plus emission masks for web builds; needs Pillow."""
import argparse
import itertools
from pathlib import Path
import re
import struct
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='bmplus_textures/brightmaps/textures/doom directory')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'room/assets/brightmaps/doom.rle')
args = parser.parse_args()

def load(directory):
    result = {}
    for path in directory.glob('*.png'):
        image = Image.open(path).convert('RGBA')
        pixels = bytearray()
        for r, g, b, a in image.get_flattened_data():
            mask = max(r, g, b) * a // 255
            code = min(7, 1 + (mask * 6 + 254) // 255)
            pixels.append(0 if mask == 0 else 128 if code <= 4 else 255)
        result[path.stem.upper()] = (image.width, image.height, pixels)
    return result

records = []
for kind, label, masks in [(0, 'texture', load(args.source)), (1, 'flat', load(args.source / 'flats'))]:
    definitions = args.source.parent / ('doom_textures.bm' if kind == 0 else 'doom_flats.bm')
    for line in definitions.read_text().splitlines():
        match = re.match(r'\s*brightmap\s+' + label + r'\s+(\S+).*?map\s+"([^"]+)"', line.split('//')[0])
        if match:
            material, path = match.groups()
            if Path(path).stem.upper() in masks:
                masks[material.upper()] = masks[Path(path).stem.upper()]
    for name, (width, height, pixels) in sorted(masks.items()):
        if len(name) > 8:
            continue
        encoded = bytearray()
        for value, group in itertools.groupby(pixels):
            count = sum(1 for _ in group)
            while count:
                chunk = min(count, 65535)
                encoded.extend(struct.pack('<HB', chunk, value))
                count -= chunk
        records.append(struct.pack('<B8sHHI', kind, name.encode().ljust(8, b'\0'), width, height, len(encoded)) + encoded)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(b'RBM1' + struct.pack('<I', len(records)) + b''.join(records))
print(f'{len(records)} masks, {args.output.stat().st_size} bytes')
