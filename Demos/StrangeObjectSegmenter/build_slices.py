#!/usr/bin/env python3
"""
build_slices.py — turn the DICOM stacks in bones/image/ into web-ready PNGs.

The browser demo can't read 159 MB of 16-bit DICOM over Wi-Fi, so this converts
each series once into windowed 8-bit grayscale PNGs plus a manifest the viewer
reads at startup. Every slice is converted; the demo decides how many to show
(see SLICE_STRIDE in web/index.html), so the full-resolution stack stays
available without a rebuild.

  python build_slices.py                  # convert every series in bones/image/
  python build_slices.py --size 512       # keep native in-plane resolution
  python build_slices.py --force          # re-convert even if outputs exist
  python build_slices.py --series right_shoulder

No third-party packages: the DICOMs are uncompressed (Implicit VR Little
Endian), so a small tag walker reads them, and PNG output only needs zlib.
Compressed transfer syntaxes are detected and reported rather than mangled.

Output (served by DemoServer at /segment/slices/<series>/…):
  bones/slices/<series>/manifest.json
  bones/slices/<series>/000.png, 001.png, …
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, 'bones', 'image')
OUT_DIR = os.path.join(HERE, 'bones', 'slices')

IMPLICIT_VR_LE = '1.2.840.10008.1.2'
EXPLICIT_VR_LE = '1.2.840.10008.1.2.1'
UNCOMPRESSED = {IMPLICIT_VR_LE, EXPLICIT_VR_LE}
PIXEL_DATA = (0x7FE0, 0x0010)
# In explicit VR these carry a 4-byte length after 2 reserved bytes.
LONG_VRS = {b'OB', b'OW', b'OF', b'SQ', b'UT', b'UN'}
# Tags we read. US ones are 16-bit ints; the rest are ASCII (DS/IS/CS/UI/LO).
US_TAGS = {(0x0028, 0x0002), (0x0028, 0x0010), (0x0028, 0x0011),
           (0x0028, 0x0100), (0x0028, 0x0101), (0x0028, 0x0102), (0x0028, 0x0103)}
TAGS = {
    (0x0008, 0x0060): 'modality',
    (0x0008, 0x103E): 'description',
    (0x0018, 0x0050): 'slice_thickness',
    (0x0020, 0x0013): 'instance',
    (0x0020, 0x0032): 'position',
    (0x0020, 0x0037): 'orientation',
    (0x0028, 0x0002): 'samples',
    (0x0028, 0x0004): 'photometric',
    (0x0028, 0x0010): 'rows',
    (0x0028, 0x0011): 'cols',
    (0x0028, 0x0030): 'pixel_spacing',
    (0x0028, 0x0100): 'bits_allocated',
    (0x0028, 0x0103): 'pixel_representation',
    (0x0028, 0x1050): 'window_center',
    (0x0028, 0x1051): 'window_width',
    (0x0028, 0x1052): 'intercept',
    (0x0028, 0x1053): 'slope',
}


class DicomError(Exception):
    pass


def read_dicom(path):
    """Return (header dict, pixel data bytes) for an uncompressed DICOM file."""
    with open(path, 'rb') as f:
        buf = f.read()
    if buf[128:132] != b'DICM':
        raise DicomError('not a Part-10 DICOM file (no DICM magic)')

    # File meta group (0002,….) is always explicit VR little-endian.
    ts, pos = None, 132
    while pos + 8 <= len(buf):
        group, elem = struct.unpack_from('<HH', buf, pos)
        if group != 0x0002:
            break
        vr = buf[pos + 4:pos + 6]
        if vr in LONG_VRS:
            (ln,) = struct.unpack_from('<I', buf, pos + 8)
            vpos = pos + 12
        else:
            (ln,) = struct.unpack_from('<H', buf, pos + 6)
            vpos = pos + 8
        if (group, elem) == (0x0002, 0x0010):
            ts = buf[vpos:vpos + ln].decode('ascii', 'replace').strip('\x00 ')
        pos = vpos + ln
    if ts not in UNCOMPRESSED:
        raise DicomError(f'transfer syntax {ts} is compressed — needs a real '
                         'DICOM library (pydicom + pylibjpeg) to decode')

    explicit = ts != IMPLICIT_VR_LE
    hdr = {}
    while pos + 8 <= len(buf):
        group, elem = struct.unpack_from('<HH', buf, pos)
        pos += 4
        if explicit:
            vr = buf[pos:pos + 2]
            pos += 2
            if vr in LONG_VRS:
                pos += 2
                (ln,) = struct.unpack_from('<I', buf, pos); pos += 4
            else:
                (ln,) = struct.unpack_from('<H', buf, pos); pos += 2
        else:
            (ln,) = struct.unpack_from('<I', buf, pos); pos += 4
        if ln == 0xFFFFFFFF:
            raise DicomError('undefined-length element (nested sequence) '
                             'before pixel data')
        if (group, elem) == PIXEL_DATA:
            return hdr, buf[pos:pos + ln]
        key = TAGS.get((group, elem))
        if key:
            raw = buf[pos:pos + ln]
            hdr[key] = (struct.unpack_from('<H', raw)[0] if (group, elem) in US_TAGS
                        else raw.decode('ascii', 'replace').strip().strip('\x00'))
        pos += ln
    raise DicomError('no pixel data element found')


def first_float(val, default):
    """DICOM multi-values are backslash separated ('500\\40'); take the first."""
    try:
        return float(str(val).split('\\')[0])
    except (TypeError, ValueError):
        return default


def floats(val):
    try:
        return [float(v) for v in str(val).split('\\')]
    except (TypeError, ValueError):
        return []


def to_grey(pix, hdr, out_size, center, width):
    """16-bit stored pixels -> windowed 8-bit bytes, area-averaged to out_size.

    Averaging happens in Hounsfield space before windowing: nearest-neighbour
    sampling drops thin bright structures (cortical bone, wires) that matter
    here, and those are exactly what a visitor is asked to trace.
    """
    rows, cols = hdr['rows'], hdr['cols']
    signed = hdr.get('pixel_representation', 0) == 1
    vals = struct.unpack_from(('<%dh' if signed else '<%dH') % (rows * cols), pix, 0)
    slope = first_float(hdr.get('slope'), 1.0) or 1.0
    inter = first_float(hdr.get('intercept'), 0.0)

    lo = center - width / 2.0
    scale = 255.0 / width
    out = bytearray(out_size * out_size)
    # Integer edges so every source pixel lands in exactly one output cell.
    ys = [y * rows // out_size for y in range(out_size + 1)]
    xs = [x * cols // out_size for x in range(out_size + 1)]
    for oy in range(out_size):
        y0, y1 = ys[oy], max(ys[oy] + 1, ys[oy + 1])
        orow = oy * out_size
        for ox in range(out_size):
            x0, x1 = xs[ox], max(xs[ox] + 1, xs[ox + 1])
            total = 0
            for sy in range(y0, y1):
                base = sy * cols
                for sx in range(x0, x1):
                    total += vals[base + sx]
            n = (y1 - y0) * (x1 - x0)
            hu = (total / n) * slope + inter
            v = (hu - lo) * scale
            out[orow + ox] = 0 if v < 0 else (255 if v > 255 else int(v))
    return out


def write_png_grey(path, data, size):
    """8-bit grayscale PNG (zlib is all it takes — no PIL needed)."""
    def chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload
                + struct.pack('>I', zlib.crc32(tag + payload) & 0xFFFFFFFF))
    raw = bytearray()
    for y in range(size):
        raw.append(0)                       # per-scanline filter: none
        raw += data[y * size:(y + 1) * size]
    blob = (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
            + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


def convert_series(folder, out_root, size, center_arg, width_arg, force):
    name = os.path.basename(folder.rstrip('/\\'))
    files = sorted(glob.glob(os.path.join(folder, '*.dicom'))
                   + glob.glob(os.path.join(folder, '*.dcm')))
    if not files:
        print(f'  {name}: no .dicom/.dcm files — skipped')
        return None
    out_dir = os.path.join(out_root, name)
    manifest_path = os.path.join(out_dir, 'manifest.json')
    if os.path.exists(manifest_path) and not force:
        print(f'  {name}: already built ({manifest_path}) — use --force to redo')
        return json.load(open(manifest_path, encoding='utf-8'))
    os.makedirs(out_dir, exist_ok=True)

    print(f'  {name}: reading {len(files)} files…')
    slices = []
    for path in files:
        try:
            hdr, pix = read_dicom(path)
        except DicomError as e:
            print(f'    SKIP {os.path.basename(path)}: {e}')
            continue
        need = hdr['rows'] * hdr['cols'] * (hdr.get('bits_allocated', 16) // 8)
        if len(pix) < need:
            print(f'    SKIP {os.path.basename(path)}: pixel data {len(pix)} < {need}')
            continue
        pos = floats(hdr.get('position'))
        slices.append({'path': path, 'hdr': hdr, 'pix': pix,
                       'z': pos[2] if len(pos) == 3 else 0.0,
                       'instance': int(first_float(hdr.get('instance'), 0))})
    if not slices:
        print(f'  {name}: nothing decodable — skipped')
        return None

    # Sort by patient-space z so the stack is geometrically ordered (file names
    # and InstanceNumber run the other way in this study), which is also the
    # frame the bones/mesh PLYs are already registered in.
    slices.sort(key=lambda s: s['z'])
    h0 = slices[0]['hdr']
    center = center_arg if center_arg is not None else first_float(h0.get('window_center'), 500.0)
    width = width_arg if width_arg is not None else first_float(h0.get('window_width'), 2000.0)
    spacing = floats(h0.get('pixel_spacing')) or [1.0, 1.0]
    zs = [s['z'] for s in slices]
    gaps = [round(b - a, 4) for a, b in zip(zs, zs[1:])]
    dz = min(gaps) if gaps else first_float(h0.get('slice_thickness'), 1.0)
    if gaps and max(gaps) - min(gaps) > 1e-3:
        print(f'    NOTE: slice spacing is not uniform ({min(gaps)}–{max(gaps)} mm)')

    print(f'    {h0["cols"]}x{h0["rows"]} -> {size}x{size}, window C{center:.0f}/W{width:.0f}')
    total_bytes, t0 = 0, time.time()
    entries = []
    for i, s in enumerate(slices):
        grey = to_grey(s['pix'], s['hdr'], size, center, width)
        fname = f'{i:03d}.png'
        total_bytes += write_png_grey(os.path.join(out_dir, fname), grey, size)
        entries.append({'file': fname, 'z': round(s['z'], 4), 'instance': s['instance']})
        if (i + 1) % 50 == 0 or i + 1 == len(slices):
            print(f'      {i + 1}/{len(slices)} slices  ({total_bytes / 1e6:.1f} MB)')

    manifest = {
        'series': name,
        'description': h0.get('description', ''),
        'modality': h0.get('modality', ''),
        'size': size,
        'count': len(entries),
        'source': {'rows': h0['rows'], 'cols': h0['cols'],
                   'bits_allocated': h0.get('bits_allocated'),
                   'photometric': h0.get('photometric')},
        'pixel_spacing': [round(spacing[0], 6), round(spacing[1], 6)],
        'display_spacing': [round(spacing[0] * h0['cols'] / size, 6),
                            round(spacing[1] * h0['rows'] / size, 6)],
        'slice_spacing': dz,
        'window': {'center': center, 'width': width},
        # Patient-space frame, so mesh ground truth can be voxelised into it.
        'origin': floats(slices[0]['hdr'].get('position')),
        'orientation': floats(h0.get('orientation')),
        'bytes': total_bytes,
        'slices': entries,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
    print(f'    done in {time.time() - t0:.0f}s — {total_bytes / 1e6:.1f} MB '
          f'({total_bytes / len(entries) / 1024:.0f} KB/slice), manifest written')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=256,
                    help='output resolution per slice (default 256; 512 keeps '
                         'native in-plane detail at ~4x the bytes)')
    ap.add_argument('--window-center', type=float, default=None,
                    help='override the window centre in HU (default: from the file)')
    ap.add_argument('--window-width', type=float, default=None,
                    help='override the window width in HU (default: from the file)')
    ap.add_argument('--series', default=None, help='convert only this series folder')
    ap.add_argument('--force', action='store_true', help='re-convert existing output')
    args = ap.parse_args()

    if not os.path.isdir(IMAGE_DIR):
        print(f'No image folder at {IMAGE_DIR}')
        return 1
    folders = [os.path.join(IMAGE_DIR, d) for d in sorted(os.listdir(IMAGE_DIR))
               if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    if args.series:
        folders = [f for f in folders if os.path.basename(f) == args.series]
        if not folders:
            print(f'No series folder named {args.series!r} in {IMAGE_DIR}')
            return 1
    if not folders:
        print(f'No series folders in {IMAGE_DIR}')
        return 1

    print(f'Converting {len(folders)} series from {IMAGE_DIR}')
    built = [convert_series(f, OUT_DIR, args.size, args.window_center,
                            args.window_width, args.force) for f in folders]
    built = [b for b in built if b]
    if not built:
        return 1
    print(f'\nWrote {sum(b["count"] for b in built)} slices, '
          f'{sum(b["bytes"] for b in built) / 1e6:.1f} MB total, to {OUT_DIR}')
    print('Served by DemoServer at /segment/slices/<series>/manifest.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
