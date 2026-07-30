#!/usr/bin/env python3
"""
build_masks.py — voxelise the bone meshes into ground-truth masks for the study.

The meshes in bones/mesh/*.ply are the true segmentation, and they are already
registered in the CT's patient coordinate frame, so they can be sampled straight
onto the slice grid that build_slices.py produced. Run build_slices.py first.

  python build_masks.py                        # all meshes, all slices
  python build_masks.py --meshes humerus scapula
  python build_masks.py --force                # redo existing masks

Output, alongside the slice PNGs (served at /segment/slices/<series>/):
  mask_<structure>_000.png …    8-bit grey, 0 or 255 — one set per structure
  manifest.json                 gains a "structures" list describing each set

One file per structure per slice, holding a plain 0/255 mask. Packing several
structures into one image as bit flags would be smaller, but reading it back
needs exact pixel values, and browsers are entitled to colour-manage a decoded
PNG — a value shifted by one would corrupt a bit test. A binary mask read with
a threshold survives that, structures that overlap each keep their own voxels,
and only the structure being traced is ever fetched. Empty area costs ~100 bytes
a slice.

Method: intersect each triangle with the slice planes it straddles, then fill
each 2D cross-section by even-odd parity along scanlines. Counting an edge as
crossing a plane when `(za <= k) != (zb <= k)` means every triangle yields
exactly 0 or 2 crossing points, which is what keeps the cross-sections closed.
Pure Python, so the thorax (1.4M triangles) takes a few minutes; the three
shoulder bones are much quicker.
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from array import array

from build_slices import write_png_grey

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, 'bones', 'mesh')
SLICES_DIR = os.path.join(HERE, 'bones', 'slices')
# Order fixes the bit assignment, so keep it stable once masks are published.
DEFAULT_MESHES = ['clavicle', 'humerus', 'scapula', 'thorax']


def read_ply(path):
    """Read a binary-little-endian triangle PLY -> (verts array('f'), faces array('i')).

    Only the layout these meshes use is supported (float x/y/z, uchar/int face
    lists); anything else raises rather than silently misreading coordinates.
    """
    with open(path, 'rb') as f:
        blob = f.read()
    marker = b'end_header\n'
    hend = blob.find(marker)
    if hend < 0:
        raise ValueError(f'{path}: no PLY end_header')
    hend += len(marker)
    header = blob[:hend].decode('ascii', 'replace')
    if 'binary_little_endian' not in header:
        raise ValueError(f'{path}: only binary_little_endian PLY is supported')

    nv = nf = None
    element = None
    vprops = []
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == 'element':
            element = tok[1]
            if element == 'vertex':
                nv = int(tok[2])
            elif element == 'face':
                nf = int(tok[2])
        elif tok[0] == 'property' and element == 'vertex':
            vprops.append(tok[1])
    if nv is None or nf is None:
        raise ValueError(f'{path}: missing vertex/face counts')
    if vprops[:3] != ['float', 'float', 'float']:
        raise ValueError(f'{path}: expected float x/y/z first, got {vprops[:3]}')

    sizes = {'float': 4, 'double': 8, 'int': 4, 'uint': 4, 'uchar': 1, 'char': 1,
             'short': 2, 'ushort': 2}
    vstride = sum(sizes[p] for p in vprops)
    vend = hend + nv * vstride
    if vstride == 12:
        verts = array('f')
        verts.frombytes(blob[hend:vend])
    else:                                    # extra per-vertex properties
        verts = array('f', [0.0]) * (nv * 3)
        for i in range(nv):
            verts[i * 3:i * 3 + 3] = array('f', struct.unpack_from('<3f', blob, hend + i * vstride))

    face_bytes = len(blob) - vend
    faces = array('i')
    if face_bytes == nf * 13:                # every face a triangle: bulk read
        for cnt, a, b, c in struct.iter_unpack('<BIII', blob[vend:]):
            if cnt != 3:
                raise ValueError(f'{path}: non-triangle face (count {cnt})')
            faces.append(a); faces.append(b); faces.append(c)
    else:
        pos = vend
        for _ in range(nf):
            cnt = blob[pos]; pos += 1
            idx = struct.unpack_from('<%dI' % cnt, blob, pos)
            pos += 4 * cnt
            for k in range(1, cnt - 1):      # fan-triangulate
                faces.append(idx[0]); faces.append(idx[k]); faces.append(idx[k + 1])
    return verts, faces


def voxelize(verts, faces, size, nslices, ox, oy, oz, sx, sy, dz):
    """Rasterise a closed mesh onto the slice grid. Returns list of bytearrays.

    verts are in patient mm; (ox, oy, oz) is the centre of voxel (0,0,0) and
    (sx, sy, dz) its size, so integer voxel coordinates land on pixel centres.
    """
    nv3 = len(verts)
    # Patient mm -> continuous voxel coordinates, once per vertex.
    vx = array('f', bytes(4 * (nv3 // 3)))
    vy = array('f', bytes(4 * (nv3 // 3)))
    vz = array('f', bytes(4 * (nv3 // 3)))
    for i in range(nv3 // 3):
        vx[i] = (verts[i * 3] - ox) / sx
        vy[i] = (verts[i * 3 + 1] - oy) / sy
        vz[i] = (verts[i * 3 + 2] - oz) / dz

    # Cross-section segments per slice, flat as x0,y0,x1,y1.
    segs = [array('f') for _ in range(nslices)]
    nseg = 0
    for t in range(0, len(faces), 3):
        i0, i1, i2 = faces[t], faces[t + 1], faces[t + 2]
        z0, z1, z2 = vz[i0], vz[i1], vz[i2]
        klo = int(math.ceil(min(z0, z1, z2)))
        khi = int(math.floor(max(z0, z1, z2)))
        if khi < 0 or klo > nslices - 1:
            continue
        if klo < 0:
            klo = 0
        if khi > nslices - 1:
            khi = nslices - 1
        x0, y0 = vx[i0], vy[i0]
        x1, y1 = vx[i1], vy[i1]
        x2, y2 = vx[i2], vy[i2]
        for k in range(klo, khi + 1):
            # Half-open crossing test: each triangle gives exactly 0 or 2 points.
            a0 = z0 <= k
            a1 = z1 <= k
            a2 = z2 <= k
            px = py = qx = qy = 0.0
            n = 0
            if a0 != a1:
                f = (k - z0) / (z1 - z0)
                px, py = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                n = 1
            if a1 != a2:
                f = (k - z1) / (z2 - z1)
                if n:
                    qx, qy = x1 + (x2 - x1) * f, y1 + (y2 - y1) * f
                    n = 2
                else:
                    px, py = x1 + (x2 - x1) * f, y1 + (y2 - y1) * f
                    n = 1
            if n == 1 and a2 != a0:
                f = (k - z2) / (z0 - z2)
                qx, qy = x2 + (x0 - x2) * f, y2 + (y0 - y2) * f
                n = 2
            if n == 2:
                s = segs[k]
                s.append(px); s.append(py); s.append(qx); s.append(qy)
                nseg += 1

    # Even-odd scanline fill, slice by slice.
    masks = []
    for k in range(nslices):
        m = bytearray(size * size)
        s = segs[k]
        if s:
            rows = [None] * size
            for j in range(0, len(s), 4):
                ax, ay, bx, by = s[j], s[j + 1], s[j + 2], s[j + 3]
                if ay == by:
                    continue                     # horizontal: no crossing
                if ay < by:
                    ylo, yhi = ay, by
                else:
                    ylo, yhi = by, ay
                r0 = int(math.ceil(ylo))
                r1 = int(math.ceil(yhi)) - 1     # ylo <= r < yhi
                if r1 < 0 or r0 > size - 1:
                    continue
                if r0 < 0:
                    r0 = 0
                if r1 > size - 1:
                    r1 = size - 1
                inv = (bx - ax) / (by - ay)
                for r in range(r0, r1 + 1):
                    x = ax + (r - ay) * inv
                    if rows[r] is None:
                        rows[r] = [x]
                    else:
                        rows[r].append(x)
            for r in range(size):
                xs = rows[r]
                if not xs:
                    continue
                xs.sort()
                base = r * size
                for j in range(0, len(xs) - 1, 2):
                    c0 = int(math.ceil(xs[j]))
                    c1 = int(math.floor(xs[j + 1]))
                    if c1 < 0 or c0 > size - 1:
                        continue
                    if c0 < 0:
                        c0 = 0
                    if c1 > size - 1:
                        c1 = size - 1
                    for c in range(base + c0, base + c1 + 1):
                        m[c] = 1
        masks.append(m)
        segs[k] = None                           # release as we go
    return masks, nseg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--series', default=None, help='series to build (default: all built)')
    ap.add_argument('--meshes', nargs='+', default=None,
                    help=f'mesh names without .ply (default: {" ".join(DEFAULT_MESHES)})')
    ap.add_argument('--force', action='store_true', help='rebuild existing masks')
    args = ap.parse_args()

    if not os.path.isdir(SLICES_DIR):
        print(f'No built slices at {SLICES_DIR} — run build_slices.py first')
        return 1
    series = ([args.series] if args.series
              else sorted(d for d in os.listdir(SLICES_DIR)
                          if os.path.exists(os.path.join(SLICES_DIR, d, 'manifest.json'))))
    if not series:
        print(f'No manifests under {SLICES_DIR} — run build_slices.py first')
        return 1

    wanted = args.meshes or DEFAULT_MESHES
    meshes = [(n, os.path.join(MESH_DIR, n + '.ply')) for n in wanted]
    missing = [n for n, p in meshes if not os.path.exists(p)]
    if missing:
        print(f'Missing mesh files in {MESH_DIR}: {", ".join(missing)}')
        return 1

    for name in series:
        out_dir = os.path.join(SLICES_DIR, name)
        manifest = json.load(open(os.path.join(out_dir, 'manifest.json'), encoding='utf-8'))
        if manifest.get('structures') and not args.force:
            print(f'{name}: masks already built — use --force to redo')
            continue

        size = manifest['size']
        nslices = manifest['count']
        src_cols = manifest['source']['cols']
        src_sx, src_sy = manifest['pixel_spacing']
        ox_src, oy_src, oz = manifest['origin']
        dz = manifest['slice_spacing']
        # The slice PNGs are area-averaged f:1, so an output pixel centre sits
        # (f-1)/2 source pixels in from the original origin.
        f = src_cols / size
        sx, sy = src_sx * f, src_sy * f
        ox = ox_src + (f - 1) / 2 * src_sx
        oy = oy_src + (f - 1) / 2 * src_sy
        print(f'{name}: {nslices} slices, {size}x{size}, voxel '
              f'{sx:.3f}x{sy:.3f}x{dz:.3f} mm, origin ({ox:.2f}, {oy:.2f}, {oz:.2f})')

        structures = []
        grand_total = 0
        for mesh_name, mesh_path in meshes:
            t0 = time.time()
            verts, faces = read_ply(mesh_path)
            ntri = len(faces) // 3
            print(f'  {mesh_name}: {ntri:,} triangles…', end='', flush=True)
            masks, nseg = voxelize(verts, faces, size, nslices, ox, oy, oz, sx, sy, dz)
            voxels = 0
            first = last = None
            written = 0
            for k in range(nslices):
                m = masks[k]
                hit = 0
                for i in range(size * size):
                    if m[i]:
                        m[i] = 255          # binary mask, read back with a threshold
                        hit += 1
                if hit:
                    voxels += hit
                    first = k if first is None else first
                    last = k
                written += write_png_grey(
                    os.path.join(out_dir, f'mask_{mesh_name}_{k:03d}.png'), m, size)
                masks[k] = None
            grand_total += written
            structures.append({
                'name': mesh_name,
                'mask_prefix': f'mask_{mesh_name}_',
                'voxels': voxels,
                'slices': [first, last] if first is not None else [],
                'triangles': ntri,
                'bytes': written,
            })
            print(f' {nseg:,} segments, {voxels:,} voxels, slices {first}-{last}, '
                  f'{written / 1e6:.2f} MB  ({time.time() - t0:.0f}s)')

        manifest['structures'] = structures
        manifest['mask_bytes'] = grand_total
        total, nslices_written = grand_total, nslices * len(meshes)
        with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=1)
        print(f'  wrote {nslices_written} mask PNGs across {len(meshes)} structures, '
              f'{total / 1e6:.2f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
