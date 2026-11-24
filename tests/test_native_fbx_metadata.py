import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap.fbx_writer import write


class NativeFbxMetadataTest(unittest.TestCase):
    def test_required_metadata_precedes_scene_records(self):
        positions = np.array([[0., 0., 0.], [0., 1., 0.]])
        basis = np.repeat(np.eye(3)[None], 2, axis=0)
        frames = np.tile(np.eye(4), (2, 2, 1, 1))
        frames[:, :, :3, 3] = positions
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'motion.fbx'
            write(target, ['Hips', 'Child'], [-1, 0], positions, basis, basis,
                  positions, frames, 30, expected_bones=2)
            data = target.read_bytes()

        offset = 27
        records = []
        for _ in range(4):
            end, prop_count, prop_size, name_size = struct.unpack_from('<IIIB', data, offset)
            start = offset + 13
            name = data[start:start + name_size]
            props_start = start + name_size
            records.append((name, prop_count, data[props_start:props_start + prop_size]))
            offset = end

        self.assertEqual([row[0] for row in records],
                         [b'FBXHeaderExtension', b'FileId', b'CreationTime', b'GlobalSettings'])
        self.assertEqual(records[1][1:], (1, b'R' + struct.pack('<I', 16)
                                        + bytes.fromhex('28b32aebb624ccc2bfc8b02aa92bfcf1')))
        creation_time = b'1970-01-01 10:00:00:000'
        self.assertEqual(records[2][1:], (1, b'S' + struct.pack('<I', len(creation_time)) + creation_time))
