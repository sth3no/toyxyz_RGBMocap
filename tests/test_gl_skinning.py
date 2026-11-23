"""GPU skinning parity against numpy references (transform feedback and compute).

These tests need a real OpenGL 4.1+ Core context. They skip when none can be
created, for example under QT_QPA_PLATFORM=offscreen on macOS or Windows. Run
this file without that variable on a desktop session to exercise the GPU.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import (QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
    QOpenGLFunctions_4_1_Core, QOpenGLFunctions_4_3_Core)
from PySide6.QtWidgets import QApplication

from mocap import mesh_gl
from mocap.mesh_data import adjacency
from mocap.rig_mesh import reference_vertices

APP = QApplication.instance() or QApplication([])


def smooth_normals(positions, offsets, neighbors):
    counts = np.diff(offsets.astype(np.int64))
    centers = np.repeat(np.arange(len(positions)), counts)
    pairs = neighbors.reshape(-1, 2).astype(np.int64)
    cross = np.cross(positions[pairs[:, 0]] - positions[centers], positions[pairs[:, 1]] - positions[centers])
    total = np.zeros_like(positions)
    np.add.at(total, centers, cross)
    return total / np.linalg.norm(total, axis=1, keepdims=True)


def sheet(rows, cols):
    """A gently curved, fully connected triangle sheet with stable normals."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    points = np.stack((c / cols - .5, r / max(1, rows - 1) - .5, .05 * np.sin(c * .02)), -1).reshape(-1, 3)
    a = (r[:-1, :-1] * cols + c[:-1, :-1]).ravel()
    faces = np.concatenate((np.stack((a, a + 1, a + cols + 1), 1), np.stack((a, a + cols + 1, a + cols), 1)))
    return points, faces.astype(np.uint32)


def rigid(rng, count, spread=.1):
    from scipy.spatial.transform import Rotation
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, :3, :3] = Rotation.from_rotvec(rng.standard_normal((count, 3)) * .3).as_matrix()
    matrices[:, :3, 3] = rng.standard_normal((count, 3)) * spread
    return matrices


def smplx_like(rng):
    """SMPL-X-sized packed model (10475 vertices) in the mesh_data layout."""
    rows, cols = 5, 2095
    template, faces = sheet(rows, cols)
    count = len(template)
    directions = rng.standard_normal((506, count, 3)) * .002
    weights = np.zeros((count, 55))
    for _ in range(4):
        weights[np.arange(count), rng.integers(0, 55, count)] += rng.random(count) + .1
    weights /= weights.sum(1, keepdims=True)
    packed = np.concatenate((template.ravel(), directions.ravel(), weights.ravel())).astype(np.float32)
    offsets, neighbors = adjacency(faces.astype(np.int64), count)
    transforms = rigid(rng, 55)
    frame = np.concatenate((transforms.reshape(-1), rng.standard_normal(486) * .3,
                            rng.standard_normal(20) * .5, [.1, -.2, .3])).astype(np.float32)
    asset = dict(key=("synthetic-smplx",), packed=packed, faces=faces, offsets=offsets, neighbors=neighbors,
                 vertex_count=np.array(count))
    return asset, frame


def smplx_reference(asset, frame):
    count = int(asset["vertex_count"]);n = 3 * count
    packed = asset["packed"].astype(np.float64);frame = frame.astype(np.float64)
    shape = packed[n:21 * n].reshape(20, n);pose = packed[21 * n:507 * n].reshape(486, n)
    posed = (packed[:n] + frame[1366:1386] @ shape + frame[880:1366] @ pose).reshape(count, 3)
    blend = (packed[507 * n:].reshape(count, 55) @ frame[:880].reshape(55, 16)).reshape(count, 4, 4)
    return np.einsum("vij,vj->vi", blend[:, :3, :3], posed) + blend[:, :3, 3] + frame[1386:]


class GLSkinningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        cls.context = QOpenGLContext()
        cls.context.setFormat(fmt)
        cls.surface = QOffscreenSurface()
        if not cls.context.create():
            raise unittest.SkipTest("No OpenGL context can be created on this platform")
        cls.surface.setFormat(cls.context.format())
        cls.surface.create()
        if not cls.surface.isValid() or not cls.context.makeCurrent(cls.surface):
            raise unittest.SkipTest("Cannot make an OpenGL context current")
        if not QOpenGLFunctions_4_1_Core().initializeOpenGLFunctions():
            cls.context.doneCurrent()
            raise unittest.SkipTest("OpenGL 4.1 Core is unavailable")
        cls.backends = ["feedback"]
        if QOpenGLFunctions_4_3_Core().initializeOpenGLFunctions():
            cls.backends.append("compute")
        framebuffer = QOpenGLFramebufferObjectFormat()
        framebuffer.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        cls.fbo = QOpenGLFramebufferObject(96, 96, framebuffer)
        cls.fbo.bind()

    @classmethod
    def tearDownClass(cls):
        cls.fbo.release()
        del cls.fbo
        cls.context.doneCurrent()

    def setUp(self):
        self.saved = os.environ.get(mesh_gl.BACKEND_ENV)

    def tearDown(self):
        if self.saved is None:
            os.environ.pop(mesh_gl.BACKEND_ENV, None)
        else:
            os.environ[mesh_gl.BACKEND_ENV] = self.saved

    def each_backend(self, factory, check):
        """Run check(renderer) on transform feedback, and on compute when 4.3 exists."""
        for backend in self.backends:
            if backend == "feedback":
                os.environ[mesh_gl.BACKEND_ENV] = "feedback"
            else:
                os.environ.pop(mesh_gl.BACKEND_ENV, None)
            with self.subTest(backend=backend, renderer=factory.__name__):
                renderer = factory()
                try:
                    self.assertEqual(renderer.backend, backend)
                    check(renderer)
                    self.assert_state_restored(renderer)
                finally:
                    renderer.close()

    def assert_state_restored(self, renderer):
        gl = renderer.gl
        self.assertEqual(gl.glGetError(), 0)
        self.assertFalse(gl.glIsEnabled(mesh_gl.GL_RASTERIZER_DISCARD))
        self.assertEqual(int(gl.glGetIntegerv(0x84E0)), mesh_gl.GL_TEXTURE0)  # GL_ACTIVE_TEXTURE
        self.assertEqual(int(gl.glGetIntegerv(0x8C8F)), 0)  # GL_TRANSFORM_FEEDBACK_BUFFER_BINDING

    def drawn_pixels(self, renderer, **kwargs):
        gl = renderer.gl
        gl.glViewport(0, 0, 96, 96)
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(0x4000 | 0x100)
        renderer.draw(np.array([0., 0., -2.]), np.eye(3), 96, 96, **kwargs)
        self.assertEqual(gl.glGetError(), 0)
        image = self.fbo.toImage()
        pixels = np.array([[image.pixelColor(x, y).getRgb()[:3] for x in range(0, 96, 4)] for y in range(0, 96, 4)])
        return pixels[pixels.sum(-1) > 0]

    def test_smplx_skin_and_normals(self):
        from mocap.mesh_gl import MeshRenderer
        asset, frame = smplx_like(np.random.default_rng(0))
        expected = smplx_reference(asset, frame)
        normals = smooth_normals(expected, asset["offsets"], asset["neighbors"])
        def check(renderer):
            renderer.load(asset)
            renderer.deform(frame, 0)
            result = renderer.read_vertices()
            self.assertEqual(result.shape, (10475, 8))
            self.assertLess(np.abs(result[:, :3] - expected).max(), 5e-5)
            np.testing.assert_array_equal(result[:, 3], 1)
            self.assertLess(np.abs(result[:, 4:7] - normals).max(), 2e-3)
            np.testing.assert_array_equal(result[:, 7], 0)
            self.assertGreater(len(self.drawn_pixels(renderer)), 20)
            # A second actor sharing the registered model buffers.
            other = MeshRenderer(renderer)
            try:
                other.load(asset)
                moved = frame.copy();moved[1386:] += 1
                other.deform(moved, 1)
                self.assertLess(np.abs(other.read_vertices()[:, :3] - (expected + 1)).max(), 5e-5)
            finally:
                other.close()
        self.each_backend(MeshRenderer, check)

    def test_bound_mesh_matches_reference_vertices(self):
        from mocap.bound_mesh_gl import BoundMeshRenderer
        rng = np.random.default_rng(1)
        asset, frame = smplx_like(rng)
        count = int(asset["vertex_count"])
        binding = dict(key="synthetic-binding",
                       vertices=(asset["packed"][:3 * count].reshape(count, 3) + rng.standard_normal((count, 3)) * .001).astype(np.float32),
                       frames=rigid(rng, 55)[:, :3, :3].astype(np.float32))
        expected = reference_vertices(asset["packed"].astype(np.float64), frame.astype(np.float64), binding, count)
        def check(renderer):
            renderer.load(asset)
            renderer.bind_skin(binding)
            renderer.deform(frame, 0)
            self.assertLess(np.abs(renderer.read_vertices()[:, :3] - expected).max(), 5e-5)
            self.assertGreater(len(self.drawn_pixels(renderer)), 20)
        self.each_backend(BoundMeshRenderer, check)

    def test_rig_and_raw_paths(self):
        from mocap.rig_gl import RigRenderer
        from mocap.bone_gl import BoneRenderer
        rng = np.random.default_rng(2)
        points, faces = sheet(20, 150)
        count = len(points);bones = 52
        normals = rng.standard_normal((count, 3));normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        bone = rng.integers(0, bones, count)
        transforms = rigid(rng, bones, .3)
        rig = dict(key=("synthetic-rig",), mode="rig", vertex_count=count, bone_count=bones, faces=faces,
                   packed=np.column_stack((points, normals, bone)).astype(np.float32),
                   edges=faces[:, :2].ravel().copy())
        rig_frame = transforms.reshape(-1).astype(np.float32)
        expected = np.einsum("vij,vj->vi", transforms[bone, :3, :3], points) + transforms[bone, :3, 3]
        expected_normals = np.einsum("vij,vj->vi", transforms[bone, :3, :3], normals)
        offsets, neighbors = adjacency(faces.astype(np.int64), count)
        raw = dict(key=("synthetic-raw",), topology_key="synthetic-raw", mode="raw", vertex_count=count,
                   faces=faces, offsets=offsets, neighbors=neighbors)
        raw_frame = (points + rng.standard_normal((count, 3)) * .002).astype(np.float32)
        def check_rig(renderer):
            renderer.load(rig)
            renderer.deform(rig_frame, 0)
            result = renderer.read_vertices()
            self.assertLess(np.abs(result[:, :3] - expected).max(), 2e-5)
            self.assertLess(np.abs(result[:, 4:7] - expected_normals).max(), 2e-5)
            self.assertGreater(len(self.drawn_pixels(renderer)), 20)

        def check_raw(renderer):
            renderer.load(raw)
            renderer.deform(raw_frame.ravel(), 0)
            result = renderer.read_vertices()
            np.testing.assert_array_equal(result[:, :3], raw_frame)
            self.assertLess(np.abs(result[:, 4:7] - smooth_normals(raw_frame.astype(np.float64), offsets, neighbors)).max(), 2e-3)
            shared = RigRenderer(renderer)
            try:
                shared.load(dict(raw, key=("synthetic-raw", 2)))
                shared.deform((raw_frame + 1).ravel(), 0)
                np.testing.assert_array_equal(shared.read_vertices()[:, :3], raw_frame + 1)
            finally:
                shared.close()
        self.each_backend(RigRenderer, check_rig)
        self.each_backend(BoneRenderer, check_rig)
        self.each_backend(RigRenderer, check_raw)

    def character(self, rng, textured):
        points, faces = sheet(30, 60)
        count = len(points);palette = 300
        influences = rng.integers(1, 5, count)
        starts = np.r_[0, np.cumsum(influences)[:-1]]
        entries = np.column_stack((rng.integers(0, palette, influences.sum()), rng.random(influences.sum()) + .05))
        for start, number in zip(starts, influences):
            entries[start:start + number, 1] /= entries[start:start + number, 1].sum()
        offsets, neighbors = adjacency(faces.astype(np.int64), count)
        asset = dict(key=("synthetic-character", textured), vertices=points.astype(np.float32),
                     rows=np.column_stack((starts, influences)).astype(np.float32), weights=entries.astype(np.float32),
                     faces=faces, offsets=offsets, neighbors=neighbors,
                     colors=np.tile([.2, .8, .3], (count, 1)).astype(np.float32))
        if textured:
            corner_ids = faces.ravel()
            corners = np.column_stack((corner_ids, points[corner_ids, 0] + .5, points[corner_ids, 1] + .5)).astype(np.float32)
            image = np.zeros((4, 4, 4), np.uint8);image[..., 0] = 255;image[..., 3] = 255
            asset["textures"] = dict(corners=corners, images={"image0": image},
                                     batches=[dict(start=0, count=len(corners), image="image0", color=[1, 1, 1, 1])])
        matrices = rigid(rng, palette, .02)
        posed = np.zeros((count, 3))
        for vertex, (start, number) in enumerate(zip(starts, influences)):
            for b, w in entries[start:start + number]:
                posed[vertex] += w * (matrices[int(b), :3, :3] @ points[vertex] + matrices[int(b), :3, 3])
        return asset, palette, matrices.astype(np.float32), posed

    def test_character_skin_and_textured_draw(self):
        from mocap.character_gl import CharacterRenderer
        for textured in (False, True):
            asset, palette, matrices, expected = self.character(np.random.default_rng(3), textured)
            def check(renderer):
                renderer.load(asset, palette)
                renderer.deform(matrices, 0)
                result = renderer.read_vertices()
                self.assertLess(np.abs(result[:, :3] - expected).max(), 2e-5)
                self.assertLess(np.abs(result[:, 4:7] - smooth_normals(expected, asset["offsets"], asset["neighbors"])).max(), 2e-3)
                pixels = self.drawn_pixels(renderer)
                self.assertGreater(len(pixels), 20)
                if textured:
                    # Red base-colour texture through the textured vertex path.
                    self.assertTrue(np.all(pixels[:, 0] > pixels[:, 1]))
                else:
                    self.assertTrue(np.all(pixels[:, 1] > pixels[:, 0]))
            self.each_backend(CharacterRenderer, check)


if __name__ == "__main__":
    unittest.main()
