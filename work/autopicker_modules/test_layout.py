from pathlib import Path
import ast
import hashlib
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'work/python'), str(ROOT / 'outputs')]
from pxr import Usd, UsdGeom, Sdf


class ModuleLayoutTests(unittest.TestCase):
    def test_development_scene_has_no_executable_graph(self):
        stage = Usd.Stage.Open(str(ROOT / 'outputs/autopicker_apple_orchard.usd'))
        self.assertFalse(stage.GetPrimAtPath('/World/ApplePickingGraph'))
        self.assertEqual(sum(prim.IsA(UsdGeom.Camera) for prim in stage.Traverse()), 4)
        self.assertFalse(any(prim.GetAttribute('inputs:script') for prim in stage.Traverse()))

    def test_scene_data_is_preserved_outside_removed_graph(self):
        original = ROOT / 'work/autopicker_modules/original.usd'
        if not original.is_file():
            self.skipTest('Local migration baseline is not included in repository clones')
        before = Usd.Stage.Open(str(original))
        after = Usd.Stage.Open(str(ROOT / 'outputs/autopicker_apple_orchard.usd'))
        before.RemovePrim('/World/ApplePickingGraph')
        self.assertEqual(before.GetRootLayer().customLayerData, after.GetRootLayer().customLayerData)
        self.assertEqual([prim.GetPath() for prim in before.TraverseAll()],
                         [prim.GetPath() for prim in after.TraverseAll()])
        for prim in before.TraverseAll():
            current = after.GetPrimAtPath(prim.GetPath())
            self.assertEqual(prim.GetAllMetadata(), current.GetAllMetadata(), str(prim.GetPath()))
            self.assertEqual(prim.GetPropertyNames(), current.GetPropertyNames(), str(prim.GetPath()))
            for prop in prim.GetProperties():
                self.assertEqual(prop.GetAllMetadata(), current.GetProperty(prop.GetName()).GetAllMetadata(),
                                 str(prop.GetPath()))

    def test_packager_reads_current_sources_and_keeps_development_scene(self):
        self.assertIsNotNone(importlib.util.find_spec('autopicker_package'))
        from autopicker_package import build_demo
        scene = ROOT / 'outputs/autopicker_apple_orchard.usd'
        before_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'demo.usd'
            build_demo(scene, output, ROOT / 'outputs')
            stage = Usd.Stage.Open(str(output))
            node = stage.GetPrimAtPath('/World/ApplePickingGraph/PickApples')
            self.assertTrue(node)
            source = node.GetAttribute('inputs:script').Get()
            compile(source, 'packaged', 'exec')
            assignments = {item.targets[0].id: ast.literal_eval(item.value)
                           for item in ast.parse(source).body if isinstance(item, ast.Assign)
                           and isinstance(item.targets[0], ast.Name) and isinstance(item.value, ast.Constant)}
            for name, filename in [('CONTROLLER_SOURCE', 'autopicker_apple_pick.py'),
                                   ('VIEWER_SOURCE', 'autopicker_camera_viewer.py'),
                                   ('FUSION_SOURCE', 'autopicker_depth_fusion.py'),
                                   ('RUNTIME_SOURCE', 'autopicker_runtime.py')]:
                self.assertEqual(assignments[name], (ROOT / 'outputs' / filename).read_text(encoding='utf-8'))
            self.assertFalse(stage.GetRootLayer().subLayerPaths)
        self.assertEqual(hashlib.sha256(scene.read_bytes()).hexdigest(), before_hash)
        with self.assertRaises(ValueError):
            build_demo(scene, scene, ROOT / 'outputs')

    def test_loader_reloads_source_and_rejects_missing_files(self):
        self.assertIsNotNone(importlib.util.find_spec('autopicker_start'))
        from autopicker_start import load_sources
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('apple_pick', 'camera_viewer', 'depth_fusion', 'runtime'):
                (folder / f'autopicker_{name}.py').write_text('value = 1\n')
            loaded = load_sources(folder)
            self.assertEqual(loaded['controller']['value'], 1)
            (folder / 'autopicker_apple_pick.py').write_text('value = 2\n')
            self.assertEqual(load_sources(folder)['controller']['value'], 2)
            (folder / 'autopicker_camera_viewer.py').unlink()
            with self.assertRaises(FileNotFoundError):
                load_sources(folder)

    def test_packager_rejects_scene_dependencies_before_writing_output(self):
        from autopicker_package import build_demo
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = Usd.Stage.CreateNew(str(folder / 'model.usda'))
            UsdGeom.Cube.Define(model, '/World/Cube')
            model.GetRootLayer().Save()
            scene = Usd.Stage.CreateNew(str(folder / 'scene.usda'))
            scene.GetRootLayer().subLayerPaths = ['model.usda']
            scene.GetRootLayer().Save()
            output = folder / 'dist/demo.usd'
            with self.assertRaisesRegex(ValueError, 'self-contained'):
                build_demo(folder / 'scene.usda', output, ROOT / 'outputs')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
