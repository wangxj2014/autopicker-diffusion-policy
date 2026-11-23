"""Build a standalone demonstration USD from the development scene and sibling sources."""
from pathlib import Path

SOURCE_FILES = {
    'CONTROLLER_SOURCE': 'autopicker_apple_pick.py',
    'FUSION_SOURCE': 'autopicker_depth_fusion.py',
    'VIEWER_SOURCE': 'autopicker_camera_viewer.py',
    'RUNTIME_SOURCE': 'autopicker_runtime.py',
}

GRAPH_ADAPTER = '''
def setup(db):
    import builtins
    previous = getattr(builtins, 'autopicker_runtime', None)
    if previous is not None:
        previous.close()
    controller = {'__name__': 'autopicker_embedded_controller'}
    exec(CONTROLLER_SOURCE, controller)
    viewer = {'__name__': 'autopicker_embedded_viewer'}
    exec(FUSION_SOURCE, viewer)
    exec(VIEWER_SOURCE, viewer)
    runtime_module = {'__name__': 'autopicker_embedded_runtime'}
    exec(RUNTIME_SOURCE, runtime_module)
    runtime = runtime_module['AutopickerRuntime'](controller, viewer)
    db.per_instance_state.runtime = runtime
    builtins.autopicker_runtime = runtime


def compute(db):
    runtime = db.per_instance_state.runtime
    if runtime.task is None and not runtime.closed:
        runtime.start()
    return True


def cleanup(db):
    runtime = getattr(db.per_instance_state, 'runtime', None)
    if runtime is not None:
        runtime.close()
'''


def build_demo(scene_path, output_path, source_dir=None):
    from pxr import Gf, Sdf, Usd, UsdUtils

    scene_path = Path(scene_path).resolve()
    output_path = Path(output_path).resolve()
    if scene_path == output_path:
        raise ValueError('The demo output must differ from the development scene')
    source_dir = Path(source_dir or Path(__file__).resolve().parent)
    sources = {}
    for name, filename in SOURCE_FILES.items():
        source = (source_dir / filename).read_text(encoding='utf-8')
        compile(source, filename, 'exec')
        sources[name] = source
    script = '\n'.join(f'{name} = {source!r}' for name, source in sources.items()) + '\n' + GRAPH_ADAPTER
    compile(script, 'autopicker_demo', 'exec')
    original = Usd.Stage.Open(str(scene_path))
    if original is None:
        raise ValueError(f'Cannot open development scene: {scene_path}')
    if original.GetPrimAtPath('/World/ApplePickingGraph'):
        raise ValueError('The input must be the development scene without an embedded graph')
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene_path))
    if len(layers) != 1 or assets or unresolved or original.GetRootLayer().subLayerPaths:
        raise ValueError('The demo packager requires a self-contained scene without external layers or assets')
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(original.GetRootLayer())
    stage = Usd.Stage.Open(layer)
    graph = stage.DefinePrim('/World/ApplePickingGraph', 'OmniGraph')
    for name, value in [('evaluationMode', 'Automatic'), ('evaluator:type', 'execution'),
                        ('fabricCacheBacking', 'StageWithoutHistory'), ('pipelineStage', 'pipelineStageSimulation')]:
        graph.CreateAttribute(name, Sdf.ValueTypeNames.Token, custom=False).Set(value)
    graph.CreateAttribute('fileFormatVersion', Sdf.ValueTypeNames.Int2, custom=False).Set(Gf.Vec2i(1, 9))
    tick = stage.DefinePrim(graph.GetPath().AppendChild('Tick'), 'OmniGraphNode')
    tick.CreateAttribute('node:type', Sdf.ValueTypeNames.Token, custom=False).Set('omni.graph.action.OnPlaybackTick')
    tick.CreateAttribute('node:typeVersion', Sdf.ValueTypeNames.Int, custom=False).Set(2)
    tick_output = tick.CreateAttribute('outputs:tick', Sdf.ValueTypeNames.UInt)
    tick_output.SetCustomDataByKey('isExecution', True)
    node = stage.DefinePrim(graph.GetPath().AppendChild('PickApples'), 'OmniGraphNode')
    node.CreateAttribute('node:type', Sdf.ValueTypeNames.Token, custom=False).Set('omni.graph.scriptnode.ScriptNode')
    node.CreateAttribute('node:typeVersion', Sdf.ValueTypeNames.Int, custom=False).Set(2)
    execution = node.CreateAttribute('inputs:execIn', Sdf.ValueTypeNames.UInt)
    execution.SetCustomDataByKey('isExecution', True)
    execution.AddConnection(tick_output.GetPath())
    node.CreateAttribute('inputs:usePath', Sdf.ValueTypeNames.Bool).Set(False)
    node.CreateAttribute('inputs:script', Sdf.ValueTypeNames.String).Set(script)
    node.CreateAttribute('state:omni_initialized', Sdf.ValueTypeNames.Bool).Set(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f'Cannot export demonstration USD: {output_path}')
    return output_path


def main():
    import argparse

    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=directory / 'autopicker_apple_orchard.usd')
    parser.add_argument('--output', type=Path, default=directory / 'dist/autopicker_apple_orchard_demo.usd')
    options = parser.parse_args()
    print(build_demo(options.scene, options.output, directory))


if __name__ == '__main__':
    main()
