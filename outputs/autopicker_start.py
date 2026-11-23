"""Start the external Autopicker modules in Isaac Sim or with its Python launcher."""
import asyncio
from pathlib import Path
import sys

SOURCE_FILES = {
    'controller': 'autopicker_apple_pick.py',
    'fusion': 'autopicker_depth_fusion.py',
    'viewer': 'autopicker_camera_viewer.py',
    'runtime': 'autopicker_runtime.py',
}


def load_sources(directory):
    directory = Path(directory).resolve()
    compiled = {}
    for name, filename in SOURCE_FILES.items():
        path = directory / filename
        compiled[name] = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    namespaces = {}
    for name, code in compiled.items():
        namespace = {'__name__': f'autopicker_external_{name}',
                     '__file__': str(directory / SOURCE_FILES[name])}
        if name == 'viewer':
            namespace.update({key: namespaces['fusion'][key] for key in
                              ('DepthFusion', 'PointCloudRaster', 'VoxelReconstruction')
                              if key in namespaces['fusion']})
        exec(code, namespace)
        namespaces[name] = namespace
    return namespaces


async def _launch(scene_path, modules, config, previous_task, previous_runtime):
    import builtins
    import omni.kit.app
    import omni.timeline
    import omni.usd

    if previous_task is not None and not previous_task.done():
        previous_task.cancel()
        await asyncio.gather(previous_task, return_exceptions=True)
    if previous_runtime is not None:
        previous_runtime.close()
        if previous_runtime.task is not None:
            await asyncio.gather(previous_runtime.task, return_exceptions=True)
    context = omni.usd.get_context()
    omni.timeline.get_timeline_interface().stop()
    await omni.kit.app.get_app().next_update_async()
    stage = context.get_stage()
    if stage is None or Path(stage.GetRootLayer().realPath).resolve() != scene_path:
        success, message = await context.open_stage_async(str(scene_path))
        if not success:
            raise RuntimeError(f'Cannot open Autopicker scene: {message}')
    stage = context.get_stage()
    if stage.GetPrimAtPath('/World/ApplePickingGraph'):
        raise ValueError('Use the development USD with autopicker_start.py; the packaged demo starts on Play.')
    runtime = modules['runtime']['AutopickerRuntime'](modules['controller'], modules['viewer'], config)
    builtins.autopicker_runtime = runtime
    try:
        await runtime.start()
    except BaseException:
        runtime.close()
        raise
    return runtime


def launch(scene_path=None, config=None):
    import builtins

    directory = Path(__file__).resolve().parent
    scene_path = Path(scene_path or directory / 'autopicker_apple_orchard.usd').resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    modules = load_sources(directory)
    previous_task = getattr(builtins, 'autopicker_launch_task', None)
    previous_runtime = getattr(builtins, 'autopicker_runtime', None)
    task = asyncio.ensure_future(_launch(scene_path, modules, config, previous_task, previous_runtime))
    builtins.autopicker_launch_task = task
    return task


def main():
    if 'omni.kit.app' in sys.modules:
        return launch()
    import argparse
    import builtins

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0, help='Exit after this many updates; 0 keeps running.')
    options = parser.parse_args()
    if options.frames < 0:
        parser.error('--frames must be nonnegative')
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': options.headless, 'hide_ui': False,
                         'width': 1440, 'height': 900})
    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
    try:
        task = launch(options.scene)
        updates = 0
        while app.is_running() and (not options.frames or updates < options.frames):
            app.update()
            updates += 1
            if task.done():
                runtime = task.result()
                if runtime.error:
                    raise RuntimeError(runtime.error)
        if options.frames:
            if not task.done():
                raise RuntimeError('Startup did not finish within --frames updates')
            runtime = task.result()
            if runtime.closed or runtime.demo is None or not runtime.demo.running:
                raise RuntimeError('Autopicker did not start')
            print('AUTOPICKER_EXTERNAL_STARTED', flush=True)
    except Exception as error:
        import traceback
        print(f'AUTOPICKER_START_FAILED: {error}', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        runtime = getattr(builtins, 'autopicker_runtime', None)
        if runtime is not None:
            runtime.close()
        pending = getattr(builtins, 'autopicker_launch_task', None)
        if pending is not None and not pending.done():
            pending.cancel()
        app.close()


if __name__ == '__main__':
    main()
