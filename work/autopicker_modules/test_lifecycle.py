from pathlib import Path
import sys
import traceback
import json
import asyncio
import runpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs'))
from isaacsim import SimulationApp

app = SimulationApp({'headless': True, 'hide_ui': False, 'width': 1440, 'height': 900})
try:
    import builtins
    import omni.usd
    import omni.timeline
    from autopicker_start import launch

    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()

    def wait_for(predicate, limit=140):
        for _ in range(limit):
            app.update()
            if predicate():
                return
        raise AssertionError('Timed out waiting for runtime state')

    def complete(task):
        wait_for(task.done)
        return task.result()

    runpy.run_path(str(ROOT / 'outputs/autopicker_start.py'), run_name='__main__')
    first_task = builtins.autopicker_launch_task
    second_task = launch()
    runtime = complete(second_task)
    assert first_task.cancelled(), 'Repeated startup must cancel the previous pending launch'
    assert runtime.demo.running and not runtime.closed
    assert not runtime.stage.GetPrimAtPath('/World/ApplePickingGraph')
    wait_for(lambda: runtime.viewer.get_point_cloud() is not None)
    demo = runtime.demo
    viewer = runtime.viewer
    timeline.pause()
    for _ in range(8):
        app.update()
    frozen = viewer.get_point_cloud()
    timeline.play()
    for _ in range(8):
        app.update()
    assert runtime.demo is demo and runtime.viewer is viewer
    assert viewer.get_point_cloud()['map_epoch'] == frozen['map_epoch']
    timeline.stop()
    for _ in range(5):
        app.update()
    assert not demo.running and demo._physics_subscription is None
    assert viewer.get_point_cloud() is None
    timeline.play()
    wait_for(lambda: runtime.demo is not demo and runtime.demo.running)
    assert runtime.viewer is viewer and not viewer.closed
    assert not runtime.error
    replacement = complete(launch(config={'grasp_error_every': 0}))
    assert runtime.closed and viewer.closed
    assert runtime.demo._physics_subscription is None
    assert replacement.demo.config['grasp_error_every'] == 0
    assert replacement.viewer is not viewer
    assert replacement.controller is not runtime.controller
    assert replacement.viewer_module is not runtime.viewer_module
    wait_for(lambda: replacement.viewer.get_point_cloud() is not None)
    second_viewer = replacement.viewer
    context.close_stage()
    for _ in range(6):
        app.update()
    assert replacement.closed and second_viewer.closed
    assert replacement.demo._physics_subscription is None
    startup = launch()
    wait_for(lambda: getattr(builtins, 'autopicker_runtime', None) is not replacement
             and not builtins.autopicker_runtime.closed)
    interrupted = builtins.autopicker_runtime
    assert not startup.done(), 'Test must close stage during initialization'
    context.close_stage()
    wait_for(startup.done)
    assert interrupted.closed and interrupted.viewer is None
    assert startup.cancelled()
    final_runtime = complete(launch())
    assert final_runtime.demo.running and not final_runtime.closed
    context.close_stage()
    for _ in range(5):
        app.update()
    assert final_runtime.closed
    result = {'duplicate_launch_cancelled': True, 'pause_resumes_same_controller_and_map': True,
              'stop_play_restarts_controller_and_keeps_windows': True,
              'explicit_launch_reloads_modules_and_config': True,
              'stage_close_cleans_up': True, 'close_during_initialization_cancels_start': True,
              'relaunch_after_cancellation': True}
    (ROOT / 'work/autopicker_modules/lifecycle.json').write_text(json.dumps(result, indent=2))
    print('EXTERNAL_LIFECYCLE_PASSED', json.dumps(result), flush=True)
except BaseException:
    traceback.print_exc()
    app.close()
    sys.exit(1)
app.close()
