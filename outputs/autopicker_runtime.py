"""Lifecycle shared by external development startup and the generated USD demo."""
import asyncio


class AutopickerRuntime:
    def __init__(self, controller, viewer, config=None):
        import omni.timeline
        import omni.usd

        self.controller = controller
        self.viewer_module = viewer
        self.config = dict(config or {})
        self.stage = omni.usd.get_context().get_stage()
        self.demo = None
        self.viewer = None
        self.task = None
        self.error = None
        self.closed = False
        self.wants_start = True
        self._timeline_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(self._on_timeline)
        self._stage_subscription = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage)

    def start(self):
        if self.closed:
            raise RuntimeError('This Autopicker runtime is closed. Run the startup entry again.')
        if self.task is not None and not self.task.done():
            return self.task
        self.wants_start = False
        self.error = None
        self.task = asyncio.ensure_future(self._start())
        return self.task

    async def _start(self):
        import builtins

        try:
            self.demo = await self.controller['start_demo'](self.config)
            if self.closed or not self.demo.running:
                self.wants_start = not self.closed
                return self
            builtins.autopicker_apple_demo = self.demo
            self.viewer = self.viewer_module['show_camera_viewer']()
            return self
        except asyncio.CancelledError:
            raise
        except Exception as error:
            import carb
            self.error = str(error)
            self.wants_start = True
            demo = self.controller.get('autopicker_demo')
            if demo is not None:
                demo.stop()
            carb.log_error('AUTOPICKER_RUNTIME_FAILED: ' + self.error)
            raise

    def _on_timeline(self, event):
        import omni.timeline

        if self.closed or (self.task is not None and not self.task.done()):
            return
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.wants_start = True
            if self.demo is not None:
                self.demo.stop(pause=False)
        elif event.type == int(omni.timeline.TimelineEventType.PLAY) and self.wants_start:
            self.start()

    def _on_stage(self, event):
        import omni.usd

        if event.type == int(omni.usd.StageEventType.CLOSING):
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.wants_start = False
        self._timeline_subscription = None
        self._stage_subscription = None
        if self.task is not None and not self.task.done():
            self.task.cancel()
        demo = self.controller.get('autopicker_demo')
        if demo is not None:
            demo.stop(pause=False)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
