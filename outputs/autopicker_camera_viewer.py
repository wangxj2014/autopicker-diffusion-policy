"""Four live RGB/depth views and synchronized world point clouds in Isaac Sim 5.1.

Development startup loads this module; the demo packager embeds the same source.
Run it in Script Editor to reopen the panel without restarting picking.
Each tile copies complete LdrColor frames into its
own UI buffer; GPU render textures are never handed directly to the UI.
"""
CAMERA_ROOT = '/World/Autopicker/root_joint'
CAMERA_ORDER = ('L_up', 'R_up', 'L_down', 'R_down')
CAMERA_LABELS = {'L_up': 'LEFT / UPPER', 'R_up': 'RIGHT / UPPER',
                 'L_down': 'LEFT / LOWER', 'R_down': 'RIGHT / LOWER'}
RESOLUTION = (640, 480)
VIEWER_VERSION = 6
DISPLAY_FPS = 15
DEPTH_NEAR_M = 0.1
DEPTH_FAR_M = 5.0


def _fusion_types(stage):
    if 'DepthFusion' in globals():
        return globals()['DepthFusion'], globals()['PointCloudRaster'], globals()['VoxelReconstruction']
    from pathlib import Path
    directory = Path(__file__).resolve().parent if '__file__' in globals() else Path(stage.GetRootLayer().realPath).parent
    source = directory / 'autopicker_depth_fusion.py'
    namespace = {'__name__': 'autopicker_depth_fusion'}
    exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
    return namespace['DepthFusion'], namespace['PointCloudRaster'], namespace['VoxelReconstruction']


def _orchard_reconstruction_bounds(stage):
    import itertools
    import numpy as np
    from pxr import UsdGeom

    orchard = stage.GetPrimAtPath('/World/Orchard')
    if not orchard:
        return [], np.eye(4), None
    units = UsdGeom.GetStageMetersPerUnit(stage)
    world_from_region = np.array(UsdGeom.Xformable(orchard).ComputeLocalToWorldTransform(0))
    world_from_region[3, :3] *= units
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    regions = []
    corners = []
    for name in ('Left', 'Right'):
        side = orchard.GetChild(name)
        if not side:
            continue
        bounds = cache.ComputeRelativeBound(side, orchard).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        lower = np.asarray(bounds.GetMin()) * units
        upper = np.asarray(bounds.GetMax()) * units
        lower[:2] -= .05
        upper[:2] += .05
        lower[2] += .02
        upper[2] += .05
        regions.append((lower, upper))
        corners.extend(itertools.product(*zip(lower, upper)))
    world_corners = (np.asarray(corners) @ world_from_region[:3, :3] + world_from_region[3, :3]
                     if corners else None)
    return regions, np.linalg.inv(world_from_region), world_corners


def _make_sync_writer(products):
    import omni.replicator.core as rep

    class AutopickerDepthWriter(rep.Writer):
        def __init__(self):
            self.version = '2.0.0'
            self.data_structure = 'renderProduct'
            self.annotators = ['rgb', 'distance_to_image_plane', 'CameraParams']

        def write_metadata(self):
            pass

        def write(self, data):
            pass

    writer = AutopickerDepthWriter()
    writer.attach(products, trigger=None)
    return writer


def depth_to_rgba(depth_m):
    import numpy as np

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    clipped = np.clip(np.where(valid, depth, DEPTH_FAR_M), DEPTH_NEAR_M, DEPTH_FAR_M)
    gray = np.rint((DEPTH_FAR_M - clipped) * (255.0 / (DEPTH_FAR_M - DEPTH_NEAR_M))).astype(np.uint8)
    rgba = np.empty((*depth.shape, 4), dtype=np.uint8)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 255
    return rgba


def _make_rgb_display(viewport_api):
    import ctypes
    import time
    import carb
    import numpy as np
    import omni.ui as ui
    from omni.kit.widget.viewport.display_delegate import ViewportDisplayDelegate
    from omni.kit.widget.viewport.capture import ByteCapture
    from omni.gpu_foundation_factory import TextureFormat

    capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_pointer.restype = ctypes.c_void_p
    capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    class RgbDisplay(ViewportDisplayDelegate):
        def __init__(self):
            super().__init__(viewport_api)
            self._closed = False
            self._pending = False
            self._next_frame = 0.0
            self._reported_error = False
            self.frames_presented = 0

        def create(self, frame, previous, *args, **kwargs):
            stack = super().create(frame, None)
            with stack:
                self.image_provider = ui.ByteImageProvider()
            self._closed = False
            return stack

        def update(self, viewport_api, texture, view, projection,
                   presentation_key=0, metadata=None):
            # Read back only the named RGB AOV. Keep displaying the last complete
            # image until a new copy arrives; never expose a reused GPU resource.
            now = time.monotonic()
            if not self._closed and not self._pending and now >= self._next_frame:
                self._pending = True
                self._next_frame = now + 1.0 / DISPLAY_FPS
                viewport_api.schedule_capture(ByteCapture(self._on_rgb, aov_name='LdrColor'))
            return self.size

        def _on_rgb(self, buffer, size, width, height, pixel_format):
            try:
                if self._closed:
                    return
                if pixel_format != TextureFormat.RGBA8_UNORM or size != width * height * 4:
                    raise ValueError(f'Unexpected RGB image format: {pixel_format}, {size} bytes')
                raw = (ctypes.c_uint8 * size).from_address(capsule_pointer(buffer, None))
                pixels = np.ctypeslib.as_array(raw).copy()
                pixels[3::4] = 255
                self.image_provider.set_data_array(pixels, [width, height])
                self.frames_presented += 1
            except Exception as error:
                if not self._reported_error:
                    carb.log_error(f'Autopicker RGB display: {error}')
                    self._reported_error = True
            finally:
                self._pending = False

        def destroy(self):
            self._closed = True
            super().destroy()

    return RgbDisplay()


class AutopickerCameraViewer:
    def __init__(self):
        import omni.ui as ui
        import omni.usd
        from omni.kit.widget.viewport import ViewportWidget

        self.stage = omni.usd.get_context().get_stage()
        self.paths = {arm: f'{CAMERA_ROOT}/{arm}_cam_link/Camera' for arm in CAMERA_ORDER}
        from pxr import UsdGeom
        if self.stage is None or any(not self.stage.GetPrimAtPath(path).IsA(UsdGeom.Camera)
                                     for path in self.paths.values()):
            raise ValueError('Open the USD containing the four cam_link/Camera sensors first.')
        self.widgets = {}
        self.depth_providers = {}
        self.depth_images = {}
        self.depth_annotators = {}
        self.depth_products = {}
        self.depth_m = {}
        self.depth_rgba = {}
        self.depth_frames = {arm: 0 for arm in CAMERA_ORDER}
        self.depth_error = None
        self.version = VIEWER_VERSION
        self.closed = False
        self._display_task = None
        self._stage_subscription = None
        self._depth_subscription = None
        self._timeline_subscription = None
        self._next_depth_frame = 0.0
        self._sync_writer = None
        self._last_payload = None
        self._last_fusion_time = None
        self.fusion_hz = 0.0
        self.fusion_ms = 0.0
        self.fusion_error = None
        self._cloud_dirty = False
        self.cloud_controls = {}
        fusion_type, raster_type, reconstruction_type = _fusion_types(self.stage)
        self.fusion = fusion_type(self.paths)
        self.cloud_raster = raster_type()
        regions, region_from_world, view_bounds = _orchard_reconstruction_bounds(self.stage)
        self.reconstruction = reconstruction_type(regions=regions, region_from_world=region_from_world)
        if view_bounds is not None:
            self.cloud_raster.fit(view_bounds)
        panel_width = min(640, max(360, (ui.Workspace.get_main_window_width() - 48) / 3))
        panel_height = panel_width * .75 + 130
        self.window = ui.Window('Autopicker Live Cameras', width=panel_width, height=panel_height,
                                flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)
        self.window.position_x = 12
        self.window.position_y = 80
        with self.window.frame:
            with ui.VStack(spacing=6, style={'background_color': 0xFF202020}):
                ui.Label('LIVE RGB  |  640 x 480 per camera  |  up to 15 fps', height=24,
                         style={'font_size': 16, 'color': 0xFFE6E6E6})
                for row in (CAMERA_ORDER[:2], CAMERA_ORDER[2:]):
                    with ui.HStack(spacing=6):
                        for arm in row:
                            with ui.VStack(spacing=3):
                                ui.Label(f'{CAMERA_LABELS[arm]}   ({arm})', height=22,
                                         style={'font_size': 15, 'color': 0xFFBBDDBB})
                                widget = ViewportWidget(camera_path=self.paths[arm],
                                                        resolution=RESOLUTION)
                                widget.viewport_api.fill_frame = False
                                widget.viewport_api.display_render_var = 'LdrColor'
                                self.widgets[arm] = widget
                ui.Label('Cameras follow the arms. Pause freezes motion; Stop + Play resets picking.',
                         height=22, style={'font_size': 13, 'color': 0xFFBBBBBB})
        self.window.set_visibility_changed_fn(self._on_visibility)
        self.depth_window = ui.Window('Autopicker Live Depth', width=panel_width, height=panel_height,
                                      flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)
        self.depth_window.position_x = panel_width + 24
        self.depth_window.position_y = 80
        with self.depth_window.frame:
            with ui.VStack(spacing=6, style={'background_color': 0xFF202020}):
                ui.Label('LIVE DEPTH  |  640 x 480  |  meters  |  up to 15 fps', height=24,
                         style={'font_size': 16, 'color': 0xFFE6E6E6})
                for row in (CAMERA_ORDER[:2], CAMERA_ORDER[2:]):
                    with ui.HStack(spacing=6):
                        for arm in row:
                            with ui.VStack(spacing=3):
                                ui.Label(f'{CAMERA_LABELS[arm]}   ({arm})', height=22,
                                         style={'font_size': 15, 'color': 0xFFBBDDBB})
                                provider = ui.ByteImageProvider()
                                provider.set_bytes_data([0, 0, 0, 255] * (RESOLUTION[0] * RESOLUTION[1]),
                                                        list(RESOLUTION))
                                self.depth_providers[arm] = provider
                                self.depth_images[arm] = ui.ImageWithProvider(
                                    provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label('Near 0.1 m = white  |  Far 5 m / invalid = black  |  Fixed scale',
                         height=22, style={'font_size': 13, 'color': 0xFFBBBBBB})
        self.depth_window.set_visibility_changed_fn(self._on_visibility)
        self.cloud_window = ui.Window('Autopicker Fused Point Cloud', width=panel_width, height=panel_height,
                                      flags=ui.WINDOW_FLAGS_NO_SCROLLBAR)
        self.cloud_window.position_x = 2 * panel_width + 36
        self.cloud_window.position_y = 80
        with self.cloud_window.frame:
            with ui.VStack(spacing=5, style={'background_color': 0xFF201611}):
                ui.Label('TREE RECONSTRUCTION | Fixed world / meters', height=26,
                         style={'font_size': 14, 'color': 0xFFE6E6E6})
                self.cloud_status = ui.Label('Waiting for repeated observations...', height=22)
                self.cloud_provider = ui.ByteImageProvider()
                self.cloud_rgba = self.cloud_raster.render(None)
                self.cloud_provider.set_data_array(self.cloud_rgba.ravel(),
                                                   [self.cloud_raster.width, self.cloud_raster.height])
                self.cloud_image = ui.ImageWithProvider(
                    self.cloud_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label('RGB colors | Synchronized pixels | Confirmed colors retained', height=22)
                ui.Label('Accumulated | Fixed view | 1 cm | Two-frame confirmation', height=20)
                with ui.HStack(height=26, spacing=6):
                    for label, attribute, minimum, maximum in (
                            ('Yaw', 'yaw', -180, 180), ('Tilt', 'elevation', -85, 85),
                            ('Zoom', 'zoom', .25, 4)):
                        ui.Label(label, width=36)
                        control = ui.FloatDrag(min=minimum, max=maximum,
                                               step=.02 if attribute == 'zoom' else .5)
                        control.model.set_value(getattr(self.cloud_raster, attribute))
                        self.cloud_controls[attribute] = control.model
                        control.model.add_value_changed_fn(
                            lambda model, name=attribute: self._cloud_view_changed(name, model.as_float))
                    ui.Button('Fit', width=55, clicked_fn=self._fit_cloud)
                ui.Button('Clear reconstruction / restart scan', height=24, clicked_fn=self.clear_reconstruction)
        self.cloud_window.set_visibility_changed_fn(self._on_visibility)
        self._stage_subscription = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
            self._on_stage)
        import omni.timeline
        self._timeline_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline)
        import asyncio
        self._display_task = asyncio.ensure_future(self._initialize_displays())

    async def _initialize_displays(self):
        import asyncio
        import omni.kit.app
        # Wait for the initial render resources as well as the UI layout.
        await asyncio.gather(*(widget.viewport_api.wait_for_rendered_frames(2)
                               for widget in self.widgets.values()))
        await omni.kit.app.get_app().next_update_async()
        if not self.closed:
            for widget in self.widgets.values():
                widget.display_delegate = _make_rgb_display(widget.viewport_api)
            try:
                import omni.replicator.core as rep
                for arm, widget in self.widgets.items():
                    product = widget.viewport_api.render_product_path
                    annotator = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane', device='cpu')
                    annotator.attach([product])
                    self.depth_products[arm] = product
                    self.depth_annotators[arm] = annotator
                self._sync_writer = _make_sync_writer(list(self.depth_products.values()))
                self._depth_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                    self._update_depth, name='autopicker_depth_display')
            except Exception as error:
                self.depth_error = str(error)
                import carb
                carb.log_error(f'Autopicker depth initialization: {error}')

    def _update_depth(self, event):
        import time
        import numpy as np
        import omni.timeline

        now = time.monotonic()
        if self.closed or not self._any_visible() or now < self._next_depth_frame:
            return
        self._next_depth_frame = now + 1.0 / DISPLAY_FPS
        try:
            if self._cloud_dirty:
                self._draw_cloud()
            if omni.timeline.get_timeline_interface().is_stopped():
                self.cloud_status.text = 'Stopped | Play starts a new reconstruction'
                return
            payload = self._sync_writer.get_data() if self._sync_writer is not None else None
            if payload is None or payload is self._last_payload:
                if not omni.timeline.get_timeline_interface().is_playing():
                    self.cloud_status.text = 'Paused | Reconstruction retained | View controls active'
                elif self._last_fusion_time is None or now - self._last_fusion_time > 1:
                    self.cloud_status.text = 'Waiting for four synchronized frames | Map retained'
                return
            started = time.perf_counter()
            if not self.fusion.update(payload):
                self.cloud_status.text = 'Incomplete camera batch | Reconstruction retained'
                return
            self._last_payload = payload
            self.reconstruction.update(self.fusion.cloud)
            for arm, frame in self.fusion.frames.items():
                depth = frame['depth_m']
                rgba = depth_to_rgba(depth)
                if self.depth_window.visible:
                    self.depth_providers[arm].set_data_array(rgba.ravel(), list(RESOLUTION))
                self.depth_m[arm] = depth
                self.depth_rgba[arm] = rgba
                self.depth_frames[arm] += 1
            self._draw_cloud()
            self.fusion_ms = (time.perf_counter() - started) * 1000
            if self._last_fusion_time is not None:
                rate = 1 / max(now - self._last_fusion_time, 1e-6)
                self.fusion_hz = rate if self.fusion_hz == 0 else .85 * self.fusion_hz + .15 * rate
            self._last_fusion_time = now
            cloud = self.reconstruction.cloud
            if cloud is not None:
                state = 'MAP FULL - Clear to rescan' if cloud['capacity_reached'] else f"{cloud['integrated_frames']} scans"
                self.cloud_status.text = (f"{len(cloud['points_world_m']):,} fixed pts | {state} | "
                                          f"{self.fusion_hz:.1f} Hz")
            self.fusion_error = None
        except Exception as error:
            if self.depth_error is None:
                import carb
                carb.log_error(f'Autopicker depth display: {error}')
            self.depth_error = str(error)
            self.fusion_error = str(error)
            self.cloud_status.text = f'Capture error: {error}'

    def _cloud_view_changed(self, attribute, value):
        setattr(self.cloud_raster, attribute, value)
        self._cloud_dirty = True

    def _fit_cloud(self):
        if self.reconstruction.cloud is not None:
            self.cloud_raster.fit(self.reconstruction.cloud['points_world_m'])
            self.cloud_controls['zoom'].set_value(self.cloud_raster.zoom)
        self._cloud_dirty = True

    def _draw_cloud(self):
        if self.cloud_window.visible:
            self.cloud_rgba = self.cloud_raster.render(self.reconstruction.cloud)
            self.cloud_provider.set_data_array(self.cloud_rgba.ravel(),
                                               [self.cloud_raster.width, self.cloud_raster.height])
        self._cloud_dirty = False

    def get_depth_frame(self, arm):
        """Copy depth, intrinsics, rendered camera pose and shared frame metadata."""
        return self.fusion.get_depth_frame(arm)

    def get_point_cloud(self):
        """Copy the accumulated world map with its latest scan and per-point first-seen IDs."""
        return self.reconstruction.get_point_cloud()

    def get_live_point_cloud(self):
        """Copy only the latest synchronized camera cloud before accumulation or region filtering."""
        return self.fusion.get_point_cloud()

    def clear_reconstruction(self):
        """Start an empty reconstruction without restarting picking or changing the view."""
        self.reconstruction.reset()
        self._last_payload = self._sync_writer.get_data() if self._sync_writer else None
        self._cloud_dirty = True
        self._draw_cloud()
        self.cloud_status.text = 'Map cleared | Waiting for repeated observations'

    def _on_timeline(self, event):
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.fusion.reset()
            self.clear_reconstruction()
            self.depth_m.clear()
            self._last_payload = self._sync_writer.get_data() if self._sync_writer else None
            self._last_fusion_time = None
            self.fusion_hz = 0.0
            self._cloud_dirty = True
            self.cloud_status.text = 'Stopped | Play starts a new reconstruction'

    def get_depth(self, arm):
        """Return an independent HxW float32 optical-axis depth image in meters, or None during warmup."""
        if arm not in self.paths:
            raise KeyError(arm)
        data = self.depth_m.get(arm)
        return None if data is None else data.copy()

    def _on_visibility(self, visible):
        enabled = self._any_visible()
        for widget in self.widgets.values():
            widget.viewport_api.updates_enabled = enabled
        self._cloud_dirty = True

    def _any_visible(self):
        return any(window is not None and window.visible for window in
                   (self.window, getattr(self, 'depth_window', None), getattr(self, 'cloud_window', None)))

    def _on_stage(self, event):
        import omni.usd
        if event.type == int(omni.usd.StageEventType.CLOSING):
            self.close()

    def show(self):
        if self.closed:
            raise RuntimeError('This camera viewer has closed. Call show_camera_viewer() to create a new one.')
        self.window.visible = True
        self.depth_window.visible = True
        self.cloud_window.visible = True
        self._on_visibility(True)
        return self

    def close(self):
        if self.closed:
            return
        import omni.usd
        context = omni.usd.get_context()
        closing_stage = (context.get_stage_state() == omni.usd.StageState.CLOSING
                         or context.get_stage() != self.stage)
        self.closed = True
        if self._display_task is not None:
            self._display_task.cancel()
            self._display_task = None
        self._stage_subscription = None
        self._depth_subscription = None
        self._timeline_subscription = None
        self.window.set_visibility_changed_fn(None)
        self.depth_window.set_visibility_changed_fn(None)
        self.cloud_window.set_visibility_changed_fn(None)
        if self._sync_writer is not None:
            if not closing_stage:
                self._sync_writer.detach()
            self._sync_writer = None
        self._last_payload = None
        self.fusion.reset()
        self.reconstruction.reset()
        for arm, annotator in self.depth_annotators.items():
            if not closing_stage:
                annotator.detach([self.depth_products[arm]])
        self.depth_annotators.clear()
        self.depth_products.clear()
        self.depth_m.clear()
        self.depth_rgba.clear()
        self.depth_images.clear()
        self.depth_providers.clear()
        for widget in self.widgets.values():
            widget.destroy()
        self.widgets.clear()
        self.window.destroy()
        self.window = None
        self.depth_window.destroy()
        self.depth_window = None
        self.cloud_window.destroy()
        self.cloud_window = None
        self.cloud_image = None
        self.cloud_provider = None
        self.cloud_controls.clear()


def show_camera_viewer():
    import builtins
    import omni.usd
    existing = getattr(builtins, 'autopicker_camera_viewer', None)
    stage = omni.usd.get_context().get_stage()
    if existing is not None and not existing.closed:
        if existing.stage == stage and getattr(existing, 'version', None) == VIEWER_VERSION:
            return existing.show()
        existing.close()
    viewer = AutopickerCameraViewer()
    builtins.autopicker_camera_viewer = viewer
    return viewer


if __name__ == '__main__':
    show_camera_viewer()
