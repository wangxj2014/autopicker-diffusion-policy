"""Frame-synchronous RGB-D registration and a CPU point-cloud preview."""
import copy
import time

import numpy as np

def product_depth(product):
    data = product['distance_to_image_plane']
    if isinstance(data, dict):
        data = data['data']
    depth = np.asarray(data, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def product_rgb(product):
    data = product['rgb']
    if isinstance(data, dict):
        data = data['data']
    rgb = np.asarray(data)
    width, height = np.asarray(product['CameraParams']['renderProductResolution'], dtype=int)
    if (rgb.ndim != 3 or rgb.shape[:2] != (height, width)
            or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8):
        raise ValueError('Expected synchronized RGB/RGBA uint8 pixels at depth resolution')
    return rgb[..., :3]


def unproject_depth(product, stride=4, near_m=.1, far_m=5.0, with_rgb=False):
    """Return world points, intrinsics, camera-to-world, and optionally matching RGB colors."""
    depth = product_depth(product)
    params = product['CameraParams']
    width, height = np.asarray(params['renderProductResolution'], dtype=int)
    projection = np.asarray(params['cameraProjection'], dtype=np.float64).reshape(4, 4)
    view = np.asarray(params['cameraViewTransform'], dtype=np.float64).reshape(4, 4)
    units = float(params['metersPerSceneUnit'])
    if (params['cameraModel'] != 'pinhole' or depth.shape != (height, width)
            or not np.isfinite(projection).all() or not np.isfinite(view).all()
            or projection[0, 0] <= 0 or projection[1, 1] <= 0
            or not np.isfinite(units) or units <= 0 or stride < 1):
        raise ValueError('Invalid pinhole render calibration or depth dimensions')
    intrinsics = np.array([[width * projection[0, 0] / 2, 0,
                            width * (1 - projection[2, 0]) / 2 - .5],
                           [0, height * projection[1, 1] / 2,
                            height * (1 + projection[2, 1]) / 2 - .5],
                           [0, 0, 1]])
    world_from_camera = np.linalg.inv(view)
    world_from_camera[3, :3] *= units
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled >= near_m) & (sampled <= far_m)
    optical_depth = sampled[valid]
    camera_points = np.column_stack(((columns[valid] - intrinsics[0, 2]) * optical_depth / intrinsics[0, 0],
                                     -(rows[valid] - intrinsics[1, 2]) * optical_depth / intrinsics[1, 1],
                                     -optical_depth))
    world_points = camera_points @ world_from_camera[:3, :3] + world_from_camera[3, :3]
    result = world_points.astype(np.float32), intrinsics, world_from_camera
    if with_rgb:
        return (*result, product_rgb(product)[::stride, ::stride][valid])
    return result


class DepthFusion:
    def __init__(self, paths, stride=4, voxel_m=.01):
        self.paths = dict(paths)
        self.stride = stride
        self.voxel_m = voxel_m
        self.capture_id = 0
        self.frames = {}
        self.cloud = None

    def update(self, payload):
        reference_time = payload.get('reference_time', (0, 0))
        if len(reference_time) != 2 or reference_time[1] <= 0:
            return False
        products = {str(product.get('camera')): product
                    for product in payload.get('renderProducts', {}).values()}
        required = {'distance_to_image_plane', 'CameraParams', 'rgb'}
        if any(path not in products or not required.issubset(products[path])
               for path in self.paths.values()):
            return False
        frames = {}
        point_sets = []
        color_sets = []
        source_sets = []
        capture_id = self.capture_id + 1
        received_at = time.monotonic()
        for camera_id, (arm, path) in enumerate(self.paths.items()):
            product = products[path]
            points, intrinsics, transform, colors = unproject_depth(product, stride=self.stride, with_rgb=True)
            frames[arm] = {
                'depth_m': product_depth(product).copy(),
                'rgb': product_rgb(product).copy(),
                'intrinsics': intrinsics,
                'world_from_camera': transform,
                'reference_time': tuple(reference_time),
                'capture_id': capture_id,
                'received_monotonic_s': received_at,
                'camera_path': path,
            }
            point_sets.append(points)
            color_sets.append(colors)
            source_sets.append(np.full(len(points), camera_id, dtype=np.uint8))
        points = np.concatenate(point_sets)
        colors_rgb = np.concatenate(color_sets)
        camera_ids = np.concatenate(source_sets)
        if len(points) and self.voxel_m > 0:
            voxels = np.floor(points / self.voxel_m).astype(np.int64)
            minimum = voxels.min(axis=0)
            shape = voxels.max(axis=0) - minimum + 1
            keys = np.ravel_multi_index((voxels - minimum).T, tuple(shape))
            _, indices = np.unique(keys, return_index=True)
            points = points[indices]
            colors_rgb = colors_rgb[indices]
            camera_ids = camera_ids[indices]
        self.cloud = {
            'points_world_m': points,
            'colors_rgb': colors_rgb,
            'camera_ids': camera_ids,
            'camera_order': tuple(self.paths),
            'reference_time': tuple(reference_time),
            'capture_id': capture_id,
            'received_monotonic_s': received_at,
            'stride': self.stride,
            'voxel_m': self.voxel_m,
            'source_counts': {arm: len(point_sets[index]) for index, arm in enumerate(self.paths)},
        }
        self.frames = frames
        self.capture_id = capture_id
        return True

    def get_depth_frame(self, arm):
        if arm not in self.paths:
            raise KeyError(arm)
        return copy.deepcopy(self.frames.get(arm))

    def get_point_cloud(self):
        return copy.deepcopy(self.cloud)

    def reset(self):
        self.frames.clear()
        self.cloud = None


class VoxelReconstruction:
    """Confirm world voxels across frames, then preserve their positions and colors."""
    def __init__(self, voxel_m=.01, max_points=400000, max_candidates=120000,
                 candidate_lifetime=30, regions=None, region_from_world=None):
        if voxel_m <= 0 or max_points < 1 or max_candidates < 1 or candidate_lifetime < 1:
            raise ValueError('Reconstruction limits must be positive')
        self.voxel_m = voxel_m
        self.max_points = max_points
        self.max_candidates = max_candidates
        self.candidate_lifetime = candidate_lifetime
        self.regions = [(np.asarray(lower), np.asarray(upper)) for lower, upper in (regions or [])]
        self.region_from_world = np.eye(4) if region_from_world is None else np.asarray(region_from_world)
        self.map_epoch = 0
        self.reset()

    def reset(self):
        self.map_epoch += 1
        self.cloud = None
        self._keys = np.empty(0, dtype='V24')
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors_rgb = np.empty((0, 3), dtype=np.uint8)
        self._camera_ids = np.empty(0, dtype=np.uint8)
        self._first_seen = np.empty(0, dtype=np.int64)
        self._pending_keys = np.empty(0, dtype='V24')
        self._pending_points = np.empty((0, 3), dtype=np.float32)
        self._pending_colors = np.empty((0, 3), dtype=np.uint8)
        self._pending_cameras = np.empty(0, dtype=np.uint8)
        self._pending_first_seen = np.empty(0, dtype=np.int64)
        self._last_capture_id = -1
        self._last_reference_time = None
        self._start_reference_time = None
        self.integrated_frames = 0

    def update(self, cloud):
        from fractions import Fraction

        if cloud is None:
            return False
        capture_id = int(cloud['capture_id'])
        reference_time = Fraction(*cloud['reference_time'])
        if capture_id <= self._last_capture_id or reference_time == self._last_reference_time:
            return False
        points = cloud['points_world_m']
        keep = np.isfinite(points).all(axis=1)
        if self.regions:
            local = points @ self.region_from_world[:3, :3] + self.region_from_world[3, :3]
            inside = np.zeros(len(points), dtype=bool)
            for lower, upper in self.regions:
                inside |= ((local >= lower) & (local <= upper)).all(axis=1)
            keep &= inside
        points = points[keep]
        colors_rgb = cloud['colors_rgb'][keep]
        camera_ids = cloud['camera_ids'][keep]
        voxels = np.ascontiguousarray(np.floor(points / self.voxel_m), dtype=np.int64)
        keys, indices = np.unique(voxels.view('V24').reshape(-1), return_index=True)
        points, camera_ids = points[indices], camera_ids[indices]
        colors_rgb = colors_rgb[indices]
        if len(self._keys):
            positions = np.searchsorted(self._keys, keys)
            known = positions < len(self._keys)
            known[known] = self._keys[positions[known]] == keys[known]
            keys, points, camera_ids = keys[~known], points[~known], camera_ids[~known]
            colors_rgb = colors_rgb[~known]
        active = capture_id - self._pending_first_seen <= self.candidate_lifetime
        candidate_keys = np.concatenate((self._pending_keys[active], keys))
        candidate_points = np.concatenate((self._pending_points[active], points))
        candidate_colors = np.concatenate((self._pending_colors[active], colors_rgb))
        candidate_cameras = np.concatenate((self._pending_cameras[active], camera_ids))
        candidate_first_seen = np.concatenate((self._pending_first_seen[active],
                                               np.full(len(keys), capture_id, dtype=np.int64)))
        unique_keys, first, counts = np.unique(candidate_keys, return_index=True, return_counts=True)
        confirmed = np.flatnonzero(counts >= 2)[:max(0, self.max_points - len(self._points))]
        if len(confirmed):
            additions = first[confirmed]
            self._keys = np.sort(np.concatenate((self._keys, unique_keys[confirmed])))
            self._points = np.concatenate((self._points, candidate_points[additions]))
            self._colors_rgb = np.concatenate((self._colors_rgb, candidate_colors[additions]))
            self._camera_ids = np.concatenate((self._camera_ids, candidate_cameras[additions]))
            self._first_seen = np.concatenate((self._first_seen, candidate_first_seen[additions]))
        pending = np.flatnonzero(counts == 1)[:self.max_candidates]
        self._pending_keys = unique_keys[pending]
        self._pending_points = candidate_points[first[pending]]
        self._pending_colors = candidate_colors[first[pending]]
        self._pending_cameras = candidate_cameras[first[pending]]
        self._pending_first_seen = candidate_first_seen[first[pending]]
        self._last_capture_id = capture_id
        self._last_reference_time = reference_time
        self.integrated_frames += 1
        if self._start_reference_time is None:
            self._start_reference_time = cloud['reference_time']
        self.cloud = {
            'points_world_m': self._points, 'camera_ids': self._camera_ids,
            'colors_rgb': self._colors_rgb,
            'first_seen_capture_ids': self._first_seen, 'camera_order': cloud['camera_order'],
            'capture_id': capture_id, 'reference_time': cloud['reference_time'],
            'start_reference_time': self._start_reference_time,
            'received_monotonic_s': cloud['received_monotonic_s'],
            'accumulated': True, 'map_epoch': self.map_epoch, 'voxel_m': self.voxel_m,
            'integrated_frames': self.integrated_frames, 'pending_points': len(self._pending_keys),
            'capacity_reached': len(self._points) >= self.max_points,
        }
        return True

    def get_point_cloud(self):
        return copy.deepcopy(self.cloud)


class PointCloudRaster:
    def __init__(self, width=800, height=480):
        self.width = width
        self.height = height
        self.yaw = -55.0
        self.elevation = 28.0
        self.zoom = 1.0
        self.center = None
        self.radius = 3.0
        self._signature = None
        self._rendered_points = 0

    def fit(self, points):
        if len(points):
            lower, upper = np.percentile(points, [1, 99], axis=0)
            self.center = (lower + upper) / 2
            self.radius = max(float(np.linalg.norm(upper - lower) / 2), .5)
        self.zoom = 1.0

    def render(self, cloud):
        points = cloud['points_world_m'] if cloud is not None else np.empty((0, 3))
        if self.center is None and len(points):
            self.fit(points)
        accumulated = cloud is not None and cloud.get('accumulated', False)
        signature = (tuple(self.center) if self.center is not None else None, self.radius,
                     self.yaw, self.elevation, self.zoom, self.width, self.height,
                     cloud.get('map_epoch') if cloud is not None else None)
        if not accumulated or signature != self._signature or len(points) < self._rendered_points:
            self._rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
            self._rgba[:] = [17, 22, 30, 255]
            self._depth = np.full((self.height, self.width), -np.inf)
            self._rendered_points = 0
            self._signature = signature
        start = self._rendered_points
        self._rendered_points = len(points)
        if self.center is None or len(points) == start:
            return self._rgba
        yaw, elevation = np.deg2rad([self.yaw, self.elevation])
        right = np.array([-np.sin(yaw), np.cos(yaw), 0])
        forward = np.array([np.cos(yaw) * np.cos(elevation),
                            np.sin(yaw) * np.cos(elevation), np.sin(elevation)])
        up = np.cross(forward, right)
        scale = min(self.width, self.height) * .44 * self.zoom / self.radius
        relative = points[start:] - self.center
        columns = np.rint(relative @ right * scale + self.width / 2).astype(int)
        rows = np.rint(self.height / 2 - relative @ up * scale).astype(int)
        visible = (columns >= 1) & (columns < self.width - 1) & (rows >= 1) & (rows < self.height - 1)
        indices = np.flatnonzero(visible)
        distances = relative @ forward
        pixel_rows = rows[indices, None] + np.array([0, 0, 1, 1])
        pixel_columns = columns[indices, None] + np.array([0, 1, 0, 1])
        pixels = (pixel_rows * self.width + pixel_columns).reshape(-1)
        depths = np.repeat(distances[indices], 4)
        colors = np.repeat(cloud['colors_rgb'][start:][indices], 4, axis=0)
        ordered = np.argsort(-depths, kind='stable')
        _, nearest = np.unique(pixels[ordered], return_index=True)
        selected = ordered[nearest]
        selected = selected[depths[selected] > self._depth.ravel()[pixels[selected]]]
        self._depth.ravel()[pixels[selected]] = depths[selected]
        self._rgba.reshape(-1, 4)[pixels[selected], :3] = colors[selected]
        return self._rgba
