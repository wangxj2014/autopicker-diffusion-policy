# Autopicker development and distribution layout

The user approved separating model data from Python behavior, with one startup entry and an optional single-file demonstration build.

The existing `outputs/autopicker_apple_orchard.usd` becomes the development scene. Remove only `/World/ApplePickingGraph`; retain all geometry, materials, camera settings, physics, and authored scene metadata. No runtime code is stored in the development USD.

Keep `autopicker_apple_pick.py`, `autopicker_camera_viewer.py`, and `autopicker_depth_fusion.py` as the editable source modules. Add `autopicker_runtime.py` to coordinate their lifecycle and `autopicker_start.py` as the entry point for an existing Isaac Sim session or Isaac Sim's Python executable. Each explicit launch loads current source from disk, replaces the previous runtime, and starts picking plus the three existing camera windows. Stop/Play resets the controller and map, Pause/Play continues, and stage close cancels initialization and releases subscriptions/windows. Source files are loaded relative to the entry point, independent of the working directory.

Add `autopicker_package.py`, which creates `outputs/dist/autopicker_apple_orchard_demo.usd` from the development USD plus current source files. The generated demo embeds the same runtime and modules and starts on Play. It must never overwrite the development scene or alter source files. Only the generated demo is copied for single-file distribution; development use requires the scene and sibling Python files.

The packager supports the supplied self-contained scene. It rejects input scenes with external layers or assets, rather than relocating unresolved composition paths. The root layer is copied into a new anonymous layer to preserve Isaac-specific metadata.

Use the existing configuration dictionary in the controller. No changes to motion planning, geometry, speeds, RGB-D registration, accumulated RGB colors, or receiver behavior. The loader reports missing/syntactically invalid source before replacing a working runtime.

Validation covers scene identity outside the graph, pure development scene, packaged source identity, modified source loading without modifying USD, duplicate launch, startup cancellation, Pause/Play, Stop/Play, stage close, and full orchard RGB/depth/reconstruction/picking regression in both modes. The documentation distinguishes development startup from demo Open/Play.
