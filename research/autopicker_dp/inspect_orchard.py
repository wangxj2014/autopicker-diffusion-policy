"""Inspect the orchard USD layout under Isaac Sim; does not modify the stage."""
import argparse
import json
from pathlib import Path

import numpy as np


def matrix_rows(value):
    return np.asarray(value, dtype=float).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=Path(__file__).resolve().parents[2] / "outputs/autopicker_apple_orchard.usd")
    parser.add_argument("--output", type=Path, default=Path("runs/autopicker/orchard_usd_inspection.json"))
    args = parser.parse_args()
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from pxr import UsdGeom, Usd

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve())):
            raise RuntimeError(f"Could not open USD: {args.scene}")
        for _ in range(120):
            app.update()
            stage = context.get_stage()
            if stage is not None and stage.GetRootLayer().GetPrimAtPath("/World"):
                break
        if stage is None:
            raise RuntimeError("USD stage did not load")

        def world_matrix(path):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                return None
            return matrix_rows(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        world_children = []
        world = stage.GetPrimAtPath("/World")
        if world:
            world_children = [
                {"path": str(child.GetPath()), "type": child.GetTypeName(),
                 "world_matrix": world_matrix(str(child.GetPath()))}
                for child in world.GetChildren()
            ]

        robot_paths = []
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if name in ("autopicker", "root_joint", "base_link") or "autopicker" in str(prim.GetPath()).lower():
                robot_paths.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                                    "world_matrix": world_matrix(str(prim.GetPath()))})

        apples = []
        tree_groups = {}
        for prim in stage.Traverse():
            rest_attr = prim.GetAttribute("picking:restTranslate")
            if not rest_attr or not rest_attr.HasAuthoredValueOpinion():
                continue
            path = str(prim.GetPath())
            xform = UsdGeom.Xformable(prim)
            world = np.asarray(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=float)
            side_attr = prim.GetAttribute("picking:side")
            side = side_attr.Get() if side_attr else None
            ancestors = []
            parent = prim.GetParent()
            while parent and parent.GetPath() != "/":
                p = str(parent.GetPath())
                if p == "/World":
                    break
                ancestors.append({"path": p, "name": parent.GetName(), "type": parent.GetTypeName(),
                                  "world_matrix": world_matrix(p)})
                parent = parent.GetParent()
            geometry_path = path.split("/Fruit/")[0] if "/Fruit/" in path else str(prim.GetParent().GetPath())
            tree_key = next((a["path"] for a in ancestors
                             if "tree" in a["name"].lower()), geometry_path)
            item = {"path": path, "side": side, "rest_local": list(rest_attr.Get()),
                    "world_position_m": world.tolist(), "geometry_path": geometry_path,
                    "tree_key": tree_key, "ancestors": ancestors}
            apples.append(item)
            tree_groups.setdefault(tree_key, []).append(path)

        result = {
            "scene": str(args.scene.resolve()),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "world_children": world_children,
            "robot_prims": robot_paths,
            "apple_count": len(apples),
            "tree_groups": [{"tree_key": key, "apple_count": len(paths), "apple_paths": paths}
                            for key, paths in sorted(tree_groups.items())],
            "apples": apples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "apple_count": len(apples),
                          "tree_groups": result["tree_groups"],
                          "world_children": [{"path": x["path"], "type": x["type"]}
                                             for x in world_children],
                          "robot_prims": [{"path": x["path"], "type": x["type"]}
                                          for x in robot_paths]}, indent=2), flush=True)
        print("AUTOPICKER_ORCHARD_USD_INSPECTION_PASS", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
