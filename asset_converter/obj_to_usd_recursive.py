#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
import asyncio

from isaacsim import SimulationApp

config = {
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(config)

from omni.kit.asset_converter import get_instance as get_asset_converter, AssetConverterContext
from pxr import Usd, Sdf, Gf, UsdGeom, UsdPhysics, PhysxSchema


async def _convert_single_obj_async(converter, src_obj: str, dst_usd: str) -> bool:
    """Async-Wrapper um die AssetConverter-Task."""
    print(f"[INFO]    Konvertiere OBJ → USD: {src_obj} -> {dst_usd}")

    ctx = AssetConverterContext()
    ctx.ignore_materials = False
    ctx.smooth_normals = True
    ctx.create_collider = False
    ctx.create_physics_scene = False

    task = converter.create_converter_task(
        src_obj,
        dst_usd,
        asset_converter_context=ctx,
    )

    success = await task.wait_until_finished()
    if not success:
        print(f"[WARN]    Konvertierung fehlgeschlagen: {src_obj}")
        print(f"         Fehler: {task.get_error_message()}")
        return False

    print(f"[INFO]    Konvertierung erfolgreich: {dst_usd}")
    return True


def convert_single_obj(converter, src_obj: str, dst_usd: str) -> bool:
    """Sync-Hülle, damit der restliche Code unverändert bleiben kann."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_convert_single_obj_async(converter, src_obj, dst_usd))
    finally:
        loop.close()


def postprocess_usd(usdfile: str, scale: float):
    print(f"[INFO]    Postprocessing USD: {usdfile}")
    stage = Usd.Stage.Open(usdfile)
    if stage is None:
        print(f"[ERROR]   Konnte Stage nicht öffnen: {usdfile}")
        return

    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        root_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(root_prim)

    mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        print("[WARN]    Keine Meshes gefunden.")
        stage.Save()
        return

    meshes = [UsdGeom.Mesh(prim) for prim in mesh_prims]

    if abs(scale - 1.0) > 1e-6:
        xformable = UsdGeom.Xformable(root_prim)
        ops = xformable.GetOrderedXformOps()
        scale_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(scale, scale, scale))


    scene_path = Sdf.Path("/World/physicsScene")
    if not stage.GetPrimAtPath(scene_path):
        phys_scene = UsdPhysics.Scene.Define(stage, scene_path)
        UsdPhysics.Scene(phys_scene).CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        UsdPhysics.Scene(phys_scene).CreateGravityMagnitudeAttr(981.0)  # cm/s²

    for mesh in meshes:
        prim = mesh.GetPrim()

        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)

        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)

        approx_attr = mesh_coll.GetApproximationAttr()
        if not approx_attr or not approx_attr.IsValid():
            approx_attr = mesh_coll.CreateApproximationAttr()
        approx_attr.Set("convexHull")


    stage.Save()
    print(f"[INFO]    Postprocessing fertig: {usdfile}")


def convert_folder_recursive(input_dir: str, output_dir: str, scale: float):
    converter = get_asset_converter()

    obj_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".obj"):
                obj_files.append(os.path.join(root, f))

    if not obj_files:
        print("[ERROR]  Keine OBJ-Dateien gefunden.")
        return

    obj_files.sort()
    print(f"[INFO]   {len(obj_files)} OBJ-Dateien gefunden, starte Konvertierung...")

    for idx, obj_path in enumerate(obj_files, start=1):
        rel = os.path.relpath(obj_path, input_dir)
        usd_rel = os.path.splitext(rel)[0] + ".usd"
        dst_usd = os.path.join(output_dir, usd_rel)
        os.makedirs(os.path.dirname(dst_usd), exist_ok=True)

        print(f"[INFO] [{idx}/{len(obj_files)}] {rel}")

        try:
            ok = convert_single_obj(converter, obj_path, dst_usd)
            if not ok:
                continue
            postprocess_usd(dst_usd, scale)
        except Exception:
            print(f"[ERROR]  Ausnahme bei Datei {obj_path}:")
            traceback.print_exc()

    print("[INFO]   Rekursiver Conversion-Run fertig.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Ordner mit OBJ-Dateien")
    p.add_argument("--output_dir", required=True, help="Zielordner für USD-Dateien")
    p.add_argument("--scale", type=float, default=1.0, help="Einheitsloser Skalierungsfaktor")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(in_dir):
        print(f"[ERROR] Input-Verzeichnis existiert nicht: {in_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    convert_folder_recursive(in_dir, out_dir, args.scale)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
