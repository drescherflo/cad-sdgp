#!/usr/bin/env python3
import argparse
import os
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
    print(f"[INFO] Konvertiere OBJ → USD: {src_obj} -> {dst_usd}")

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
        print(f"[WARN] Konvertierung fehlgeschlagen: {src_obj}")
        print(f"Fehler: {task.get_error_message()}")
        return False

    print(f"[INFO] Konvertierung erfolgreich: {dst_usd}")
    return True


def convert_single_obj(converter, src_obj: str, dst_usd: str) -> bool:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_convert_single_obj_async(converter, src_obj, dst_usd))
    finally:
        loop.close()


def postprocess_usd(usdfile: str, scale: float, collision, rigid_body, mass_props, density):
    print(f"[INFO] Postprocessing USD: {usdfile}")
    stage = Usd.Stage.Open(usdfile)
    if stage is None:
        print(f"[ERROR] Konnte Stage nicht öffnen: {usdfile}")
        return

    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        root_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(root_prim)

    mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        print("[WARN] Keine Meshes gefunden.")
        stage.Save()
        return

    meshes = [UsdGeom.Mesh(prim) for prim in mesh_prims]

    if abs(scale - 1.0) > 1e-6:
        for mesh in meshes:
            pts_attr = mesh.GetPointsAttr()
            pts = pts_attr.Get()
            if not pts:
                continue

            scaled = [Gf.Vec3f(p[0] * scale, p[1] * scale, p[2] * scale) for p in pts]
            pts_attr.Set(scaled)

            min_p = Gf.Vec3f(scaled[0])
            max_p = Gf.Vec3f(scaled[0])
            for p in scaled[1:]:
                min_p[0] = min(min_p[0], p[0])
                min_p[1] = min(min_p[1], p[1])
                min_p[2] = min(min_p[2], p[2])
                max_p[0] = max(max_p[0], p[0])
                max_p[1] = max(max_p[1], p[1])
                max_p[2] = max(max_p[2], p[2])

            mesh.GetExtentAttr().Set([min_p, max_p])

    scene_path = Sdf.Path("/World/physicsScene")
    if not stage.GetPrimAtPath(scene_path):
        phys_scene = UsdPhysics.Scene.Define(stage, scene_path)
        UsdPhysics.Scene(phys_scene).CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        UsdPhysics.Scene(phys_scene).CreateGravityMagnitudeAttr(981.0)  # cm/s²

    for mesh in meshes:
        prim = mesh.GetPrim()

        if rigid_body:
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)

        if collision == "convexHull":
            UsdPhysics.CollisionAPI.Apply(prim)
            coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
            coll.CreateApproximationAttr().Set("convexHull")

        if mass_props:
            extent = mesh.GetExtentAttr().Get()
            if not extent:
                continue

            min_p, max_p = extent
            dx = max_p[0] - min_p[0]
            dy = max_p[1] - min_p[1]
            dz = max_p[2] - min_p[2]

            if dx <= 0 or dy <= 0 or dz <= 0:
                continue

            volume = dx * dy * dz
            mass = density * volume

            com = Gf.Vec3f(
                (min_p[0] + max_p[0]) * 0.5,
                (min_p[1] + max_p[1]) * 0.5,
                (min_p[2] + max_p[2]) * 0.5,
            )

            Ix = (1/12) * mass * (dy*dy + dz*dz)
            Iy = (1/12) * mass * (dx*dx + dz*dz)
            Iz = (1/12) * mass * (dx*dx + dy*dy)

            mapi = UsdPhysics.MassAPI.Apply(prim)
            mapi.CreateMassAttr(mass)
            mapi.CreateCenterOfMassAttr(com)
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(Ix, Iy, Iz))

    stage.Save()
    print(f"[INFO] Postprocessing fertig: {usdfile}")



def convert_folder_recursive(args):
    converter = get_asset_converter()

    obj_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(".obj"):
                obj_files.append(os.path.join(root, f))

    if not obj_files:
        print("[ERROR] Keine OBJ-Dateien gefunden.")
        return

    obj_files.sort()
    print(f"[INFO] {len(obj_files)} OBJ-Dateien gefunden, starte Konvertierung...")

    for idx, obj_path in enumerate(obj_files, start=1):
        rel = os.path.relpath(obj_path, args.input_dir)
        usd_rel = os.path.splitext(rel)[0] + ".usd"
        dst_usd = os.path.join(args.output_dir, usd_rel)
        os.makedirs(os.path.dirname(dst_usd), exist_ok=True)

        print(f"[INFO] [{idx}/{len(obj_files)}] {rel}")

        try:
            ok = convert_single_obj(converter, obj_path, dst_usd)
            if not ok:
                continue
            postprocess_usd(dst_usd, args.scale, args.collision, args.rigid_body, args.mass_props, args.density)
        except Exception:
            print(f"[ERROR] Ausnahme bei Datei {obj_path}:")
            traceback.print_exc()

    print("[INFO] Rekursiver Conversion-Run fertig.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scale", type=float, default=1.0)

    p.add_argument("--collision",
                   choices=["none", "convexHull"],
                   default="convexHull")

    p.add_argument("--rigid_body", action="store_true")
    p.add_argument("--mass_props", action="store_true")
    p.add_argument("--density", type=float, default=1000.0)

    return p.parse_args()


if __name__ == "__main__":
    try:
        convert_folder_recursive(parse_args())
    finally:
        simulation_app.close()
