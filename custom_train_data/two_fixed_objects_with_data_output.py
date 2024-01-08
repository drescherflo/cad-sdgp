"""
This script was used to create repeatable / debuggable input data for the ROCA training data generation script
"""

# Launch Isaac Sim
import os
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)

# Register signal handler for Ctrl+C for exiting
import signal


def signal_handler(signal, frame):
    print("Received Ctrl+C, exiting...")
    simulation_app.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)



import omni.replicator.core as rep
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.semantics import add_update_semantics


def main():
    # Setup simulation
    ## Create new stage
    create_new_stage()

    ## Create world
    world = World()
    scene = world.scene

    ## Add ground plane to scene
    scene.add_default_ground_plane()

    # Add objects
    usd_path = "CAD Models/OBJ_converted/MA Simple Object_obj.usd"
    xform_prims = []
    xform_prims.append(prims.create_prim(prim_path="/simple_object_1", usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[-1, 0, 0]))
    xform_prims.append(prims.create_prim(prim_path="/simple_object_2", usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[+1, 0, 0]))

    # Apply semantics
    for xform_prim in xform_prims:
        add_update_semantics(xform_prim, semantic_label="simple_object", type_label="class")

    # Add camera
    camera = rep.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))  # Look at (0, 0, 0) without weird rotations applied
    render_product = rep.create.render_product(camera=camera, resolution=(480, 360))

    # Initialize and attach writer
    out_dir = os.getcwd() + "/temp_replicator_out"
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=out_dir, rgb=True, distance_to_camera=True,  distance_to_image_plane=True, camera_params=True, image_output_format="jpg")
    writer.attach([render_product])

    # Render once
    simulation_app.update()

    # Capture data
    num_frames = 1
    for i in range(num_frames):
        print(f"Writing frame {str(i + 1)} of {num_frames}")
        rep.orchestrator.step(rt_subframes=32)  # Generate 32 subframes for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
    #rep.orchestrator.run_until_complete(num_frames=1)

    # Isaac Sim run-loop (only for testing, do NOT use this when generating data)
    while True:
        simulation_app.update()


if __name__ == '__main__':
    main()
    simulation_app.close()
