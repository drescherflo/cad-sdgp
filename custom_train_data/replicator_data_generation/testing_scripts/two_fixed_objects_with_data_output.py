"""
This script was used to create repeatable / debuggable input data for the ROCA training data generation script
"""

# Launch Isaac Sim
import os
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)

import omni
import omni.replicator.core as rep
import omni.graph.core as og
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils import prims, extensions
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# enable ROS bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")

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
    xform_prims.append(prims.create_prim(prim_path="/simple_object_1", usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[-1, -0.5, 0]))
    xform_prims.append(prims.create_prim(prim_path="/simple_object_2", usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[+1, +0.5, 0]))

    # Apply semantics
    for xform_prim in xform_prims:
        add_update_semantics(xform_prim, semantic_label="simple_object", type_label="class")

    # Add camera
    camera = rep.create.camera(position=(0, 0, 5), rotation=(-90, -90, 0))  # Look at (0, 0, 0) with x axis to the right
    render_product = rep.create.render_product(camera=camera, resolution=(1920, 1080))

    # Initialize and attach basic writer
    out_dir = os.getcwd() + "/temp_replicator_out"
    train_data_writer = rep.WriterRegistry.get("BasicWriter")
    train_data_writer.initialize(output_dir=out_dir, rgb=True, distance_to_camera=True,  distance_to_image_plane=True, camera_params=True, image_output_format="jpg")
    train_data_writer.attach([render_product])

    # Initialize and attach camera info publisher writer
    topic_name = "camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = "camera_frame"
    stereo_offset = [0, 0]
    pub_freq = 60
    step_size = int(60/pub_freq)

    pub_writer = rep.writers.get("ROS2PublishCameraInfo")
    pub_writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        stereoOffset=stereo_offset,
    )
    pub_writer.attach([render_product])

    # Set Execution of render_product to 60 Hz
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product.path
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    # Render once
    simulation_app.update()

    # Capture training data
    num_frames = 2
    for i in range(num_frames):
        print(f"Writing frame {str(i + 1)} of {num_frames}")
        rep.orchestrator.step(rt_subframes=32)  # Generate 32 subframes for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))

    # Detach train_data_writer to prevent generation of more frames than specified because of OmniGraph registration
    train_data_writer.detach()

    # Isaac Sim run-loop (only for testing, do NOT use this when generating data)
    world.reset()  # This is required instead of sim_app.update for ros publishers to work
    while simulation_app.is_running():
        world.step(render=True)  # This is required instead of sim_app.update for ros publishers to work


if __name__ == '__main__':
    main()
    simulation_app.close()
