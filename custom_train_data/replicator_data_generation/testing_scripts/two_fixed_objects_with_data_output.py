"""
This script was used to create repeatable / debuggable input data for the ROCA training data generation script
"""

# Launch Isaac Sim
import os
import json
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)

import omni
import omni.replicator.core as rep
import omni.graph.core as og
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import prims, extensions
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# enable ROS bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")

def main():
    out_dir = os.getcwd() + "/temp_replicator_out"

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
    ## Reconstruct obj path from usd path
    obj_path_split = usd_path.split("/")
    obj_path_split[-2] = obj_path_split[-2].removesuffix("_converted")
    obj_path_split[-1] = obj_path_split[-1].replace("_obj.usd", ".obj")
    obj_path = "/".join(obj_path_split)

    ## Create valid prim_path from usd_path
    prim_path_start = usd_path.split("/")[-1].replace(" ", "_").removesuffix(".usd")
    prim_paths = [f"/{prim_path_start}_{i}" for i in range(2)]

    ## Map prim path to obj path
    prim_path_to_obj_path = {}
    for prim_path in prim_paths:
        prim_path_to_obj_path[prim_path] = obj_path

    ## Add prims to stage
    xform_prims = []
    xform_prims.append(prims.create_prim(prim_path=prim_paths[0], usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[-1, -0.5, 0]))
    xform_prims.append(prims.create_prim(prim_path=prim_paths[1], usd_path=usd_path, scale=[0.001, 0.001, 0.001], position=[+1, +0.5, 0]))

    # Apply semantics
    label = obj_path.split("/")[-1].removesuffix(".obj")
    for xform_prim in xform_prims:
        add_update_semantics(xform_prim, semantic_label=label, type_label="class")

    # Add camera
    camera = rep.create.camera(position=(0, 0, 5), rotation=(-90, -90, 0))  # Look at (0, 0, 0) with x axis to the right
    render_product = rep.create.render_product(camera=camera, resolution=(1920, 1080))

    # Initialize and attach basic writer
    train_data_writer = rep.WriterRegistry.get("BasicWriter")
    train_data_writer.initialize(output_dir=out_dir, rgb=True, distance_to_camera=True,  distance_to_image_plane=True, camera_params=True, image_output_format="png")
    train_data_writer.attach([render_product])

    # Initialize and attach bounding box 2d tight annotator to detect which objects are visible in the image
    bbox_2d_tight_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_2d_tight_annotator.attach(render_product)

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
    for frame_nr in range(num_frames):
        print(f"Writing frame {str(frame_nr + 1)} of {num_frames}")

        # Execute orchestrator to run "randomization" and capture data with writers
        rep.orchestrator.step(rt_subframes=32)  # Generate 32 subframes for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))

        # Store world pose of visible objects in the image
        bbox_data = bbox_2d_tight_annotator.get_data()
        visible_prims_paths = bbox_data["info"]["primPaths"]
        objs_and_world_poses = []
        for prim_path in visible_prims_paths:
            prim = XFormPrim(prim_path=prim_path)
            prim_pose = prim.get_world_pose()
            obj_and_pose = {
                "obj_path": prim_path_to_obj_path[prim_path],
                "pose": {
                    "position": {
                        "x": prim_pose[0][0].astype(float),
                        "y": prim_pose[0][1].astype(float),
                        "z": prim_pose[0][2].astype(float)
                    },
                    "orientation": {
                        "w": prim_pose[1][0].astype(float),
                        "x": prim_pose[1][1].astype(float),
                        "y": prim_pose[1][2].astype(float),
                        "z": prim_pose[1][3].astype(float),
                    }
                }
            }
            objs_and_world_poses.append(obj_and_pose)
        with open(f"{out_dir}/world_pose_visible_objects_{frame_nr:04d}.json", "w") as f:
            json.dump(objs_and_world_poses, f)

    # Detach train_data_writer to prevent generation of more frames than specified because of OmniGraph registration for ROS publisher
    train_data_writer.detach()

    # Isaac Sim run-loop (only for testing and ROS publishing, do NOT use this when generating data)
    #world.reset()  # This is required instead of sim_app.update for ros publishers to work
    #while simulation_app.is_running():
    #    world.step(render=True)  # This is required instead of sim_app.update for ros publishers to work


if __name__ == '__main__':
    main()
    simulation_app.close()
