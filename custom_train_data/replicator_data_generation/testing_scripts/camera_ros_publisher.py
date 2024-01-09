'''
Template code for publishing ros camera info.
Script can be ignored.

Code from https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/tutorial_ros2_camera_publishing.html (09.01.2024)
'''

import carb

from omni.isaac.kit import SimulationApp

import sys

import argparse

parser = argparse.ArgumentParser(description="Ros2 Bridge Sample")

parser.add_argument(

    "--ros2_bridge",

    default="omni.isaac.ros2_bridge",

    nargs="?",

    choices=["omni.isaac.ros2_bridge", "omni.isaac.ros2_bridge-humble"],

)

args, unknown = parser.parse_known_args()

BACKGROUND_STAGE_PATH = "/background"

BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of stages and manual publishing of images

simulation_app = SimulationApp(CONFIG)

import omni

import numpy as np

from omni.isaac.core import SimulationContext

from omni.isaac.core.utils import stage, extensions, nucleus

import omni.graph.core as og

import omni.replicator.core as rep

import omni.syntheticdata._syntheticdata as sd

from omni.isaac.core.utils.prims import set_targets

from omni.isaac.sensor import Camera

import omni.isaac.core.utils.numpy.rotations as rot_utils

from omni.isaac.core.utils.prims import is_prim_path_valid

from omni.isaac.core_nodes.scripts.utils import set_target_prims


def publish_camera_info(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.

    render_product = camera._render_product_path

    step_size = int(60 / freq)

    topic_name = camera.name + "_camera_info"

    queue_size = 1

    node_namespace = ""

    frame_id = camera.prim_path.split("/")[-1]  # This matches what the TF tree is publishing.

    stereo_offset = [0.0, 0.0]

    writer = rep.writers.get("ROS2PublishCameraInfo")

    writer.initialize(

        frameId=frame_id,

        nodeNamespace=node_namespace,

        queueSize=queue_size,

        topicName=topic_name,

        stereoOffset=stereo_offset,

    )

    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(

        "PostProcessDispatch" + "IsaacSimulationGate", render_product

    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate

    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


# enable ROS2 bridge extension

extensions.enable_extension(args.ros2_bridge)

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder to load environment and robot stages

assets_root_path = nucleus.get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

    simulation_app.close()

    sys.exit()

# Loading the simple_room environment

stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)

############### Some Camera helper functions for setting up publishers. ###############


# Paste functions from the tutorials here


########################################################################################


# Create a Camera prim.

camera = Camera(

    prim_path="/World/camera",

    position=np.array([0.0, 0.0, 2.0]),

    frequency=20,

    resolution=(256, 256),

    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),

)

simulation_app.update()

camera.initialize()

############### Calling Camera publishing functions ###############


# Setup publishers.

# publish_camera_tf(camera)

publish_camera_info(camera, 30)

# publish_rgb(camera, 30)

# publish_depth(camera, 30)

# publish_pointcloud_from_depth(camera, 30)


####################################################################


# Need to initialize physics getting any articulation..etc

simulation_context.initialize_physics()

simulation_context.play()

while simulation_app.is_running():
    # Run with a fixed step size

    simulation_context.step(render=True)

simulation_context.stop()

simulation_app.close()
