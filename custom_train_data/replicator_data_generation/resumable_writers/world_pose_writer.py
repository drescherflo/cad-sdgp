import json
from .writer_interface import ResumableWriterInterface
from omni.replicator.core import AnnotatorRegistry
from omni.isaac.core.prims import XFormPrim


class WorldPoseWriter(ResumableWriterInterface):
    def __init__(self, output_dir: str, init_frame_nr: int = 0, frame_padding: int = 4):
        self._output_dir = output_dir
        self._frame_nr = init_frame_nr
        self._frame_padding = frame_padding
        self.annotators = []

        self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight"))

    def write(self, data: dict):
        if "bounding_box_2d_tight" in data:
            # Store world pose of visible objects in the image
            bbox_data = data["bounding_box_2d_tight"]
            visible_prims_paths = bbox_data["info"]["primPaths"]
            objs_and_world_poses = []
            for i, prim_path in enumerate(visible_prims_paths):
                prim = XFormPrim(prim_path=prim_path)
                prim_pose = prim.get_world_pose()
                semantic_id = bbox_data["data"][i][0]
                semantic_labels = bbox_data["info"]["idToLabels"][str(semantic_id)]  # Disabled because mapping from bbox_id to labels does not work

                semantic_labels_and_pose = {
                    "semantic_labels": semantic_labels,  # just as additional info
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

                objs_and_world_poses.append(semantic_labels_and_pose)

            with open(f"{self._output_dir}/world_pose_visible_objects_{self._frame_nr:0{self._frame_padding}}.json", "w") as f:
                json.dump(objs_and_world_poses, f, indent=4)
            self._frame_nr += 1