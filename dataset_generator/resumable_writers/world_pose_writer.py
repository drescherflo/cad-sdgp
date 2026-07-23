import json
from .writer_interface import ResumableWriterInterface
from omni.replicator.core import AnnotatorRegistry
from omni.isaac.core.prims import XFormPrim


class WorldPoseWriter(ResumableWriterInterface):
    """
    ResumableWriter that focuses on writing the world pose of visible objects.
    """

    def __init__(self, output_dir: str, init_frame_nr: int = 0, frame_padding: int = 4):
        """
        Initializes an instance of WorldPoseWriter for writing world poses of visible objects to a specified output directory.

        This class captures and writes the world pose (position and orientation) of visible objects within a scene,
        starting from a given frame number. It supports frame numbering with customizable padding for filename consistency.

        :param output_dir: The directory where world pose data files will be written.
        :type output_dir: str
        :param init_frame_nr: The initial frame number from which to start writing data. Defaults to 0.
        :type init_frame_nr: int
        :param frame_padding: The number of digits to use for zero-padding the frame number in the output filenames. Defaults to 4.
        :type frame_padding: int
        """

        self._output_dir = output_dir
        self._frame_nr = init_frame_nr
        self._frame_padding = frame_padding
        self.annotators = []

        self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight"))

    def write(self, data: dict):
        """
        Writes the world poses of visible objects for the current frame to a JSON file. The data includes both position and orientation
        for each visible object identified by bounding box annotations. The output JSON file is named using the frame number with
        zero-padding as configured.

        This method processes data provided by the 'bounding_box_2d_tight' annotator, extracting the world poses of objects visible in
        the scene and saving them to a JSON file in the output directory.

        :param data: Dictionary containing data for the current frame, expected to include 'bounding_box_2d_tight' information.
        :type data: dict
        """

        if "bounding_box_2d_tight" in data:
            # Store world pose of visible objects in the image
            bbox_data = data["bounding_box_2d_tight"]
            visible_prims_paths = bbox_data["info"]["primPaths"]
            objs_and_world_poses = []
            for i, prim_path in enumerate(visible_prims_paths):
                prim = XFormPrim(prim_path=prim_path)
                prim_pose = prim.get_world_pose()
                semantic_id = bbox_data["data"][i][0]
                semantic_labels = bbox_data["info"]["idToLabels"][str(semantic_id)]
                occlusion_ratio = bbox_data["data"][i]["occlusionRatio"].astype(float)

                semantic_labels_and_pose = {
                    "prim_path": prim_path,
                    "semantic_labels": semantic_labels,
                    "occlusion_ratio": occlusion_ratio,
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
