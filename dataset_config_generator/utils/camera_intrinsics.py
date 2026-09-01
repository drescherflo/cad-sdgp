"""
Camera intrinsics CLI flags and config generation, shared by all dataset_config_generator scripts.

Isaac Sim/Replicator has no native pixel-space intrinsics model (fx, fy, cx, cy), only a
photographic, millimeter-based one (focal_length, horizontal_aperture, aperture offsets). Camera
calibration procedures however produce fx, fy, cx, cy in pixels. This module exposes the pixel-space
interface on the CLI and converts it to Replicator's physical parameters via the standard pinhole
camera formulas, so rep.create.camera(...) remains the only integration point with Isaac Sim.
"""

import argparse

from utils import config

# Isaac Sim's default horizontal_aperture (mm). Only the ratio focal_length/horizontal_aperture and
# aperture_offset/horizontal_aperture matter for the rendered image, so this is a fixed reference value,
# not a physical sensor size.
_REFERENCE_HORIZONTAL_APERTURE = 20.955

# Replicator's own rep.create.camera() default focal length (mm), used to keep the pixel-space default
# equivalent to today's (pre-feature) rendering behavior.
_REFERENCE_FOCAL_LENGTH = 24.0


def add_camera_intrinsics_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds CLI flags for the camera's intrinsics, in the pixel-space format a camera calibration produces.

    :param parser: The argument parser to add the flags to.
    :type parser: argparse.ArgumentParser
    """

    parser.add_argument("--focal_length_px", default=None, type=float,
                        help="Focal length in pixels (fx == fy, Isaac Sim only supports square pixels). Defaults to the equivalent of Replicator's built-in 24mm/20.955mm default for the given --frame_width")
    parser.add_argument("--principal_point_x", default=None, type=float,
                        help="Principal point x coordinate in pixels (cx). Defaults to frame_width / 2 (centered)")
    parser.add_argument("--principal_point_y", default=None, type=float,
                        help="Principal point y coordinate in pixels (cy). Defaults to frame_height / 2 (centered)")
    parser.add_argument("--f_stop", default=0.0, type=float, help="Lens aperture. 0.0 disables depth-of-field")
    parser.add_argument("--focus_distance", default=400.0, type=float, help="Focus distance in world units")
    parser.add_argument("--clipping_range_near", default=1.0, type=float, help="Near clipping distance in world units")
    parser.add_argument("--clipping_range_far", default=1000000.0, type=float, help="Far clipping distance in world units")


def build_camera_intrinsics_conf(args: argparse.Namespace, frame_width: int, frame_height: int) -> dict:
    """
    Converts the parsed pixel-space CLI args into Isaac Sim's physical camera parameters.

    :param args: Parsed CLI arguments, as added by add_camera_intrinsics_args.
    :type args: argparse.Namespace
    :param frame_width: Width of the generated frames in pixels.
    :type frame_width: int
    :param frame_height: Height of the generated frames in pixels.
    :type frame_height: int
    :return: A dictionary with the intrinsic camera parameters expected by rep.create.camera(...).
    :rtype: dict
    :raises ValueError: If focal_length_px is not greater than 0, or clipping_range is implausible.
    """

    config.check_range_plausibility(args.clipping_range_near, args.clipping_range_far)

    focal_length_px = args.focal_length_px if args.focal_length_px is not None else _REFERENCE_FOCAL_LENGTH * frame_width / _REFERENCE_HORIZONTAL_APERTURE
    principal_point_x = args.principal_point_x if args.principal_point_x is not None else frame_width / 2
    principal_point_y = args.principal_point_y if args.principal_point_y is not None else frame_height / 2

    if focal_length_px <= 0:
        raise ValueError("focal_length_px must be greater than 0")

    horizontal_aperture = _REFERENCE_HORIZONTAL_APERTURE
    vertical_aperture = horizontal_aperture * frame_height / frame_width

    return {
        "focal_length": focal_length_px * horizontal_aperture / frame_width,
        "focus_distance": args.focus_distance,
        "f_stop": args.f_stop,
        "horizontal_aperture": horizontal_aperture,
        "horizontal_aperture_offset": (principal_point_x - frame_width / 2) * horizontal_aperture / frame_width,
        "vertical_aperture_offset": (principal_point_y - frame_height / 2) * vertical_aperture / frame_height,
        "clipping_range": [args.clipping_range_near, args.clipping_range_far],
    }
