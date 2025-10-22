"""
Batch-Konverter für EGAD-Objekte:
- Durchsucht --input_root rekursiv nach .obj-Dateien
- Konvertiert sie zu .usd mit identischer Ordnerstruktur unter --output_root
- Funktioniert auch ohne aktive GPU / NVIDIA-Treiber (CPU-only fallback)
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import List

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Prüfe, ob GPU verfügbar ist (bzw. ob nvidia-smi funktioniert)


def has_gpu() -> bool:
    try:
        import subprocess
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except Exception:
        return False


def find_obj_files_recursive(root_dir: str) -> List[str]:
    out = []
    for dp, dn, fn in os.walk(root_dir):
        for f in fn:
            if f.lower().endswith(".obj"):
                out.append(os.path.join(dp, f))
    return sorted(out)


def make_output_path(input_path: str, input_root: str, output_root: str) -> str:
    rel = os.path.relpath(input_path, input_root)
    rel_usd = os.path.splitext(rel)[0] + ".usd"
    out_path = os.path.join(output_root, rel_usd)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("obj_to_usd_recursive")

    parser = argparse.ArgumentParser(
        description="Recursively convert all .obj files to .usd format.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.input_root):
        logger.error("Input root does not exist: %s", args.input_root)
        sys.exit(1)
    os.makedirs(args.output_root, exist_ok=True)

    gpu_available = has_gpu()
    if gpu_available:
        logger.info("GPU detected — running normal Isaac Sim mode.")
    else:
        logger.warning(
            "No GPU or driver found — running CPU-only/headless fallback mode.")

    # Headless fallback config
    sim_config = {
        "headless": True,
        "renderer": "RayTracedLighting" if gpu_available else None,
        "renderer.enable": gpu_available,
        "audio.enabled": False,
        "app.filecache.enabled": False,
        "app.window.width": 1,
        "app.window.height": 1,
    }

    # Import SimulationApp
    from isaacsim import SimulationApp
    kit = SimulationApp(sim_config)

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")

    try:
        from asset_converter.obj_to_usd import convert as convert_single
    except Exception as e:
        logger.error("Failed to import converter: %s", e)
        kit.close()
        sys.exit(1)

    objs = find_obj_files_recursive(args.input_root)
    logger.info("Found %d OBJ files.", len(objs))

    loop = asyncio.get_event_loop()
    converted, failed = 0, 0
    for idx, in_path in enumerate(objs, 1):
        out_path = make_output_path(in_path, args.input_root, args.output_root)
        logger.info("(%d/%d) Converting %s -> %s",
                    idx, len(objs), in_path, out_path)
        try:
            loop.run_until_complete(convert_single(in_path, out_path))
            converted += 1
        except Exception as e:
            logger.exception("Failed to convert %s: %s", in_path, e)
            failed += 1

    logger.info("Done. Converted: %d, Failed: %d", converted, failed)
    kit.close()


if __name__ == "__main__":
    main()
