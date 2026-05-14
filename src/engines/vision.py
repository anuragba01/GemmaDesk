"""
vision.py - Visual Processing Engine

This module contains the VisionEngine class, which manages raw image files 
used by the multimodal LLM. Since images are not vectorized, this engine 
handles copying them to a persistent directory and tracking them via a JSON manifest.
"""
import json
import logging
import os
import shutil

log = logging.getLogger("rag.vision")


class VisionEngine:
    """
    Manages image tracking, storage, and retrieval for visual context.
    """
    def __init__(self, image_dir: str, manifest_path: str):
        """
        Initializes the VisionEngine.
        
        Args:
            image_dir: Directory where uploaded images are physically stored.
            manifest_path: Path to the JSON file that tracks indexed images.
        """
        self.image_dir = image_dir
        self.manifest_path = manifest_path
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_paths = self._load_manifest()

    def _load_manifest(self) -> list:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                return json.load(f)
        return []

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.image_paths, f, indent=2)

    def _sync_manifest(self):
        valid = [path for path in self.image_paths if os.path.exists(path)]
        if valid != self.image_paths:
            self.image_paths = valid
            self._save_manifest()

    def ingest_image(self, src_path: str) -> bool:
        """
        Copies an image from the source path to the internal storage directory 
        and registers it in the JSON manifest.
        
        Args:
            src_path: Absolute path to the original image file.
            
        Returns:
            bool: True if the image was newly ingested, False if it was already indexed.
        """
        dest = os.path.join(self.image_dir, os.path.basename(src_path))
        if dest in self.image_paths:
            log.info("Image already indexed: %s", dest)
            return False
        shutil.copy2(src_path, dest)
        self.image_paths.append(dest)
        self._save_manifest()
        log.info("Image registered: %s", dest)
        return True

    def get_valid_images(self, filter_paths: list = None) -> list:
        """
        Retrieves a list of verified image paths. Filters out any deleted images 
        and applies optional user filtering.
        
        Args:
            filter_paths: Optional list of specific paths to restrict the output to.
            
        Returns:
            list: List of absolute paths to valid images.
        """
        self._sync_manifest()
        valid = [path for path in self.image_paths if os.path.exists(path)]
        if filter_paths:
            valid = [path for path in valid if path in filter_paths]
        return valid

    def get_source_map(self) -> dict:
        return {os.path.basename(path): path for path in self.get_valid_images()}

    def get_stats(self) -> int:
        self._sync_manifest()
        return len(self.image_paths)

    def clear(self):
        """
        Deletes all stored images and the manifest file, effectively wiping 
        all visual context from the system.
        """
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        if os.path.exists(self.manifest_path):
            os.unlink(self.manifest_path)
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_paths = []
