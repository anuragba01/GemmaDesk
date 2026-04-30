import json
import logging
import os
import shutil

log = logging.getLogger("rag.vision")


class VisionEngine:
    def __init__(self, image_dir: str, manifest_path: str):
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
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        if os.path.exists(self.manifest_path):
            os.unlink(self.manifest_path)
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_paths = []
