import logging

import litert_lm

log = logging.getLogger("rag.gemma")


class GemmaEngine:
    def __init__(self, model_path: str):
        log.info("Loading LiteRT engine (Gemma 4)...")
        self.engine = litert_lm.Engine(
            model_path,
            audio_backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.GPU,
        )
        log.info("Gemma engine ready.")

    def answer(self, payload: dict) -> str:
        with self.engine.create_conversation() as conv:
            system_text = payload.get("system_text")
            if system_text:
                conv.send_message({
                    "role": "user",
                    "content": [{"type": "text", "text": f"[SYSTEM]\n{system_text}"}],
                })

            for msg in payload.get("history", []):
                conv.send_message({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            content = []
            prompt_text = payload.get("prompt_text")
            if prompt_text:
                content.append({"type": "text", "text": prompt_text})

            for image_path in payload.get("image_paths", []):
                content.append({"type": "image", "path": image_path})

            response = conv.send_message({"role": "user", "content": content})
            return response["content"][0]["text"]
