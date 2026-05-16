"""
gemma.py - Gemma 4 LiteRT Integration Engine

This module provides the GemmaEngine class which wraps the litert_lm engine 
to handle multimodal (text, audio, image) RAG queries both synchronously 
and via streaming.
"""
import logging

import litert_lm

log = logging.getLogger("rag.gemma")


class GemmaEngine:
    """
    Engine for interacting with the Gemma 4 LiteRT model.
    Handles conversation history, multi-modal inputs (images, audio), and prompt construction.
    """

    def __init__(self, model_path: str):
        """
        Initializes the Gemma LiteRT engine.
        
        Args:
            model_path (str): The file path to the Gemma model.
        """
        log.info("Loading LiteRT engine (Gemma 4)...")
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        try:
            self.engine = litert_lm.Engine(
                model_path,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.GPU,
            )
            log.info("Gemma engine ready (GPU vision backend).")
        except Exception:
            log.warning("GPU backend unavailable. Falling back to CPU for all backends.")
            self.engine = litert_lm.Engine(
                model_path,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
            log.info("Gemma engine ready (CPU fallback).")

    def answer(self, payload: dict) -> str:
        """
        Generates a complete, synchronous response from the Gemma model.
        
        Args:
            payload (dict): A dictionary containing 'history', 'system_text', 'prompt_text', and 'image_paths'.
            
        Returns:
            str: The generated text response.
        """
        with self.engine.create_conversation() as conv:
            history = payload.get("history", [])
            image_paths = payload.get("image_paths", [])

            for msg in history:
                # Avoid anchoring a new visual turn to stale assistant text.
                if image_paths and msg["role"] == "assistant":
                    continue
                conv.send_message({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            content = []
            system_text = payload.get("system_text")
            prompt_text = payload.get("prompt_text")
            text_parts = []
            if system_text:
                text_parts.append(f"[SYSTEM]\n{system_text}")
            if prompt_text:
                text_parts.append(prompt_text)
            if text_parts:
                content.append({"type": "text", "text": "\n\n".join(text_parts)})

            for media_path in image_paths:
                ext = media_path.lower().split('.')[-1]
                if ext in ['wav', 'mp3']:
                    content.append({"type": "audio", "path": media_path})
                else:
                    content.append({"type": "image", "path": media_path})

            response = conv.send_message({"role": "user", "content": content})
            return response["content"][0]["text"]

    def answer_stream(self, payload: dict):
        """
        Generates a streaming response from the Gemma model.
        
        Args:
            payload (dict): A dictionary containing 'history', 'system_text', 'prompt_text', and 'image_paths'.
            
        Yields:
            str: Chunks of the generated text response.
        """
        with self.engine.create_conversation() as conv:
            history = payload.get("history", [])
            image_paths = payload.get("image_paths", [])

            for msg in history:
                if image_paths and msg["role"] == "assistant":
                    continue
                conv.send_message({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            content = []
            system_text = payload.get("system_text")
            prompt_text = payload.get("prompt_text")
            text_parts = []
            if system_text:
                text_parts.append(f"[SYSTEM]\n{system_text}")
            if prompt_text:
                text_parts.append(prompt_text)
            if text_parts:
                content.append({"type": "text", "text": "\n\n".join(text_parts)})

            for media_path in image_paths:
                ext = media_path.lower().split('.')[-1]
                if ext in ['wav', 'mp3']:
                    content.append({"type": "audio", "path": media_path})
                else:
                    content.append({"type": "image", "path": media_path})

            stream = conv.send_message_async({"role": "user", "content": content})
            try:
                for chunk in stream:
                    for item in chunk.get("content", []):
                        if item.get("type") == "text":
                            yield item["text"]
            except GeneratorExit:
                log.info("Streamlit aborted stream. Generator explicitly closed. Releasing Litert session lock.")
                return
