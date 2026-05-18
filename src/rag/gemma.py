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
        self.model_path = model_path
        self._load_engine()

    def _load_engine(self):
        log.info("Loading LiteRT engine (Gemma 4)...")
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        try:
            self.engine = litert_lm.Engine(
                self.model_path,
                audio_backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
            log.info("Gemma engine ready (CPU backend).")
        except Exception as e:
            log.error(f"Failed to load Gemma engine: {e}")
            raise

    def reset(self):
        """Forcefully destroys the current engine and re-instantiates it to clear deadlocks."""
        log.warning("Force resetting Gemma engine to clear stale session locks...")
        self.engine = None
        self._load_engine()

    def answer(self, payload: dict) -> str:
        """
        Generates a complete, synchronous response from the Gemma model.
        """
        try:
            return self._answer_internal(payload)
        except Exception as e:
            if "session already exists" in str(e).lower() or "FAILED_PRECONDITION" in str(e):
                log.warning("Caught session deadlock in answer(). Resetting engine and retrying...")
                self.reset()
                return self._answer_internal(payload)
            raise

    def _answer_internal(self, payload: dict) -> str:
        history = payload.get("history", [])
        image_paths = payload.get("image_paths", [])

        preface_messages = []
        for msg in history:
            # Avoid anchoring a new visual turn to stale assistant text.
            if image_paths and msg["role"] == "assistant":
                continue
            preface_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })

        with self.engine.create_conversation(messages=preface_messages) as conv:

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
        """
        try:
            yield from self._answer_stream_internal(payload)
        except Exception as e:
            if "session already exists" in str(e).lower() or "FAILED_PRECONDITION" in str(e):
                log.warning("Caught session deadlock in answer_stream(). Resetting engine and retrying...")
                self.reset()
                yield from self._answer_stream_internal(payload)
            else:
                raise

    def _answer_stream_internal(self, payload: dict):
        print("[GemmaEngine] Entering _answer_stream_internal. Opening C++ conversation session...", flush=True)
        try:
            history = payload.get("history", [])
            image_paths = payload.get("image_paths", [])

            preface_messages = []
            for msg in history:
                # Avoid anchoring a new visual turn to stale assistant text.
                if image_paths and msg["role"] == "assistant":
                    continue
                preface_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            print(f"[GemmaEngine] Initializing C++ conversation session with {len(preface_messages)} history messages...", flush=True)
            with self.engine.create_conversation(messages=preface_messages) as conv:
                print("[GemmaEngine] C++ conversation session successfully created.", flush=True)

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

                print(f"[GemmaEngine] Attaching {len(image_paths)} media assets to the conversation payload...", flush=True)
                for media_path in image_paths:
                    ext = media_path.lower().split('.')[-1]
                    if ext in ['wav', 'mp3']:
                        content.append({"type": "audio", "path": media_path})
                    else:
                        content.append({"type": "image", "path": media_path})

                print("[GemmaEngine] Sending payload async to LiteRT model...", flush=True)
                stream = conv.send_message_async({"role": "user", "content": content})
                
                first_chunk = True
                try:
                    for chunk in stream:
                        if first_chunk:
                            print("[GemmaEngine] First response chunk received from LiteRT!", flush=True)
                            first_chunk = False
                        for item in chunk.get("content", []):
                            if item.get("type") == "text":
                                yield item["text"]
                except GeneratorExit:
                    print("[GemmaEngine] GeneratorExit detected! Streamlit aborted stream. Releasing C++ session lock...", flush=True)
                    raise
                else:
                    print("[GemmaEngine] Stream finished naturally.", flush=True)
        finally:
            print("[GemmaEngine] Exited _answer_stream_internal. C++ conversation context destroyed.", flush=True)
