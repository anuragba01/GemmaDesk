import litert_lm
import uuid
import httpx
import tempfile
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI()

engine = litert_lm.Engine(
    "./gemma-4-it.litertlm",
    audio_backend=litert_lm.Backend.CPU,
    vision_backend=litert_lm.Backend.GPU,
)

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stream: bool = False



async def download_file(url: str) -> str:
    ""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        tmp.write(resp.content)
        tmp.close()
        return tmp.name


def normalize_content(content):
    ""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


async def process_content(content):
    ""
    processed = []
    temp_files = []

    content = normalize_content(content)

    for block in content:
        if block.get("type") == "text":
            processed.append({"type": "text", "text": block.get("text", "")})

        elif block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url")
            if url:
                local_path = await download_file(url)
                temp_files.append(local_path)
                processed.append({"type": "image", "path": local_path})
                
        elif block.get("type") == "image":
            # Added for local GemmaDesk integration
            processed.append({"type": "image", "path": block.get("path")})

        elif block.get("type") == "audio":
            processed.append({"type": "audio", "path": block.get("path")})

    return processed, temp_files


def cleanup_files(paths: List[str]):
    for p in paths:
        try:
            os.unlink(p)
        except:
            pass


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    temp_files = []

    try:
        with engine.create_conversation() as conv:
            # Replay full history (stateless API)
            for msg in req.messages[:-1]:
                role = msg.role
                content = normalize_content(msg.content)

                if role == "system":
                    role = "user"
                    content = [{"type": "text", "text": f"[SYSTEM]\n{msg.content[0]['text'] if isinstance(msg.content, list) else msg.content}"}]

                conv.send_message({
                    "role": role,
                    "content": content
                })

            # Process last message
            last_msg = req.messages[-1]
            role = last_msg.role

            processed_content, files = await process_content(last_msg.content)
            temp_files.extend(files)

            if role == "system":
                role = "user"

            response = conv.send_message(
                {
                    "role": role,
                    "content": processed_content
                },
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            text_out = response["content"][0]["text"]

        # Streaming (basic SSE)
        if req.stream:
            async def stream():
                yield f"data: {JSONResponse(content={'choices':[{'delta':{'content': text_out}}]}).body.decode()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        # Normal response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_out
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )

    finally:
        cleanup_files(temp_files)


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "gemma-4",
                "object": "model",
                "owned_by": "litert"
            }
        ]
    }
