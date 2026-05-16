"""
chat_ingestion.py - Chat History Vectorizer

Reads a JSONL chat session file and embeds it into ChromaDB in blocks of 8 messages.
Triggered whenever `message_id % 8 == 0` during a live session.

Each embedded chunk gets metadata:
  - type: "chat"
  - session_id: <the current session UUID>
  - block: <which block number, e.g. 1 for messages 1-8, 2 for 9-16 ...>
"""
import os
import json
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

log = logging.getLogger("rag.chat_ingestion")


class ChatHistoryIngestion:
    """
    Vectorizes blocks of 8 messages from a JSONL chat session file into ChromaDB.
    Called every time the conversation hits a multiple-of-8 message count.
    """

    def __init__(self, chat_path: str, database_dir: str, embed_model: str, session_id: str):
        """
        Args:
            chat_path:    Absolute path to the session .jsonl file.
            database_dir: ChromaDB persist directory (same one used by VectorStoreEngine).
            embed_model:  HuggingFace model id for embeddings.
            session_id:   The UUID of the chat session (used as metadata for prefiltering).
        """
        self.chat_path = chat_path
        self.database_dir = database_dir
        self.embed_model = embed_model
        self.session_id = session_id

    def _load_embeddings(self):
        """Loads the local embedding model used by the main vector store."""
        return FastEmbedEmbeddings(model_name=self.embed_model)

    def _read_messages(self) -> list[dict]:
        """
        Reads all message lines from the JSONL file (skipping the metadata header).
        Returns a list of message dicts with sequential 'id' fields.
        """
        messages = []
        if not os.path.exists(self.chat_path):
            log.warning("Chat JSONL not found: %s", self.chat_path)
            return messages

        with open(self.chat_path, "r") as f:
            lines = f.readlines()

        # Line 0 is the metadata header — skip it
        for line in lines[1:]:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return messages

    def _format_block(self, block: list[dict]) -> str:
        """Formats a list of messages into a readable text block for embedding."""
        parts = []
        for msg in block:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def ingest_latest_block(self) -> int:
        """
        Reads the session file and embeds only the LATEST block of 8 messages
        (i.e. the most recently completed chunk).

        This should be called when `len(messages) % 8 == 0`.

        Returns:
            int: 1 if a block was indexed successfully, 0 otherwise.
        """
        messages = self._read_messages()
        if not messages:
            log.warning("No messages found in %s", self.chat_path)
            return 0

        total = len(messages)
        if total % 8 != 0:
            log.info("Message count %d is not a multiple of 8 — skipping ingestion.", total)
            return 0

        # Determine the latest completed block (1-indexed)
        block_number = total // 8
        block_start = (block_number - 1) * 8   # 0-based index
        block_end = block_number * 8             # exclusive

        block = messages[block_start:block_end]
        if not block:
            return 0

        block_text = self._format_block(block)

        # Get the ID range covered by this block for traceability
        first_id = block[0].get("id", block_start + 1)
        last_id = block[-1].get("id", block_end)

        doc = Document(
            page_content=block_text,
            metadata={
                "type": "chat",
                "session_id": self.session_id,
                "block": block_number,
                "id_range": f"{first_id}-{last_id}",
                "source": self.chat_path,
            },
        )

        log.info(
            "Indexing chat block %d (messages %d–%d) for session %s",
            block_number, first_id, last_id, self.session_id,
        )

        try:
            embeddings = self._load_embeddings()
            vectorstore = Chroma(
                persist_directory=self.database_dir,
                embedding_function=embeddings,
                collection_name="text_docs",
            )
            vectorstore.add_documents([doc])
            log.info("Chat block %d successfully indexed.", block_number)
            return 1
        except Exception as e:
            log.error("Failed to index chat block: %s", e)
            return 0
