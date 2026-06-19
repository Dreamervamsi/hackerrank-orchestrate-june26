"""Hugging Face VLM client for local or remote inference."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient
from PIL import Image

from config import (
    HF_MODEL_ID,
    HF_PROVIDER,
    HF_TOKEN,
    MAX_JSON_RETRIES,
    MAX_TOKENS,
    TEMPERATURE,
    USE_LOCAL,
)
from data.loader import ImageRef
from pipeline.prompts import build_json_repair_message, build_system_prompt
from schema.columns import OUTPUT_COLUMNS
from schema.json_parser import JSONParseError, parse_prediction_json

PREDICTION_KEYS = list(OUTPUT_COLUMNS)


class HFVLMClient:
    """Vision-language client with remote Inference Providers and optional local mode."""

    def __init__(
        self,
        model_id: str | None = None,
        use_local: bool | None = None,
        provider: str | None = None,
        token: str | None = None,
        max_json_retries: int | None = None,
    ) -> None:
        self.model_id = model_id or HF_MODEL_ID
        self.use_local = USE_LOCAL if use_local is None else use_local
        self.provider = provider or HF_PROVIDER
        self.token = token or HF_TOKEN
        self.max_json_retries = MAX_JSON_RETRIES if max_json_retries is None else max_json_retries

        self._local_model = None
        self._local_processor = None
        self._remote_client: InferenceClient | None = None

        if self.use_local:
            self._init_local()
        else:
            self._init_remote()

    def _init_remote(self) -> None:
        if not self.token:
            raise ValueError("HF_TOKEN is required for remote inference")
        kwargs: dict[str, Any] = {"token": self.token}
        if self.provider and self.provider.lower() != "auto":
            kwargs["provider"] = self.provider
        self._remote_client = InferenceClient(**kwargs)

    def _init_local(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Local mode requires: pip install torch transformers qwen-vl-utils"
            ) from exc

        self._local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self._local_processor = AutoProcessor.from_pretrained(self.model_id)
        self._torch = torch

    @staticmethod
    def _image_to_base64_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _build_api_messages(self, system_prompt: str, user_text: str, images: list[ImageRef]) -> list[dict]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image in images:
            if not image.exists:
                content.append(
                    {
                        "type": "text",
                        "text": f"[Missing image file for {image.image_id} at {image.path}]",
                    }
                )
                continue
            content.append({"type": "text", "text": f"Image ID: {image.image_id}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_base64_url(image.absolute_path)},
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _call_remote(self, messages: list[dict]) -> str:
        assert self._remote_client is not None
        response = self._remote_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    def _call_local(self, system_prompt: str, user_text: str, images: list[ImageRef]) -> str:
        assert self._local_model is not None and self._local_processor is not None

        pil_images: list[Image.Image] = []
        image_notes = []
        for image in images:
            if image.exists:
                pil_images.append(Image.open(image.absolute_path).convert("RGB"))
                image_notes.append(f"Image ID: {image.image_id}")
            else:
                image_notes.append(f"[Missing image file for {image.image_id}]")

        prompt_text = f"{system_prompt}\n\n{user_text}\n\n" + "\n".join(image_notes)
        inputs = self._local_processor(
            text=[prompt_text],
            images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self._local_model.device)

        generated = self._local_model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self._local_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def complete(self, system_prompt: str, user_text: str, images: list[ImageRef] | None = None) -> str:
        images = images or []
        if self.use_local:
            return self._call_local(system_prompt, user_text, images)
        messages = self._build_api_messages(system_prompt, user_text, images)
        return self._call_remote(messages)

    def complete_json(
        self,
        user_text: str,
        images: list[ImageRef] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Call the model and parse a prediction JSON object with retries."""
        system_prompt = system_prompt or build_system_prompt()
        images = images or []
        messages_history: list[tuple[str, list[ImageRef] | None]] = [(user_text, images)]
        last_error: Exception | None = None
        raw_response = ""

        for attempt in range(self.max_json_retries + 1):
            current_text, current_images = messages_history[-1]
            raw_response = self.complete(system_prompt, current_text, current_images)
            try:
                return parse_prediction_json(raw_response, PREDICTION_KEYS)
            except JSONParseError as exc:
                last_error = exc
                if attempt >= self.max_json_retries:
                    break
                repair_text = (
                    f"{build_json_repair_message(PREDICTION_KEYS)}\n\n"
                    f"Previous invalid response:\n{raw_response}"
                )
                messages_history.append((repair_text, None))

        raise JSONParseError(
            f"Failed to parse JSON after {self.max_json_retries + 1} attempts: {last_error}\n"
            f"Last response:\n{raw_response}"
        ) from last_error
