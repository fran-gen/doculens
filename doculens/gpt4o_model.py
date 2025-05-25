import base64
import os
import logging
from dotenv import load_dotenv
from typing import Any
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI

from base import BaseVLMInferenceModel
from utils.gpt import GPT4VDetail

log = logging.getLogger(__name__)

load_dotenv()

langfuse = Langfuse()

content = "You are a helpful assistant that reads off pictures, extracting text passages and analyzing graphics."

class GPT4OInferenceModel(BaseVLMInferenceModel):
    """A class for interacting with the Gpt4v model."""

    def __init__(
        self,
        model_path: str = "gpt-4o",
        api_key: str | None = os.environ.get("OPENAI_API_KEY", None),
        base_url: str | None = os.environ.get("OPENAI_API_BASE", None),
    ):
        if api_key is None:
            raise ValueError("The OPENAI_API_KEY environment variable must be set.")

        self.model_path = model_path
        self.model = OpenAI(api_key=api_key, base_url=base_url)

    @observe(name="chat_function", capture_input=False, capture_output=False)
    def chat(
        self,
        image_bytes: bytes,
        query: str,
        max_tokens: int = 300,
        temperature: float = 0.1,
        detail: GPT4VDetail = GPT4VDetail.HIGH,
        **extra_chat_args: dict[str, Any],
    ) -> str:
        """Chat with the model using the given image bytes and query."""

        # Encode image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Langfuse generation tracking
        generation = langfuse.generation(
            name="chat_generation",
            model="gpt-4o",
            input={"max_tokens": max_tokens, "temperature": temperature},
            trace_id=langfuse_context.get_current_trace_id(),
            parent_observation_id=langfuse_context.get_current_observation_id(),
        )

        try:
            # Call the model's API to get a response
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[
                    {
                        "role": "system",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail.value},
                            },
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_chat_args,
            )  # type: ignore[call-overload]

            # Directly return the response's message content
            response_text = response.choices[0].message.content

            # Log output to Langfuse
            generation.end(output={"response_text": response_text})

        except Exception as e:
            # Log the error to Langfuse
            generation.end(error=str(e))
            raise e

        return response_text

    def unload(self) -> None:
        """Unload the model."""
        pass


if __name__ == "__main__":
    # Example usage
    # Use image bytes and query from RAG output
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAABqQAAAiYCAIAAAA+NVHkAAEAAElEQVR4nOzdd1gUx/8H8KGDoCJNwY4gdkDs2MFoxIZiV+wNu9HERI36t..."
    )  # Truncated Base64 string
    query = "Tell me what the BLEU score for the transformer base model is."

    # Initialize the model
    model = GPT4OInferenceModel()

    # Call the chat method with image bytes and query
    response = model.chat(
        image_bytes=image_bytes,
        query=query,
        max_tokens=2048,
        temperature=0.0,
    )
    print(response)
