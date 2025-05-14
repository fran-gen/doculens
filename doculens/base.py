from abc import ABC, abstractmethod
from typing import Any


class BaseVLMInferenceModel(ABC):
    """A class for interacting with the CogVLM model."""

    @abstractmethod
    def chat(
        self,
        images: list[str] | str,
        prompt: str,
        *args,
        **kwargs: Any,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Chat with the model using the given image and prompt.

        Args:
            images: The path to the images to chat about.
            prompt: The prompt to give to the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The model's response, and the updated history.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Unload the model."""
        raise NotImplementedError
