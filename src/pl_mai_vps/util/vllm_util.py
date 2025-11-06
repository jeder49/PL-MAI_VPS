from typing import Any

import cv2
from PIL import Image
from numpy.typing import NDArray


def convert_from_cv2_to_image(img: NDArray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class VLLMPrompt:

    def __init__(self, conversation: list[Any] | None = None):
        super().__init__()

        if conversation is None:
            conversation = []
        self.conversation = conversation

    def clone(self) -> "VLLMPrompt":
        return VLLMPrompt(self.conversation.copy())

    def add_image_from_url(self, role: str, url: str):
        self.conversation.append({
            "role": role,
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }]
        })

    def add_image_from_cv2(self, role: str, image: NDArray):
        self.conversation.append({
            "role": role,
            "content": [{
                "type": "image_pil",
                "image_pil": convert_from_cv2_to_image(image),
            }]
        })

    def add_text(self, role: str, text: str):
        self.conversation.append({
            "role": role,
            "content": text
        })

    def add_system_text(self, text: str):
        self.add_text("system", text)

    def add_user_text(self, text: str):
        self.add_text("user", text)

    def add_assistant_text(self, text: str):
        self.add_text("assistant", text)

    def add_system_image_from_url(self, url: str):
        self.add_image_from_url("system", url)

    def add_user_image_from_url(self, url: str):
        self.add_image_from_url("user", url)

    def add_assistant_image_from_url(self, url: str):
        self.add_image_from_url("assistant", url)

    def add_system_image_from_cv2(self, image: NDArray):
        self.add_image_from_cv2("system", image)

    def add_user_image_from_cv2(self, image: NDArray):
        self.add_image_from_cv2("user", image)

    def add_assistant_image_from_cv2(self, image: NDArray):
        self.add_image_from_cv2("assistant", image)

    def build(self) -> Any:
        return self.conversation
