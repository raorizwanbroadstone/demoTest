"""
Multimodal AI example:
  - OpenAI vision model
"""

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def analyze_image():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/test.jpg"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(analyze_image())
