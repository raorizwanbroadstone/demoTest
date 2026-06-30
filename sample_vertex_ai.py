"""
Google Vertex AI example.
"""

import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "demo-project"
LOCATION = "us-central1"


def generate():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    model = GenerativeModel("gemini-1.5-pro")

    response = model.generate_content(
        "Explain machine learning in one sentence."
    )

    return response.text


if __name__ == "__main__":
    print(generate())
