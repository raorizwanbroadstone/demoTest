"""
AWS Bedrock example using boto3.
"""

import json

import boto3

bedrock = boto3.client("bedrock-runtime")


def invoke():
    body = {
        "inputText": "Explain neural networks."
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps(body),
    )

    return response


if __name__ == "__main__":
    print(invoke())
