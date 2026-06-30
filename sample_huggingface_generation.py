"""
Hugging Face generative model usage:
  - AutoModelForCausalLM
  - AutoTokenizer
  - text-generation pipeline
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

MODEL = "gpt2"


def generate_text(prompt):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return generator(prompt, max_new_tokens=50)


if __name__ == "__main__":
    print(generate_text("Artificial intelligence"))
