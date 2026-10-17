import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.environ["AMD_API_KEY"],
    base_url=os.environ["AMD_BASE_URL"],
)

models = client.models.list()

print("Connected to AMD! Available models:")
for model in models.data:
    print(model.id)