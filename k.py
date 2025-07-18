import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()
def analyze_image(image_url: str) -> dict:
    """
    Sends a chat completion request with an image URL to OpenRouter.ai
    and returns the parsed JSON response.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemini-2.5-pro-exp-03-25",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()  # will raise an error for non-2xx responses
    return response.json()

if __name__ == "__main__":
    IMAGE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )

    result = analyze_image(IMAGE_URL)
    # Pretty-print the full JSON response
    print(json.dumps(result, indent=2))
