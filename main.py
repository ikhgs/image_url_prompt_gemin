import os
import tempfile
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    prompt = request.form.get('prompt')
    image_url = request.form.get('image_url')

    if not prompt or not image_url:
        return jsonify({"error": "Image URL and prompt are required."}), 400

    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"error": "Failed to download image."}), 400

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        temp_file.write(response.content)

        # Upload the image to Gemini
        file_uri = upload_to_gemini(image_path, mime_type='image/jpeg')

        # Create the chat session with the image and prompt
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        file_uri,
                        prompt,
                    ],
                },
            ]
        )

        response = chat_session.send_message(prompt)

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Create the chat session with the prompt
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [prompt],
            },
        ]
    )

    response = chat_session.send_message(prompt)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
