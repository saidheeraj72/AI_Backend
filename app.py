from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
from dotenv import load_dotenv
import json 

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# LLM Model configurations
LLM_MODELS = {
    'daily': {
        'model': 'gemma2-9b-it',
        'name': 'Daily Use LLM',
        'system_prompt': 'You are a helpful assistant for everyday tasks. Provide clear, concise answers.'
    },
    'image': {
        'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
        'name': 'Image Reasoning LLM', 
        'system_prompt': 'You are an expert in image analysis and visual reasoning. Provide detailed insights about images and visual content.'
    },
    'complex': {
        'model': 'qwen-qwq-32b',
        'name': 'Complex Tasks LLM',
        'system_prompt': 'You are an advanced AI assistant capable of handling complex reasoning, analysis, and problem-solving tasks. Provide thorough, well-reasoned responses.'
    }
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get request data
        data = request.get_json()
        user_message = data.get('message', '')
        llm_type = data.get('llm_type', 'daily')
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
            
        if llm_type not in LLM_MODELS:
            return jsonify({'error': 'Invalid LLM type'}), 400
        
        # Get model configuration
        model_config = LLM_MODELS[llm_type]
        
        # Build messages with chat history
        messages = [
            {
                "role": "system",
                "content": model_config['system_prompt']
            }
        ]
        
        # Add chat history
        for msg in chat_history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Send message to Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_config['model'],
            temperature=0.7,
            max_tokens=1024,
        )
        
        # Extract response
        response = chat_completion.choices[0].message.content
        
        return jsonify({
            'response': response,
            'llm_type': llm_type,
            'model_name': model_config['name'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    try:
        # Get request data
        data = request.get_json()
        user_message = data.get('message', '')
        llm_type = data.get('llm_type', 'daily')
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
            
        if llm_type not in LLM_MODELS:
            return jsonify({'error': 'Invalid LLM type'}), 400
        
        # Get model configuration
        model_config = LLM_MODELS[llm_type]
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": model_config['system_prompt']
            }
        ]
        
        # Add chat history
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Stream response
        def generate():
            try:
                stream = client.chat.completions.create(
                    messages=messages,
                    model=model_config['model'],
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                        
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return app.response_class(generate(), mimetype='text/plain')
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available LLM models"""
    models = []
    for key, config in LLM_MODELS.items():
        models.append({
            'id': key,
            'name': config['name'],
            'model': config['model']
        })
    return jsonify({'models': models})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
