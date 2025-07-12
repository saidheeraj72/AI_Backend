from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

@app.post('/chat')
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    llm_type = data.get('llm_type', 'daily')
    chat_history = data.get('history', [])

    if not user_message:
        raise HTTPException(status_code=400, detail='`message` is required')
    if llm_type not in LLM_MODELS:
        raise HTTPException(status_code=400, detail='Invalid `llm_type`')

    config = LLM_MODELS[llm_type]
    # Build messages
    messages = [{'role': 'system', 'content': config['system_prompt']}]
    for msg in chat_history[-10:]:
        messages.append({'role': msg.get('role'), 'content': msg.get('content')})
    messages.append({'role': 'user', 'content': user_message})

    try:
        completion = groq_client.chat.completions.create(
            messages=messages,
            model=config['model'],
            temperature=0.7,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content
        return JSONResponse({
            'response': response_text,
            'llm_type': llm_type,
            'model_name': config['name'],
            'status': 'success'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/stream')
async def chat_stream(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    llm_type = data.get('llm_type', 'daily')
    chat_history = data.get('history', [])

    if not user_message:
        raise HTTPException(status_code=400, detail='`message` is required')
    if llm_type not in LLM_MODELS:
        raise HTTPException(status_code=400, detail='Invalid `llm_type`')

    config = LLM_MODELS[llm_type]
    messages = [{'role': 'system', 'content': config['system_prompt']}]
    for msg in chat_history[-10:]:
        messages.append({'role': msg.get('role'), 'content': msg.get('content')})
    messages.append({'role': 'user', 'content': user_message})

    async def event_generator():
        try:
            stream = groq_client.chat.completions.create(
                messages=messages,
                model=config['model'],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@app.get('/models')
async def get_models():
    return JSONResponse({'models': [
        {'id': k, 'name': v['name'], 'model': v['model']} for k, v in LLM_MODELS.items()
    ]})

@app.get('/health')
async def health_check():
    return JSONResponse({'status': 'healthy'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
