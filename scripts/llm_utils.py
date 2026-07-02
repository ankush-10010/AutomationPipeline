import json
import requests
import sys
import time
from pathlib import Path

# Need to load the config
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"

def get_llm_config():
    from scripts.config_loader import load_pipeline_config
    return load_pipeline_config().get("llm", {})

def generate_text(prompt, system_prompt=None, base64_image=None):
    """
    Universal wrapper to generate text from the configured LLM engine.
    Supports both Ollama and Gemini APIs.
    """
    llm_config = get_llm_config()
    engine = llm_config.get("engine", "ollama").lower()
    
    if engine == "ollama":
        return _generate_ollama(prompt, system_prompt, base64_image, llm_config.get("ollama", {}))
    elif engine == "gemini":
        return _generate_gemini(prompt, system_prompt, base64_image, llm_config.get("gemini", {}))
    else:
        raise ValueError(f"Unknown LLM engine: {engine}")

def _generate_ollama(prompt, system_prompt, base64_image, config):
    url = f"{config.get('base_url', 'http://localhost:11434')}/api/generate"
    model = config.get("model", "llama3.1:8b")
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.get("temperature", 0.8),
            "num_predict": config.get("max_tokens", 4096)
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
        
    if base64_image:
        payload["images"] = [base64_image]

    try:
        response = requests.post(url, json=payload, timeout=config.get("timeout_seconds", 300))
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return ""

def _generate_gemini(prompt, system_prompt, base64_image, config):
    api_key = config.get("api_key")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("Gemini API key is not set in pipeline_config.yaml")
        
    model = config.get("model", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    parts = [{"text": prompt}]
    if base64_image:
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_image
            }
        })
        
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": config.get("temperature", 0.8),
            "maxOutputTokens": config.get("max_tokens", 4096)
        }
    }
    
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    max_retries = 7
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=config.get("timeout_seconds", 300))
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 503]:
                wait_time = min(60, (2 ** attempt) * 5)  # 5s, 10s, 20s, 40s, 60s, 60s, 60s
                print(f"Gemini API Error ({response.status_code}). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"Gemini API Error: {response.text}")
            return ""
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return ""
            
    print("Gemini API Error: Max retries exceeded.")
    return ""
