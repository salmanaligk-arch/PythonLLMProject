import os
import json
import gradio as gr
from services.llm_manager import llm_manager
from services.embed_manager import embed_manager
from services.rag_agents import agent_manager


def save_general_settings(llm_name, embed_name, agent_name, chunk_size, chunk_overlap):
    status_msgs = []
    try:
        if llm_name:
            llm_manager.set_selected(llm_name)
            status_msgs.append(f"LLM set: {llm_name}")
        if embed_name:
            embed_manager.set_selected(embed_name)
            status_msgs.append(f"Embedding set: {embed_name}")
        if agent_name:
            status_msgs.append(f"Agent default set: {agent_name}")

        settings_path = os.path.join(os.getcwd(), "data", "settings.json")
        payload = {
            "default_llm": llm_name,
            "default_embedding": embed_name,
            "default_agent": agent_name,
            "chunk_size": int(chunk_size or 1000),
            "chunk_overlap": int(chunk_overlap or 200)
        }
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            status_msgs.append(f"Settings saved to {settings_path}")
        except Exception as e:
            status_msgs.append(f"Failed to write settings file: {e}")

        llm_update = gr.update(choices=llm_manager.get_names(), value=llm_manager.get_selected())
        embed_update = gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected())
        agent_update = gr.update(choices=agent_manager.get_available_agents(), value=agent_name)
        return "\n".join(status_msgs), llm_update, embed_update, agent_update
    except Exception as e:
        return f"Error saving settings: {e}", None, None, None

def load_settings():
    """Read `data/default_settings.json` and return flat UI values without writing.

    Returns: llm_name, embed_name, agent_name, chunk_size, chunk_overlap, status_message
    """
    default_path = os.path.join(os.getcwd(), "data", "settings.json")
    if not os.path.exists(default_path):
        return None, None, None, None, None, "No settings.json found"
    try:
        with open(default_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        # Support both flat and nested structures
        llm = payload.get('default_llm') or payload.get('llm_config', {}).get('hf_model') or payload.get('llm_config', {}).get('ollama_model') or payload.get('llm_config', {}).get('model')
        embed = payload.get('default_embedding') or payload.get('file_config', {}).get('embedding_model')
        agent = payload.get('default_agent') or payload.get('agent_config', {}).get('default_agent')
        chunk_size = payload.get('chunk_size') or payload.get('file_config', {}).get('max_chunk_size') or int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = payload.get('chunk_overlap') or payload.get('file_config', {}).get('chunk_overlap') or int(os.getenv('CHUNK_OVERLAP', 200))

        return llm, embed, agent, int(chunk_size), int(chunk_overlap), f"Loaded defaults from {default_path}"
    except Exception as e:
        return None, None, None, None, None, f"Failed to load defaults: {e}"


def load_defaults():
    """Read `data/default_settings.json` and return flat UI values without writing.

    Returns: llm_name, embed_name, agent_name, chunk_size, chunk_overlap, status_message
    """
    default_path = os.path.join(os.getcwd(), "data", "default_settings.json")
    if not os.path.exists(default_path):
        return None, None, None, None, None, "No default_settings.json found"
    try:
        with open(default_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        # Support both flat and nested structures
        llm = payload.get('default_llm') or payload.get('llm_config', {}).get('hf_model') or payload.get('llm_config', {}).get('ollama_model') or payload.get('llm_config', {}).get('model')
        embed = payload.get('default_embedding') or payload.get('file_config', {}).get('embedding_model')
        agent = payload.get('default_agent') or payload.get('agent_config', {}).get('default_agent')
        chunk_size = payload.get('chunk_size') or payload.get('file_config', {}).get('max_chunk_size') or int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = payload.get('chunk_overlap') or payload.get('file_config', {}).get('chunk_overlap') or int(os.getenv('CHUNK_OVERLAP', 200))

        return llm, embed, agent, int(chunk_size), int(chunk_overlap), f"Loaded defaults from {default_path}"
    except Exception as e:
        return None, None, None, None, None, f"Failed to load defaults: {e}"


def get_llms_table():
    rows = []
    for name in llm_manager.get_names():
        cfg = llm_manager.get_config(name) or {}
        provider = cfg.get("provider", "openai")
        model = cfg.get("model", "")
        rows.append([name, provider, model])
    return rows


def add_llm(name, provider, model, base_url, api_key, timeout, max_retries, temperature):
    if not name:
        return get_llms_table(), None, f"Name is required"
    if not base_url:
        return get_llms_table(), None, "Base URL is required"
    if not api_key:
        return get_llms_table(), None, "API Key is required"
    cfg = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": int(timeout or 30),
        "max_retries": int(max_retries or 2),
        "temperature": float(temperature or 0.1)
    }
    try:
        llm_manager.register(name, cfg)
        return get_llms_table(), gr.update(choices=llm_manager.get_names(), value=llm_manager.get_selected()), "LLM added"
    except Exception as e:
        return get_llms_table(), None, f"Error adding LLM: {e}"


def get_llm_for_form(name):
    cfg = llm_manager.get_config(name) or {}
    provider = cfg.get("provider", "openai")
    model = cfg.get("model", "")
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")
    timeout = int(cfg.get("timeout", 30))
    max_retries = int(cfg.get("max_retries", 2))
    temperature = float(cfg.get("temperature", 0.1))
    return name, provider, model, base_url, api_key, timeout, max_retries, temperature


def remove_llm(name):
    ok = llm_manager.remove(name)
    # Prepare updates for dropdowns
    select_update = gr.update(choices=llm_manager.get_names(), value=llm_manager.get_selected())
    chat_update = gr.update(choices=llm_manager.get_names(), value=llm_manager.get_selected())
    if ok:
        return get_llms_table(), "Removed: " + name, select_update, chat_update
    return get_llms_table(), "Model not found", select_update, chat_update


def test_llm_with_values(name, provider, model, base_url, api_key, timeout, max_retries, temperature):
    cfg = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": int(timeout or 30),
        "max_retries": int(max_retries or 2),
        "temperature": float(temperature or 0.1)
    }
    try:
        ok, msg = llm_manager.test_model(cfg, "test")
        return ("Success") if ok else ("Failed: " + msg)
    except Exception as e:
        return f"Error testing LLM: {e}"


def get_embeds_table():
    rows = []
    for name in embed_manager.get_names():
        cfg = embed_manager.get_config(name) or {}
        provider = cfg.get("provider", "ollama")
        model = cfg.get("model", "")
        rows.append([name, provider, model])
    return rows


def run_test_embed(name, text="testing embedding model", provider=None, model=None, api_key=None, url=None, timeout=None):
    ok, msg = test_embed(name, text=text, provider=provider, model=model, api_key=api_key, url=url)
    return ("Success") if ok else ("Failed: " + msg)

def test_embed(name, text="testing embedding model", provider=None, model=None, api_key=None, url=None, timeout=None):
    """Run an embedding test using parameters provided by the UI.

    The UI must pass the connection parameters (provider, model, api_key, url,
    timeout). These are forwarded as `params` to `embed_manager.test_model`.
    """
    params = {}
    if provider is not None:
        params["provider"] = provider
    if model is not None:
        params["model"] = model
    if api_key is not None:
        params["api_key"] = api_key
    if url is not None:
        params["url"] = url
    # Always ensure timeout is a valid int or float, default to 30
    try:
        timeout_val = int(timeout) if timeout not in (None, "", []) else 30
    except Exception:
        timeout_val = 30
    params["timeout"] = timeout_val

    return embed_manager.test_model(name, text, params=params)

def add_embedding(name, provider, model, api_key, url=None):
    if not name:
        return get_embeds_table(), None, "Name is required"
    if not url:
        return get_embeds_table(), None, "URL is required"
    cfg = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "url": url
    }

    _, dim = test_embed(name, text="testing embedding model", provider=provider, model=model, api_key=api_key, url=url)
    dimensions = dim.strip().lstrip("(").rstrip(")").split(",")[0]
    if  int(dimensions) != 768:
        return get_embeds_table(), None, f"Failed: Embedding dimension must be 768. Model {model} has {dimensions} Dimensions."
    
    try:
        embed_manager.register(name, cfg)
        return get_embeds_table(), gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected()), "Embedding added"
    except Exception as e:
        return get_embeds_table(), None, f"Error adding embedding: {e}"


def get_embed_for_form(name):
    cfg = embed_manager.get_config(name) or {}
    provider = cfg.get("provider", "ollama")
    model = cfg.get("model", "")
    api_key = cfg.get("api_key", "")
    url = cfg.get("url", "")
    return name, provider, model, api_key, url


def remove_embedding(name):
    ok = embed_manager.remove(name)
    select_update = gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected())
    upload_update = gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected())
    if ok:
        return get_embeds_table(), "Removed: " + name, select_update, upload_update
    return get_embeds_table(), "Model not found", select_update, upload_update

