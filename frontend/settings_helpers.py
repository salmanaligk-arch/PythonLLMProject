import os
import json
from functools import lru_cache
import gradio as gr
from services.llm_manager import llm_manager
from services.embed_manager import embed_manager
from services.rag_agents import agent_manager
from services.chatbot import set_active_llm

CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", 1))

def _load_settings_from_file(filepath: str):
    """
    Load settings from a JSON file.

    Args:
        filepath: The path to the settings file.

    Returns:
        A tuple containing the settings and a status message.
    """
    if not os.path.exists(filepath):
        return None, None, None, None, None, f"No settings file found at {filepath}"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        llm = payload.get('default_llm') or payload.get('llm_config', {}).get('hf_model') or payload.get('llm_config', {}).get('ollama_model') or payload.get('llm_config', {}).get('model')
        embed = payload.get('default_embedding') or payload.get('file_config', {}).get('embedding_model')
        agent = payload.get('default_agent') or payload.get('agent_config', {}).get('default_agent')
        chunk_size = payload.get('chunk_size') or payload.get('file_config', {}).get('max_chunk_size') or int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = payload.get('chunk_overlap') or payload.get('file_config', {}).get('chunk_overlap') or int(os.getenv('CHUNK_OVERLAP', 200))

        return llm, embed, agent, int(chunk_size), int(chunk_overlap), f"Loaded settings from {os.path.basename(filepath)}"
    except Exception as e:
        return None, None, None, None, None, f"Failed to load settings from {filepath}: {e}"

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

        load_settings.cache_clear()
        llm_update = gr.update(choices=llm_manager.get_names(), value=llm_manager.get_selected())
        embed_update = gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected())
        chat_embed_update = gr.update(choices=embed_manager.get_names(), value=embed_manager.get_selected())
        agent_update = gr.update(choices=agent_manager.get_available_agents(), value=agent_name)
        chunk_size_update = gr.update(value=int(chunk_size or int(os.getenv("CHUNK_SIZE", 1000))))
        chunk_overlap_update = gr.update(value=int(chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))))
        return llm_update, embed_update, agent_update, chat_embed_update, chunk_size_update, chunk_overlap_update, "\n".join(status_msgs)
    except Exception as e:
        return None, None, None, None, None, None, f"Error saving settings: {e}"

@lru_cache(maxsize=CACHE_MAXSIZE)
def load_settings():
    """Read `data/settings.json` and return flat UI values without writing (cached)."""
    settings_path = os.path.join(os.getcwd(), "data", "settings.json")
    return _load_settings_from_file(settings_path)

def load_defaults():
    """Read `data/default_settings.json` and return Gradio update objects."""
    default_path = os.path.join(os.getcwd(), "data", "default_settings.json")
    llm, embed, agent, chunk_size, chunk_overlap, status_msg = _load_settings_from_file(default_path)

    if llm is None and embed is None: # Loading failed
        return None, None, None, None, None, status_msg

    llm_choices = llm_manager.get_names()
    embed_choices = embed_manager.get_names()
    agent_choices = agent_manager.get_available_agents()

    return (
        gr.update(choices=llm_choices, value=llm or llm_manager.get_selected()),
        gr.update(choices=embed_choices, value=embed or embed_manager.get_selected()),
        gr.update(choices=agent_choices, value=agent or os.getenv("DEFAULT_AGENT", "simple")),
        gr.update(value=chunk_size),
        gr.update(value=chunk_overlap),
        status_msg + " Click Save to apply them."
    )

def update_fields_on_load():
    llm, embed, agent, chunk_size, chunk_overlap, status = load_settings()

    llm_choices = llm_manager.get_names()
    embed_choices = embed_manager.get_names()
    agent_choices = agent_manager.get_available_agents()

    return (
        gr.update(choices=llm_choices, value=llm_manager.get_selected()),
        gr.update(choices=embed_choices, value=embed_manager.get_selected()),
        gr.update(choices=agent_choices, value=agent or os.getenv("DEFAULT_AGENT", "simple")),
        gr.update(value=int(chunk_size or int(os.getenv("CHUNK_SIZE", 1000)))),
        gr.update(value=int(chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200)))),
        gr.update(choices=embed_choices, value=embed_manager.get_selected()),
        gr.update(choices=llm_choices, value=llm or llm_manager.get_selected()),
        gr.update(choices=embed_choices, value=embed or embed_manager.get_selected()),
        gr.update(choices=agent_choices, value=agent),
        gr.update(value=int(chunk_size or int(os.getenv("CHUNK_SIZE", 1000)))),
        gr.update(value=int(chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200)))),
        status or "",
        gr.update(value=get_llms_table()),
        gr.update(choices=llm_choices, value=llm_manager.get_selected()),
        gr.update(value=get_embeds_table()),
        gr.update(choices=embed_choices, value=embed_manager.get_selected()),
    )

@lru_cache(maxsize=CACHE_MAXSIZE)
def get_llms_table():
    """Get LLMs table (cached)."""
    rows = []
    for name in llm_manager.get_names():
        cfg = llm_manager.get_config(name) or {}
        provider = cfg.get("provider", "openai")
        model = cfg.get("model", "")
        rows.append([name, provider, model])
    return rows

def add_llm(name, provider, model, base_url, api_key, timeout, max_retries, temperature):
    if not name:
        return get_llms_table(), None, None, None, "Name is required"
    if not base_url:
        return get_llms_table(), None, None, None, "Base URL is required"
    if not api_key:
        return get_llms_table(), None, None, None, "API Key is required"
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
        get_llms_table.cache_clear()
        llm_choices = llm_manager.get_names()
        selected_llm = llm_manager.get_selected()
        return get_llms_table(), gr.update(choices=llm_choices, value=selected_llm), gr.update(choices=llm_choices, value=selected_llm), gr.update(choices=llm_choices, value=selected_llm), "LLM added"
    except Exception as e:
        return get_llms_table(), None, None, None, f"Error adding LLM: {e}"

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
    get_llms_table.cache_clear()
    llm_choices = llm_manager.get_names()
    selected_llm = llm_manager.get_selected()
    select_update = gr.update(choices=llm_choices, value=selected_llm)
    chat_update = gr.update(choices=llm_choices, value=selected_llm)
    general_update = gr.update(choices=llm_choices, value=selected_llm)
    if ok:
        return get_llms_table(), "Removed: " + name, select_update, chat_update, general_update
    return get_llms_table(), "Model not found", select_update, chat_update, general_update

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
        return "✅ Success" if ok else f"❌ Failed: {msg}"
    except Exception as e:
        return f"❌ Error testing LLM: {e}"

@lru_cache(maxsize=CACHE_MAXSIZE)
def get_embeds_table():
    """Get embeds table (cached)."""
    rows = []
    for name in embed_manager.get_names():
        cfg = embed_manager.get_config(name) or {}
        provider = cfg.get("provider", "ollama")
        model = cfg.get("model", "")
        rows.append([name, provider, model])
    return rows

def run_test_embed(name, text="testing embedding model", provider=None, model=None, api_key=None, url=None, timeout=None):
    ok, msg = test_embed(name, text=text, provider=provider, model=model, api_key=api_key, url=url, timeout=timeout)
    return "✅ Success" if ok else f"❌ Failed: {msg}"

def test_embed(name, text="testing embedding model", provider=None, model=None, api_key=None, url=None, timeout=None):
    """Run an embedding test using parameters provided by the UI."""
    params = {}
    if provider is not None:
        params["provider"] = provider
    if model is not None:
        params["model"] = model
    if api_key is not None:
        params["api_key"] = api_key
    if url is not None:
        params["url"] = url
    try:
        timeout_val = int(timeout) if timeout not in (None, "", []) else 30
    except (ValueError, TypeError):
        timeout_val = 30
    params["timeout"] = timeout_val

    return embed_manager.test_model(name, text, params=params)

def add_embedding(name, provider, model, api_key, url=None):
    if not name:
        return get_embeds_table(), None, None, None, "Name is required"
    if not url:
        return get_embeds_table(), None, None, None, "URL is required"
    cfg = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "url": url
    }

    ok, result = test_embed(name, text="testing embedding model", provider=provider, model=model, api_key=api_key, url=url)
    if not ok:
        return get_embeds_table(), None, None, None, f"Test Failed for adding new embedding model: {result}"
    
    try:
        dimensions_str = str(result)
        dimensions = int(''.join(filter(str.isdigit, dimensions_str)))
        if dimensions != 768:
            return get_embeds_table(), None, None, None, f"Failed: Embedding dimension must be 768. Model {model} has {dimensions} Dimensions."
    except (ValueError, IndexError):
         return get_embeds_table(), None, None, None, f"Could not determine embedding dimensions from test result: {result}"

    try:
        embed_manager.register(name, cfg)
        get_embeds_table.cache_clear()
        embed_choices = embed_manager.get_names()
        selected_embed = embed_manager.get_selected()
        return get_embeds_table(), gr.update(choices=embed_choices, value=selected_embed), gr.update(choices=embed_choices, value=selected_embed), gr.update(choices=embed_choices, value=selected_embed), "Embedding added"
    except Exception as e:
        return get_embeds_table(), None, None, None, f"Error adding embedding: {e}"

def get_embed_for_form(name):
    cfg = embed_manager.get_config(name) or {}
    provider = cfg.get("provider", "ollama")
    model = cfg.get("model", "")
    api_key = cfg.get("api_key", "")
    url = cfg.get("url", "")
    return name, provider, model, api_key, url

def remove_embedding(name):
    ok = embed_manager.remove(name)
    get_embeds_table.cache_clear()
    embed_choices = embed_manager.get_names()
    selected_embed = embed_manager.get_selected()
    select_update = gr.update(choices=embed_choices, value=selected_embed)
    upload_update = gr.update(choices=embed_choices, value=selected_embed)
    general_update = gr.update(choices=embed_choices, value=selected_embed)
    if ok:
        return get_embeds_table(), "Removed: " + name, select_update, upload_update, general_update
    return get_embeds_table(), "Model not found", select_update, upload_update, general_update

