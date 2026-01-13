import os
import importlib

def test_send_logs_disabled_without_key(monkeypatch):
    # Ensure no API key
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    main = importlib.import_module('main')
    import asyncio
    out = asyncio.get_event_loop().run_until_complete(main.send_logs_to_gemini("test logs"))
    assert out.get('status') == 'disabled'
    assert 'gemini_api_key_missing' in out.get('reason', '')
