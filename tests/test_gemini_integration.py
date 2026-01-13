import os
import json
import pytest
from fastapi.testclient import TestClient

# These tests cover parse/format logic and health endpoints without real API calls.
# Integration tests that hit Gemini are skipped unless GEMINI_API_KEY is set.

import importlib
main = importlib.import_module('main')
client = TestClient(main.app)


def test_health_endpoint_basic():
    r = client.get('/api/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert 'gemini' in data


def test_parse_and_format_response_handles_json():
    text = json.dumps({
        "security_analysis": "No threat detected",
        "attack_detection": [],
        "recommendations": []
    })
    out = main.parse_and_format_response(text)
    assert out['status'] == 'ok'
    assert out['security_analysis'] == 'No threat detected'
    assert isinstance(out['attack_detection'], list)
    assert isinstance(out['recommendations'], list)


def test_parse_and_format_response_handles_text():
    text = "Likely reconnaissance based on nmap scan"
    out = main.parse_and_format_response(text)
    assert out['status'] == 'ok'
    assert 'Likely' in out['security_analysis']


@pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason='GEMINI_API_KEY not set')
def test_send_logs_to_gemini_integration():
    import asyncio
    # Use a small sample; we only assert response structure, not exact content
    logs = "CMD: nmap -sS -Pn -p 1-100 10.0.0.1\nOUT: scan results...\n"
    out = asyncio.get_event_loop().run_until_complete(main.send_logs_to_gemini(logs))
    assert isinstance(out, dict)
    assert 'status' in out
    assert 'security_analysis' in out
    assert 'recommendations' in out
