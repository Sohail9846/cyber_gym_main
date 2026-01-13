import os
import json
import pytest
import importlib
import asyncio

main = importlib.import_module('main')

SAMPLES = [
    ("normal", "CMD: ls -la\nOUT: total\n", ["no", "benign", "normal"]),
    ("nmap", "CMD: nmap -sS -Pn -p 1-100 192.168.1.1\nOUT: scan result...\n", ["recon", "nmap", "scan"]),
    ("brute", "OUT: Failed password for root from 10.0.0.5 port 22 ssh2\n", ["brute", "password", "ssh"]),
    ("sql", "OUT: GET /login?user=1' OR '1'='1 -- HTTP/1.1\n", ["sql", "injection", "web"]),
]

@pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason='GEMINI_API_KEY not set')
@pytest.mark.parametrize("name,logs,keywords", SAMPLES)
def test_log_patterns_integration(name, logs, keywords):
    out = asyncio.get_event_loop().run_until_complete(main.send_logs_to_gemini(logs))
    assert isinstance(out, dict)
    assert 'status' in out
    assert 'security_analysis' in out
    # non-strict: at least one keyword should appear in analysis or detection
    text = (out.get('security_analysis') or '').lower() + ' ' + ' '.join([str(x).lower() for x in out.get('attack_detection', [])])
    assert any(k in text for k in keywords)
