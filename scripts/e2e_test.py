#!/usr/bin/env python3
"""
End-to-end test script:
- Loads VM config (if present)
- Tests SSH connections to attacker and target
- Executes whoami on both
- Runs a simple attack command (nmap) from attacker against target
- Collects tail of auth.log from target
- Sends combined logs to Gemini for analysis
- Prints a test report

Usage:
  GEMINI_API_KEY=... python scripts/e2e_test.py
"""
import os
import time
import json
import tempfile
from dataclasses import asdict

import google.generativeai as genai
import paramiko

from vm_config import load_decrypted_config, test_ssh_connection, HostConfig

MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def ssh_exec(cfg: HostConfig, cmd: str, timeout: int = 10):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=cfg.host, port=cfg.port, username=cfg.username, password=cfg.password or None,
                timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
                look_for_keys=False, allow_agent=False)
    stdin, stdout, stderr = cli.exec_command(cmd)
    out = stdout.read().decode(errors='ignore')
    err = stderr.read().decode(errors='ignore')
    cli.close()
    return out, err


def main():
    report = {"steps": []}

    cfg = load_decrypted_config(mask_passwords=False)
    if not cfg:
        print("[FAIL] No VM config found. Use /config to set it up.")
        return 1

    # 1. Connection tests
    for name in ("attacker", "target"):
        hc = getattr(cfg, name)
        res = test_ssh_connection(hc)
        report["steps"].append({"name": f"ssh:{name}", "ok": res.get("ok"), "details": res})
        if not res.get("ok"):
            print(json.dumps(report, indent=2))
            return 2

    # 2. Execute simple commands
    a_out, _ = ssh_exec(cfg.attacker, "whoami")
    t_out, _ = ssh_exec(cfg.target, "whoami")
    report["steps"].append({"name": "cmd:whoami", "ok": True, "details": {"attacker": a_out.strip(), "target": t_out.strip()}})

    # 3. Run a basic nmap scan from attacker to target (requires nmap installed)
    try:
        scan_cmd = f"nmap -sS -Pn -p 1-100 {cfg.target.host} | head -n 20"
        scan_out, _ = ssh_exec(cfg.attacker, scan_cmd)
        report["steps"].append({"name": "attack:nmap", "ok": True})
    except Exception as e:
        scan_out = str(e)
        report["steps"].append({"name": "attack:nmap", "ok": False, "error": str(e)})

    # 4. Collect logs from target
    try:
        logs_cmd = "tail -n 200 /var/log/auth.log"
        logs_out, _ = ssh_exec(cfg.target, logs_cmd)
        report["steps"].append({"name": "logs:collect", "ok": True})
    except Exception as e:
        logs_out = str(e)
        report["steps"].append({"name": "logs:collect", "ok": False, "error": str(e)})

    # 5. Send to Gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        report["steps"].append({"name": "gemini:key", "ok": False, "error": "GEMINI_API_KEY not set"})
        print(json.dumps(report, indent=2))
        return 3

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    combined = f"CMD: {scan_cmd}\nOUT: {scan_out}\nOUT: {logs_out}\n"
    prompt = (
        "You are a cybersecurity analyst monitoring a live attack simulation lab. "
        "Given terminal commands and output logs, identify potential attacks, provide a concise security analysis, "
        "and give actionable recommendations. Respond in compact JSON with keys: security_analysis, attack_detection[], recommendations[]."
    )
    resp = model.generate_content([prompt, json.dumps({"input": "Terminal commands and output logs", "logs": combined})])
    txt = getattr(resp, "text", "")
    report["steps"].append({"name": "gemini:analyze", "ok": bool(txt), "response_text": txt[:2000]})

    # 6. Print final report
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
