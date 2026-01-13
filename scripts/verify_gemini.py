import os
import json
import time
import asyncio
import argparse
import google.generativeai as genai

# This script verifies Gemini API connectivity and a basic analysis flow.
# Usage:
#   GEMINI_API_KEY=... python scripts/verify_gemini.py --model gemini-1.5-flash
# Optional: provide a sample log via --file logs.txt or --text "..."

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--file")
    parser.add_argument("--text")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment.")
        return 2

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    logs = ""
    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            logs = f.read()
    elif args.text:
        logs = args.text
    else:
        logs = """CMD: nmap -sS -Pn -p 1-100 192.168.1.20\nOUT: Starting Nmap 7.80...\nCMD: tail -n 50 /var/log/auth.log\nOUT: Failed password for invalid user admin from 10.0.0.5 port 50324 ssh2\n"""

    system_prompt = (
        "You are a cybersecurity analyst monitoring a live attack simulation lab. "
        "Given terminal commands and output logs, identify potential attacks, provide a concise security analysis, "
        "and give actionable recommendations. Respond in compact JSON with keys: security_analysis, attack_detection[], recommendations[]."
    )

    started = time.time()
    resp = model.generate_content([system_prompt, json.dumps({"input": "Terminal commands and output logs", "logs": logs})])
    elapsed = time.time() - started
    print(f"OK: Request completed in {int(elapsed*1000)} ms")
    print("Raw response text:\n", getattr(resp, "text", "<empty>"))

if __name__ == "__main__":
    asyncio.run(main())
