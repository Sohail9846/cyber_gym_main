# Cybersecurity Attack Simulation Lab

This project provides a modern, responsive home page and a Python FastAPI backend for a cybersecurity training platform featuring:

- Dark theme with blue/red color scheme (Tailwind CDN)
- Live VM terminals over WebSockets
- Button panel for automated attack/defense/util commands
- Real-time AI analysis using Google Gemini

## Prerequisites

- Python 3.10+
- A Google Gemini API key (from Google AI Studio or Vertex AI Generative Language API)

## Setup

1. Create your environment file

   - Copy the template and fill in your values:

     cp .env.example .env

   - IMPORTANT: Ensure .env.example contains NO real secrets before committing. If you see a value for GEMINI_API_KEY in the example, clear it first. Only the .env file should contain the real key and must not be committed.

2. Install dependencies (preferably inside a virtualenv)

   pip install -U fastapi uvicorn[standard] python-dotenv google-generativeai paramiko psutil

3. Run the backend

   uvicorn main:app --host 0.0.0.0 --port 8000

4. Open the UI

   Visit http://localhost:8000/ in your browser.

## Environment Variables

See .env.example for all options. Key values:

- GEMINI_API_KEY: Your Gemini API key (required for AI analysis)
- GEMINI_MODEL: Default: gemini-1.5-flash
- ANALYSIS_ENABLED: true|false (enable/disable real-time analysis)
- ANALYSIS_SAMPLE_SIZE: How many characters from the end of the terminal buffer to send to Gemini
- ANALYSIS_DEBOUNCE_MS: Debounce delay between analysis requests

## Security Notes

- Never commit real secrets to version control. Keep real keys only in .env (gitignored).
- The backend does not log your API key and will disable analysis if GEMINI_API_KEY is missing.
- Predefined commands are safe defaults for local testing. Replace with real tools in your isolated lab as needed.

## Integrations

- WebSocket: /ws/terminal/{vm_id} for both terminal IO and AI analysis events.
  - Messages from server include:
    - { type: "terminal_output", data: "..." }
    - { type: "analysis", vm_id: "vm1|vm2", data: { security_analysis, attack_detection[], recommendations[] } }
  - Messages to server include:
    - { type: "terminal_input", data: "user keystrokes/commands" }

- REST: GET /security-check invokes the existing remote log collection flow and requests a structured report from Gemini.

## Testing

Install test dependencies:

    pip install -U pytest httpx cryptography

Run tests (unit + optional integration):

    pytest

To run Gemini integration tests (requires a real key):

    export GEMINI_API_KEY=<your_key>
    pytest -k gemini

Manual verification of Gemini:

    export GEMINI_API_KEY=<your_key>
    python scripts/verify_gemini.py --model gemini-1.5-flash

End-to-end test (requires VM config and SSH access):

    export GEMINI_API_KEY=<your_key>
    python scripts/e2e_test.py

Debug dashboard:

- Visit /debug to view real-time Gemini call events and counters

Automated health check:

- Run every 5 minutes via cron:

    */5 * * * * /bin/bash -lc 'cd /home/aka/Downloads/cyber-gym-main && bash scripts/health_check.sh >> health.log 2>&1'

## Customizing Attack/Defense Buttons

Edit COMMANDS in static/index.html to change predefined commands. For production, replace the placeholders with real tools (e.g., nmap, hydra) inside your controlled environment.
# CyberGym
