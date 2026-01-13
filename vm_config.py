import os
import json
import re
import ipaddress
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SECRET_PATH = os.path.join(DATA_DIR, "secret.key")
CONFIG_PATH = os.path.join(DATA_DIR, "vm_config.enc")

os.makedirs(DATA_DIR, exist_ok=True)

# Helpers for secret management

def _load_or_create_secret() -> bytes:
    # Prefer env var if provided
    env_key = os.getenv("VM_CONFIG_SECRET", "").strip()
    if env_key:
        try:
            # Accept raw Fernet key or base64-like string
            Fernet(env_key)
            return env_key.encode()
        except Exception:
            pass  # fall back to file
    # Use local secret file (not committed to VCS)
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, "rb") as f:
            return f.read().strip()
    key = Fernet.generate_key()
    with open(SECRET_PATH, "wb") as f:
        f.write(key)
    try:
        os.chmod(SECRET_PATH, 0o600)
    except Exception:
        pass
    return key

FERNET = Fernet(_load_or_create_secret())

@dataclass
class HostConfig:
    host: str
    hostname: Optional[str] = None
    username: str = ""
    password: str = ""
    port: int = 22

@dataclass
class VMConfig:
    attacker: HostConfig
    target: HostConfig

IP_REGEX = re.compile(r"^(?:(?:\d{1,3}\.){3}\d{1,3}|[a-zA-Z0-9_.-]+)$")

class ConfigError(Exception):
    pass

# Validation

def validate_host_config(h: HostConfig) -> None:
    if not h.host:
        raise ConfigError("Host/IP required")
    # If looks like a numeric IPv4, validate strictly
    parts = h.host.split('.')
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        try:
            ipaddress.ip_address(h.host)
        except ValueError:
            raise ConfigError("Invalid IPv4 address")
    else:
        # Otherwise, allow DNS hostnames with basic pattern
        if not IP_REGEX.match(h.host):
            raise ConfigError("Invalid host/hostname format")
    # Port range
    try:
        port_val = int(h.port)
    except Exception:
        raise ConfigError("Invalid port number")
    if not (1 <= port_val <= 65535):
        raise ConfigError("Invalid port number")
    # Username
    if not h.username:
        raise ConfigError("Username required")

# Encryption helpers

def _encrypt(data: bytes) -> bytes:
    return FERNET.encrypt(data)

def _decrypt(data: bytes) -> bytes:
    return FERNET.decrypt(data)

# Storage

def save_encrypted_config(cfg: VMConfig) -> None:
    payload: Dict[str, Any] = {
        "attacker": asdict(cfg.attacker),
        "target": asdict(cfg.target),
    }
    # Encrypt passwords inside payload (optional: double encryption)
    for key in ("attacker", "target"):
        pw = payload[key]["password"].encode()
        payload[key]["password"] = _encrypt(pw).decode()
    data = json.dumps(payload).encode()
    blob = _encrypt(data)
    with open(CONFIG_PATH, "wb") as f:
        f.write(blob)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass


def load_decrypted_config(mask_passwords: bool = True) -> Optional[VMConfig]:
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "rb") as f:
        blob = f.read()
    data = json.loads(_decrypt(blob).decode())
    # Decrypt nested passwords
    for key in ("attacker", "target"):
        try:
            enc = data[key]["password"].encode()
            data[key]["password"] = _decrypt(enc).decode()
        except Exception:
            data[key]["password"] = ""
    attacker = HostConfig(**data["attacker"]) if data.get("attacker") else None
    target = HostConfig(**data["target"]) if data.get("target") else None
    if mask_passwords:
        attacker.password = "***"
        target.password = "***"
    return VMConfig(attacker=attacker, target=target)

# Connection test

def test_ssh_connection(host_cfg: HostConfig, timeout: int = 5) -> Dict[str, Any]:
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host_cfg.host,
            port=host_cfg.port,
            username=host_cfg.username,
            password=host_cfg.password or None,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            look_for_keys=False,
            allow_agent=False,
        )
        stdin, stdout, stderr = client.exec_command("whoami")
        who = stdout.read().decode().strip()
        client.close()
        return {"ok": True, "whoami": who}
    except Exception as e:
        return {"ok": False, "error": str(e)}
