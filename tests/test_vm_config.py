import os
import pytest
import importlib
import asyncio

vmc = importlib.import_module('vm_config')


def test_validate_host_config_rejects_bad_ip():
    bad = vmc.HostConfig(host='999.999.999.999', username='u', password='p', port=22)
    with pytest.raises(vmc.ConfigError):
        vmc.validate_host_config(bad)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    # Use temp data dir
    d = tmp_path
    monkeypatch.setattr(vmc, 'DATA_DIR', str(d))
    monkeypatch.setattr(vmc, 'SECRET_PATH', os.path.join(str(d), 'secret.key'))
    monkeypatch.setattr(vmc, 'CONFIG_PATH', os.path.join(str(d), 'vm_config.enc'))
    # Recreate fernet with fresh secret
    from importlib import reload
    reload(vmc)

    a = vmc.HostConfig(host='127.0.0.1', username='auser', password='apass', port=22)
    t = vmc.HostConfig(host='localhost', username='tuser', password='tpass', port=22)
    cfg = vmc.VMConfig(attacker=a, target=t)
    vmc.save_encrypted_config(cfg)

    loaded_masked = vmc.load_decrypted_config(mask_passwords=True)
    assert loaded_masked is not None
    assert loaded_masked.attacker.password == '***'
    assert loaded_masked.target.password == '***'

    loaded = vmc.load_decrypted_config(mask_passwords=False)
    assert loaded.attacker.password == 'apass'
    assert loaded.target.password == 'tpass'
