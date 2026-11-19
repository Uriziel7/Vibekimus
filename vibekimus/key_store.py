"""Where the index key is kept: a small JSON file next to the program.

The key itself is never stored as it is. It is protected either by the Windows
account (DPAPI: no password, but only this account on this computer can unlock it)
or by a password (scrypt + AES-GCM: works anywhere, asked for at every start).
Switching between the two, or changing the password, only re-protects the same key;
the encrypted files stay as they are. An exported key is the same file format,
always protected by a password, to move a library to another computer or keep a
spare copy."""
import base64
import binascii
import json
import os
import secrets
from enum import StrEnum

from vibekimus import dpapi
from vibekimus.encryption import KEY_BYTES, KEY_ID_BYTES, new_key, new_key_id
from vibekimus.files import write_atomically
from vibekimus.units import BYTES_PER_MB

FORMAT_VERSION = 1
SALT_BYTES = 16
NONCE_BYTES = 12
# scrypt cost: about a third of a second and 128 MB per try, so guessing passwords is slow.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 17, 8, 1
SCRYPT_MAX_MEMORY = 256 * BYTES_PER_MB
SCRYPT_BYTES_PER_N_R = 128   # memory scrypt takes: 128 * n * r bytes
ASSOCIATED_DATA = b"Vibekimus index key"


class Protection(StrEnum):
    """What unlocks the key (stored in the key file as these words)."""
    WINDOWS = "windows"
    PASSWORD = "password"


class WrongPasswordError(ValueError):
    """The password does not unlock this key."""


class KeyFileError(ValueError):
    """The key file is missing, damaged or not a Vibekimus key."""


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _unb64(text):
    return base64.b64decode(text.encode("ascii"))


def _password_key(password, salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P):
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    return Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p).derive(password.encode("utf-8"))


def _protected_by_password(key, key_id, password):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = secrets.token_bytes(SALT_BYTES), secrets.token_bytes(NONCE_BYTES)
    wrapped = AESGCM(_password_key(password, salt)).encrypt(nonce, key, ASSOCIATED_DATA + key_id)
    return {"protection": Protection.PASSWORD, "salt": _b64(salt), "nonce": _b64(nonce), "wrapped": _b64(wrapped),
            "scrypt": {"n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P}}


def _unlocked_by_password(record, key_id, password):
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    cost = record["scrypt"]
    if cost["n"] * cost["r"] * SCRYPT_BYTES_PER_N_R > SCRYPT_MAX_MEMORY:
        raise KeyFileError("The key file asks for more memory than allowed.")
    kek = _password_key(password, _unb64(record["salt"]), cost["n"], cost["r"], cost["p"])
    try:
        return AESGCM(kek).decrypt(_unb64(record["nonce"]), _unb64(record["wrapped"]), ASSOCIATED_DATA + key_id)
    except InvalidTag as error:
        raise WrongPasswordError("Wrong password.") from error


def _write_json(path, record):
    write_atomically(path, json.dumps(record, indent=2).encode("utf-8"))


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        key_id = _unb64(record["key_id"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise KeyFileError(f"Not a readable Vibekimus key file: {error}") from error
    if record.get("protection") not in tuple(Protection) or len(key_id) != KEY_ID_BYTES:
        raise KeyFileError("Not a Vibekimus key file.")
    return record, key_id


class KeyStore:
    """The key file at `path`."""

    def __init__(self, path):
        self.path = path

    def exists(self):
        return os.path.exists(self.path)

    def protection(self):
        """A Protection, or None when there is no key yet."""
        return _read_json(self.path)[0]["protection"] if self.exists() else None

    def create(self):
        """A new key, protected by the Windows account. Returns (key, key_id)."""
        key, key_id = new_key(), new_key_id()
        self.protect_with_windows(key, key_id)
        return key, key_id

    def unlock(self, password=None):
        """(key, key_id). Raises WrongPasswordError, dpapi.DpapiError or KeyFileError."""
        return self._unlock_record(*_read_json(self.path), password)

    def protect_with_windows(self, key, key_id):
        _write_json(self.path, {"version": FORMAT_VERSION, "key_id": _b64(key_id), "protection": Protection.WINDOWS,
                                "blob": _b64(dpapi.protect(key))})

    def protect_with_password(self, key, key_id, password):
        _write_json(self.path, {"version": FORMAT_VERSION, "key_id": _b64(key_id),
                                **_protected_by_password(key, key_id, password)})

    @staticmethod
    def export(key, key_id, path, password):
        """A copy of the key in a file protected by `password` (to move a library)."""
        _write_json(path, {"version": FORMAT_VERSION, "key_id": _b64(key_id),
                           **_protected_by_password(key, key_id, password)})

    @classmethod
    def read_export(cls, path, password):
        """(key, key_id) from an exported key file."""
        record, key_id = _read_json(path)
        return cls._unlock_record(record, key_id, password)

    @staticmethod
    def _unlock_record(record, key_id, password):
        if record["protection"] == Protection.PASSWORD and password is None:
            raise WrongPasswordError("A password is needed.")
        try:
            if record["protection"] == Protection.WINDOWS:
                key = dpapi.unprotect(_unb64(record["blob"]))
            else:
                key = _unlocked_by_password(record, key_id, password)
        except (KeyError, TypeError, binascii.Error) as error:
            raise KeyFileError(f"The key file is damaged: {error}") from error
        if len(key) != KEY_BYTES:
            raise KeyFileError("The key file is damaged.")
        return key, key_id
