"""Encrypting the files the program writes about the user's media: indexes (paths,
embeddings, file details), the text read from pictures and the exclusions.

AES-256-GCM: a stolen file cannot be read, and a changed byte is detected (the file
is refused, not half-read). A sealed file starts with MAGIC and the id of the key it
needs, so a file sealed with another key gets a clear message.

The key itself comes from key_store (protected by the Windows account or by a
password). The program uses one vault at a time (use_vault / active_vault); while
encryption is off the vault only reads: files are written as they are, and sealed
files can still be opened as long as the key is there."""
import io
import secrets

from vibekimus.files import write_atomically

MAGIC = b"VIBEKIMUS-SEALED-1"
KEY_BYTES = 32           # AES-256
KEY_ID_BYTES = 8
NONCE_BYTES = 12         # the size AES-GCM is made for
HEADER_BYTES = len(MAGIC) + KEY_ID_BYTES + NONCE_BYTES


class LockedError(RuntimeError):
    """A sealed file, but the key is not available (locked, other computer, other key)."""


class DamagedError(RuntimeError):
    """A sealed file that was changed or cut short: it is refused."""


def new_key():
    return secrets.token_bytes(KEY_BYTES)


def new_key_id():
    return secrets.token_bytes(KEY_ID_BYTES)


def is_sealed_bytes(data):
    return data[:len(MAGIC)] == MAGIC


class Vault:
    """seal() encrypts what is written while encryption is on; unseal() opens sealed
    files with the key, and passes plain (older) files through."""

    def __init__(self, key=None, key_id=None, enabled=False):
        self._key, self.key_id = key, key_id
        self.enabled = enabled

    @property
    def has_key(self):
        return self._key is not None

    def seal(self, data):
        """Encrypted while encryption is on. With encryption on but the key locked,
        nothing is written in plain text: LockedError."""
        if not self.enabled:
            return data
        if self._key is None:
            raise LockedError("Index encryption is on but the key is locked; nothing was saved.")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(NONCE_BYTES)
        header = MAGIC + self.key_id + nonce
        return header + AESGCM(self._key).encrypt(nonce, data, header)

    def unseal(self, data):
        if not is_sealed_bytes(data):
            return data
        key_id = data[len(MAGIC):len(MAGIC) + KEY_ID_BYTES]
        if self._key is None:
            raise LockedError("This file is encrypted and the encryption key is locked or missing.")
        if key_id != self.key_id:
            raise LockedError("This file is encrypted with a key this program does not have "
                              "(made on another computer, or before the key was replaced). "
                              "Import that key, or index again.")
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        header = data[:HEADER_BYTES]
        nonce = header[-NONCE_BYTES:]
        try:
            return AESGCM(self._key).decrypt(nonce, data[HEADER_BYTES:], header)
        except InvalidTag as error:
            raise DamagedError("This encrypted file was changed or is damaged, so it is not used.") from error


_active = Vault()


def use_vault(vault):
    global _active
    _active = vault


def active_vault():
    return _active


# ── files ────────────────────────────────────────────────────────────────────
def read_file(path):
    """The contents of a file the program wrote, decrypted if it is sealed."""
    with open(path, "rb") as f:
        return _active.unseal(f.read())


def open_read(path):
    """read_file as a file object (for numpy.load)."""
    return io.BytesIO(read_file(path))


def write_file_atomically(path, data):
    """files.write_atomically, sealed while encryption is on."""
    write_atomically(path, _active.seal(data))


def is_sealed(path):
    """Whether a file on disk is encrypted."""
    try:
        with open(path, "rb") as f:
            return is_sealed_bytes(f.read(len(MAGIC)))
    except OSError:
        return False
