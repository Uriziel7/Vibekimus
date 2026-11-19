"""Turning index encryption on and off, and the key's protection (Windows account or
password). Keeps the program's vault (encryption.use_vault) in step with the choice.

Without the GUI: the window asks for passwords and calls these methods."""
from vibekimus.encryption import LockedError, Vault, use_vault
from vibekimus.key_store import KeyStore, Protection


class IndexProtection:
    def __init__(self, key_file):
        self._store = KeyStore(key_file)
        self._key = self._key_id = None
        self.enabled = False

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def protection(self):
        """A key_store.Protection, or None when no key was made yet."""
        return self._store.protection()

    @property
    def is_unlocked(self):
        return self._key is not None

    @property
    def needs_password(self):
        """A key protected by a password that has not been given yet."""
        return not self.is_unlocked and self.protection == Protection.PASSWORD

    def _apply(self):
        use_vault(Vault(self._key, self._key_id, self.enabled))

    # ── start ────────────────────────────────────────────────────────────────
    def start(self, enabled):
        """At program start: unlock a Windows-protected key at once (a password-
        protected one waits for unlock()). Raises when the key cannot be unlocked."""
        self.enabled = enabled
        if self._store.exists() and self.protection == Protection.WINDOWS:
            self._key, self._key_id = self._store.unlock()
        self._apply()

    def unlock(self, password):
        """Raises key_store.WrongPasswordError."""
        self._key, self._key_id = self._store.unlock(password)
        self._apply()

    # ── choices ──────────────────────────────────────────────────────────────
    def enable(self):
        """Encrypt from now on; the first time a key is made (protected by Windows).
        A locked key is never replaced: that would orphan every file it encrypted."""
        if self.needs_password:
            raise LockedError("Unlock the key with its password first.")
        if not self.is_unlocked:
            self._key, self._key_id = self._store.create()
        self.enabled = True
        self._apply()

    def disable(self):
        """Write files unencrypted from now on (encrypted ones still open)."""
        self.enabled = False
        self._apply()

    def protect_with_windows(self):
        self._store.protect_with_windows(self._key, self._key_id)

    def protect_with_password(self, password):
        """Also used to change the password."""
        self._store.protect_with_password(self._key, self._key_id, password)

    def export_key(self, path, password):
        KeyStore.export(self._key, self._key_id, path, password)

    def import_key(self, path, password):
        """Use a key exported earlier (e.g. on another computer), protected by Windows
        here. Files encrypted with the replaced key no longer open."""
        self._key, self._key_id = KeyStore.read_export(path, password)
        self._store.protect_with_windows(self._key, self._key_id)
        self._apply()
