"""Index encryption: the key protected by Windows (DPAPI) or a password, what is encrypted,
locked and lost keys, exporting the key, turning it off, and the .exe's console file."""
import base64
import json
import os

import pytest

from support import open_library, raised
from vibekimus import encryption, settings
from vibekimus.console_file import PRIVATE_NOTICE, PrivateConsoleFile
from vibekimus.index_protection import IndexProtection
from vibekimus.key_store import KeyStore, Protection, WrongPasswordError
from vibekimus.last_index import LastIndex
from vibekimus.scope import FolderScope, MultiFolderScope

PASSWORD, EXPORT_PASSWORD = "correct horse", "for the other pc"
ZIP_START = b"PK"   # an index file once decrypted (.npz)


@pytest.fixture
def key_path(tmp_path):
    (tmp_path / "vault").mkdir()
    return str(tmp_path / "vault" / "key.json")


@pytest.fixture
def protection(key_path):
    """Encryption with a new key, off; afterwards no key is active any more."""
    protection = IndexProtection(key_path)
    protection.start(enabled=False)
    yield protection
    encryption.use_vault(encryption.Vault())


@pytest.fixture
def with_text(details):
    """details with text read on two pictures and an exclusion: every kind of file a library has."""
    details.texts.set("red_small.png", "Darmowe piwo")
    details.texts.set("trip/beach.jpg", "Zażółć")
    details.texts.save()
    details.exclusions.add("tmp")
    return details


@pytest.fixture
def encrypted(with_text, protection):
    protection.enable()
    with_text.rewrite_files()
    return with_text


def test_with_encryption_off_files_are_written_as_before(with_text, protection):
    assert not encryption.is_sealed(with_text.texts.path) and protection.protection is None


def test_turning_it_on_encrypts_the_index_the_text_and_the_exclusions(encrypted, protection):
    sealed = [encryption.is_sealed(path) for path in encrypted._saved_files()]
    raw = open(encrypted.images.path, "rb").read()
    assert all(sealed) and len(sealed) >= 3
    assert b"red_small" not in raw and b"Darmowe" not in raw, "no file name is left readable"
    assert protection.protection == Protection.WINDOWS


def test_the_name_and_folders_of_an_index_of_several_folders_are_encrypted_too(details, protection, tmp_path):
    named = open_library(MultiFolderScope("Named", [details.scope.data_dir], str(tmp_path / "named index")))
    named.scope.prepare()
    protection.enable()
    named.rewrite_files()
    named_file = os.path.join(named.scope.data_dir, settings.LIBRARY_FILE)
    assert encryption.is_sealed(named_file)
    assert named.scope.name == MultiFolderScope.load(named.scope.data_dir).name
    protection.disable()
    assert named.needs_rewrite()
    named.rewrite_files()
    assert not encryption.is_sealed(named_file) and not named.needs_rewrite(), "turned off: readable again"


def test_with_the_key_encrypted_files_load_as_before(encrypted):
    reopened = open_library(encrypted.scope.data_dir)
    reopened.load_images()
    reopened.load_texts()
    assert sorted(reopened.images.keys) == sorted(encrypted.images.keys) and reopened.texts.with_text() == 2
    assert not reopened.needs_rewrite(), "a refresh finds nothing to redo"


def test_a_changed_byte_is_detected_and_the_file_refused(encrypted, tmp_path):
    tampered = bytearray(open(encrypted.images.path, "rb").read())
    tampered[len(tampered) // 2] ^= 1
    path = tmp_path / "tampered.npz"
    path.write_bytes(bytes(tampered))
    assert isinstance(raised(lambda: encryption.read_file(str(path))), encryption.DamagedError)


def test_the_index_opened_last_is_remembered_encrypted(encrypted):
    LastIndex.remember(FolderScope(encrypted.scope.data_dir))
    assert encryption.is_sealed(settings.LAST_INDEX_FILE) and LastIndex.scope() is not None


def test_without_the_key_nothing_encrypted_opens_and_the_program_starts_with_no_index(encrypted):
    LastIndex.remember(FolderScope(encrypted.scope.data_dir))
    encryption.use_vault(encryption.Vault())   # as on another computer: no key at all
    assert isinstance(raised(lambda: open_library(encrypted.scope.data_dir).load_images()), encryption.LockedError)
    assert LastIndex.scope() is None


@pytest.fixture
def locked(encrypted, protection, key_path):
    """The key protected by a password, as at the next start: locked."""
    protection.protect_with_password(PASSWORD)
    locked = IndexProtection(key_path)
    locked.start(enabled=True)
    return locked


def test_with_a_password_the_key_stays_locked_until_it_is_given(encrypted, locked, tmp_path):
    assert locked.needs_password and locked.protection == Protection.PASSWORD
    secret = str(tmp_path / "x")
    assert isinstance(raised(lambda: encryption.write_file_atomically(secret, b"secret")), encryption.LockedError)
    assert not os.path.exists(secret), "never written in the clear"
    assert isinstance(raised(locked.enable), encryption.LockedError), "a locked key is never replaced by a new one"
    assert locked.protection == Protection.PASSWORD
    assert isinstance(raised(lambda: locked.unlock("wrong")), WrongPasswordError)
    locked.unlock(PASSWORD)
    assert locked.is_unlocked
    assert encryption.read_file(encrypted.texts.path)[:2] == ZIP_START, "the same key: files encrypted before open"


def test_an_exported_key_imported_elsewhere_opens_the_same_files(encrypted, locked, key_path, tmp_path):
    locked.unlock(PASSWORD)
    exported = str(tmp_path / "exported key.json")
    locked.export_key(exported, EXPORT_PASSWORD)
    locked.protect_with_windows()
    other_pc = IndexProtection(str(tmp_path / "other key.json"))
    other_pc.start(enabled=True)
    other_pc.import_key(exported, EXPORT_PASSWORD)
    assert encryption.read_file(encrypted.images.path)[:2] == ZIP_START and other_pc.protection == Protection.WINDOWS
    assert isinstance(raised(lambda: KeyStore.read_export(exported, "nope")), WrongPasswordError)
    key = locked._key
    for path in (key_path, exported):
        text = open(path, "rb").read()
        written = [form for form in (key, key.hex().encode(), base64.b64encode(key)) if form in text]
        assert not written, "neither the key file nor the export holds the key itself"
    assert json.loads(open(key_path, "rb").read())["protection"] == Protection.WINDOWS


def test_turning_it_off_writes_the_files_readable_again(encrypted, protection):
    protection.disable()
    encrypted.rewrite_files()
    assert not any(encryption.is_sealed(path) for path in encrypted._saved_files())


def test_with_encryption_on_the_exe_console_file_keeps_no_file_names_or_searches(tmp_path):
    console_path, previous_path = tmp_path / "console.txt", tmp_path / "console.previous.txt"
    previous_path.write_text("")
    private = [False]
    with open(console_path, "w", encoding="utf-8") as stream:
        console = PrivateConsoleFile(stream, str(previous_path), lambda: private[0])
        console.write("[SEARCH] Starting search for: 'holiday'\n")
        private[0] = True
        console.write("[VINDEX] Analyzing: secret.mp4\n")
        console.flush()
    assert console_path.read_text(encoding="utf-8") == PRIVATE_NOTICE and not previous_path.exists()
