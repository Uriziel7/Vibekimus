"""Hard disks are read one file at a time, SSDs in parallel; moving a deleted file to the Recycle Bin."""
import threading
import time

import pytest

from vibekimus import settings
from vibekimus.file_ops import trash_files
from vibekimus.storage import Drives, StorageMode

KNOWN_DRIVES = {"C:": False, "D:": True}   # drive -> has a seek penalty (spins)
READERS = 6
TRASH_BATCH = 100


@pytest.fixture
def drives():
    """Drives whose kind comes from KNOWN_DRIVES (any other: Windows cannot tell); remembers what was asked."""
    asked = []
    return Drives(ask=lambda drive: asked.append(drive) or KNOWN_DRIVES.get(drive)), asked


def test_each_drive_is_asked_once_and_one_windows_cannot_tell_about_counts_as_an_ssd(drives, monkeypatch):
    found, asked = drives
    monkeypatch.setattr(settings, "STORAGE_MODE", StorageMode.AUTO)
    kinds = (found.spins("c:/photos/a.jpg"), found.spins("D:/old/b.jpg"), found.spins("D:/old/c.jpg"),
             found.spins("Z:/unknown.jpg"))
    assert kinds == (False, True, True, False) and sorted(asked) == ["C:", "D:", "Z:"]


def test_a_hard_disk_is_read_one_file_at_a_time_and_an_ssd_does_not_wait(drives, monkeypatch):
    found, _asked = drives
    monkeypatch.setattr(settings, "STORAGE_MODE", StorageMode.AUTO)
    overlaps, running = [], [0]

    def read_from(path):
        with found.reading(path):
            running[0] += 1
            overlaps.append(running[0])
            time.sleep(0.01)
            running[0] -= 1
    readers = [threading.Thread(target=read_from, args=(f"D:/file{n}.jpg",)) for n in range(READERS)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    assert found.reading("D:/x.jpg") is found.reading("D:/y.mp4"), "pictures and videos share its turn"
    assert max(overlaps) == 1
    assert found.reading("C:/a.jpg") is not found.reading("C:/b.jpg")


def test_set_by_hand_every_drive_is_read_as_chosen(drives, monkeypatch):
    found, _asked = drives
    monkeypatch.setattr(settings, "STORAGE_MODE", StorageMode.HDD)
    assert found.spins("C:/a.jpg")
    monkeypatch.setattr(settings, "STORAGE_MODE", StorageMode.SSD)
    assert not found.spins("D:/a.jpg")


def test_the_recycle_bin_gets_files_100_at_a_time_and_a_failing_batch_file_by_file(tmp_path):
    calls = []

    def fake_trash(paths):   # never the real Recycle Bin
        calls.append(paths)
        if any("locked" in path for path in ([paths] if isinstance(paths, str) else paths)):
            raise OSError("in use")
    files = [tmp_path / f"file{i}.jpg" for i in range(250)] + [tmp_path / "locked.jpg"]
    for path in files:
        path.write_bytes(b"")
    trashed = trash_files([str(path) for path in files], fake_trash)
    assert [len(call) for call in calls[:3]] == [TRASH_BATCH, TRASH_BATCH, 51]
    assert len(trashed.done) == 250 and len(trashed.errors) == 1 and "locked.jpg" in trashed.errors[0]
