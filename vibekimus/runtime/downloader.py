"""Downloading one file: only from the expected servers, resumed after an interruption,
checked against its size and SHA-256 before it is used. Standard library only."""
import hashlib
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

# The only servers files may come from (also after redirects): Python packages, PyTorch,
# Hugging Face and its download network. Every file is also checked by its SHA-256.
ALLOWED_HOST_SUFFIXES = ("pypi.org", "pythonhosted.org", "pytorch.org", "huggingface.co", "hf.co")
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60
ATTEMPTS = 4                 # a dropped connection is resumed this many times
RETRY_DELAY_SECONDS = 3
PARTIAL_SUFFIX = ".part"
HTTP_PARTIAL_CONTENT = 206
USER_AGENT = "Vibekimus-setup"


class DownloadError(Exception):
    """The file could not be downloaded, or is not the expected file."""


class DownloadCancelled(Exception):
    """The user stopped the download (what is downloaded so far is kept for next time)."""


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str
    size: int
    label: str   # shown while it downloads


class ProgressCounter:
    """Bytes done of a whole job, shared between the downloading thread and the window."""

    def __init__(self, total):
        self.total = total
        self._done = 0
        self._item = ""
        self._lock = threading.Lock()

    def add(self, count):
        with self._lock:
            self._done += count

    def set_item(self, text):
        with self._lock:
            self._item = text

    def snapshot(self):
        """(bytes done, bytes in all, what is being done)."""
        with self._lock:
            return self._done, self.total, self._item


def host_allowed(url, allowed=ALLOWED_HOST_SUFFIXES, schemes=("https",)):
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme in schemes and any(host == suffix or host.endswith("." + suffix) for suffix in allowed)


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed, schemes):
        super().__init__()
        self._allowed, self._schemes = allowed, schemes

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not host_allowed(newurl, self._allowed, self._schemes):
            raise DownloadError(f"refused a redirect to an unexpected server: {urllib.parse.urlsplit(newurl).hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Downloader:
    def __init__(self, allowed_hosts=ALLOWED_HOST_SUFFIXES, schemes=("https",)):
        """schemes: plain http only in tests (a local server)."""
        self._allowed, self._schemes = allowed_hosts, schemes
        self._opener = urllib.request.build_opener(_CheckedRedirects(allowed_hosts, schemes))

    def fetch(self, remote, destination, progress, should_stop):
        """Download `remote` to `destination`. A partial file from an earlier attempt is
        continued. progress: ProgressCounter (gets every byte, also the resumed ones)."""
        if not host_allowed(remote.url, self._allowed, self._schemes):
            raise DownloadError(f"refused to download from an unexpected server: {remote.url}")
        partial = destination + PARTIAL_SUFFIX
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        counted = 0   # bytes of this file reported to progress: always what the partial file holds
        for attempt in range(ATTEMPTS):
            done, digest = self._hash_existing(partial, remote.size)   # every attempt starts from the disk
            progress.add(done - counted)
            counted = done
            try:
                done, digest = self._continue(remote, partial, done, digest, progress, should_stop)
                break
            except (urllib.error.URLError, OSError) as error:
                counted = os.path.getsize(partial) if os.path.exists(partial) else 0
                if attempt == ATTEMPTS - 1:
                    raise DownloadError(f"{remote.label}: {error}") from error
                time.sleep(RETRY_DELAY_SECONDS)
        if done != remote.size or digest.hexdigest() != remote.sha256.lower():
            os.remove(partial)
            raise DownloadError(f"{remote.label}: the downloaded file is not the expected one (size or SHA-256)")
        os.replace(partial, destination)

    @staticmethod
    def _hash_existing(partial, size):
        """(bytes already there, their hash) of a partial file from an earlier attempt."""
        digest = hashlib.sha256()
        if not os.path.exists(partial):
            return 0, digest
        if os.path.getsize(partial) > size:
            os.remove(partial)
            return 0, digest
        with open(partial, "rb") as f:
            for block in iter(lambda: f.read(CHUNK_BYTES), b""):
                digest.update(block)
        return os.path.getsize(partial), digest

    def _continue(self, remote, partial, done, digest, progress, should_stop):
        if done >= remote.size:
            return done, digest
        request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
        if done:
            request.add_header("Range", f"bytes={done}-")
        with self._opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            if done and response.status != HTTP_PARTIAL_CONTENT:   # the server sends it all again
                progress.add(-done)
                done, digest = 0, hashlib.sha256()
            with open(partial, "ab" if done else "wb") as out:
                for block in iter(lambda: response.read(CHUNK_BYTES), b""):
                    if should_stop():
                        raise DownloadCancelled()
                    out.write(block)
                    digest.update(block)
                    done += len(block)
                    progress.add(len(block))
        if done < remote.size:   # http.client ends a cut-off body quietly: resume it
            raise ConnectionError(f"the connection ended after {done} of {remote.size} bytes")
        return done, digest
