"""Files of the local Hugging Face cache: where one is, and removing one that has been
replaced by a smaller copy of its own."""
import os


def cached_file(repo, filenames, cache_dir=None):
    """Path of the first of `filenames` that is in the cache for `repo`, or None."""
    from huggingface_hub import try_to_load_from_cache
    for filename in filenames:
        path = try_to_load_from_cache(repo, filename, cache_dir=cache_dir)
        if isinstance(path, str) and os.path.isfile(path):
            return path
    return None


def remove_cached_file(path):
    """Remove a cached file; where the cache keeps it as a link to a blob, the blob too.
    Returns the bytes freed."""
    target = os.path.realpath(path)
    freed = os.path.getsize(target)
    os.remove(path)
    if os.path.normcase(target) != os.path.normcase(os.path.abspath(path)) and os.path.exists(target):
        os.remove(target)
    return freed
