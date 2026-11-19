"""Values worked out from large objects — an index's embeddings, a loaded model — kept for as
long as they are asked for with the very same objects. A replaced index array or a newly
loaded model is another object, so its values are worked out again."""
import weakref


def _reference(source):
    """A weak reference where the object allows one (arrays, models), so the cache never keeps a
    replaced index alive; plain lists and tuples are kept with the value."""
    try:
        return weakref.ref(source)
    except TypeError:
        return lambda: source


class IdentityCache:
    """One value per name, valid while its source objects are the same (compared by identity)."""

    def __init__(self):
        self._entries = {}   # name -> (references to the sources, value)

    def get(self, name, *sources):
        """The value kept for `name` if it was made from these very objects, otherwise None."""
        entry = self._entries.get(name)
        if entry is None:
            return None
        references, value = entry
        if len(references) == len(sources) and all(ref() is source for ref, source in zip(references, sources)):
            return value
        return None

    def put(self, name, value, *sources):
        """Keep `value`, made from `sources`, under `name`; returns it."""
        self._entries[name] = (tuple(_reference(source) for source in sources), value)
        return value

    def drop(self, name):
        self._entries.pop(name, None)

    def clear(self):
        self._entries.clear()
