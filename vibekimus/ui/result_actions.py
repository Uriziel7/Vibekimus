"""What can be done with a single result: play or open it, copy it to the clipboard."""
from vibekimus.clipboard import ClipboardUnavailable, copy_files, copy_picture
from vibekimus.imaging import open_image
from vibekimus.media_kind import MediaKind
from vibekimus.shell import open_with_default_app
from vibekimus.ui import theme
from vibekimus.ui.formatting import format_moment
from vibekimus.video_io import read_frame_at
from vibekimus.video_player import open_video_at


class ResultActions:
    def __init__(self, set_status):
        self._set_status = set_status

    def open(self, result):
        """Videos play from the matching moment; images open in the default viewer."""
        if result.kind != MediaKind.VIDEO:
            open_with_default_app(result.path)
            return
        player = open_video_at(result.path, result.timestamp)
        if player:
            self._set_status(f"Playing in {player} from {format_moment(result.timestamp)}", theme.STATUS_OK)
        else:
            self._set_status("Opened from the start — install VLC or PotPlayer to jump to the moment",
                             theme.STATUS_BUSY)

    def copy_files(self, paths):
        self._copy(lambda: copy_files(paths),
                   f"Copied {len(paths)} files — paste with Ctrl+V" if len(paths) > 1
                   else "Copied the file — paste with Ctrl+V")

    def copy_picture(self, result):
        def copy():
            if result.kind == MediaKind.VIDEO:
                picture = read_frame_at(result.path, result.timestamp)
            else:
                picture = open_image(result.path)
            if picture is None:
                raise ClipboardUnavailable("the picture could not be read")
            copy_picture(picture)
        self._copy(copy, "Copied the picture — paste with Ctrl+V")

    def _copy(self, action, done_text):
        try:
            action()
        except (ClipboardUnavailable, OSError, MemoryError) as error:
            self._set_status(f"Could not copy: {error}", theme.STATUS_ERROR)
            return
        self._set_status(done_text, theme.STATUS_OK)
