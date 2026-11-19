"""Copy, Move and Delete for the files selected in the results grid. The files are copied,
moved or sent to the Recycle Bin in a worker thread, so the window keeps working (deleting
thousands of duplicate copies takes a while); one operation runs at a time."""
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox

from vibekimus import file_ops
from vibekimus.embedding_index import IndexSaveError
from vibekimus.ui import theme
from vibekimus.ui.background import when_done

MAX_DELETE_ERRORS_LISTED = 8
POLL_MS = 100
TRASH_MISSING_MESSAGE = ("Cannot delete safely — 'send2trash' is not installed.\n\n"
                         "Files will NOT be deleted. Install it with:\n"
                         "    pip install send2trash\n\n"
                         "This ensures files go to the Recycle Bin and can be recovered.")
BUSY_MESSAGE = "Files are still being copied, moved or deleted. Try again when that is done."
INDEX_NOT_SAVED = "The files are gone from their place, but the index could not be saved:\n\n{error}"


class SelectionActions:
    """Asks the user where / whether, performs the file operation, reports the
    result, and keeps the library and the grid in step with what happened on disk."""

    def __init__(self, root, library, results, set_status, refresh_stats):
        self._root = root
        self._library = library
        self._results = results
        self._set_status = set_status
        self._refresh_stats = refresh_stats
        self._worker = ThreadPoolExecutor(max_workers=1, initializer=file_ops.prepare_worker_thread)
        self._running = None   # Future of the operation in progress

    def _selection(self):
        """The selected paths, or None (after telling the user) when there are none or
        another operation is still running."""
        if self._running is not None:
            messagebox.showinfo("Busy", BUSY_MESSAGE)
            return None
        if not self._results.selected:
            messagebox.showinfo("Info", "No images selected")
            return None
        return list(self._results.selected)

    def _in_background(self, busy_text, work, finish):
        """work() in the worker thread, then finish(its FileOperationReport) on the Tk thread."""
        self._set_status(busy_text, theme.STATUS_BUSY)
        self._running = self._worker.submit(work)
        when_done(self._root, self._running, lambda future: self._finished(future, finish), POLL_MS)

    def _finished(self, future, finish):
        self._running = None
        finish(future.result())   # the operations report failures in the report, they do not raise

    def _forget(self, paths):
        """Files that left their place: drop them from the grid and the index. Files
        that were skipped or failed stay selected for another try."""
        if not paths:
            return
        self._results.remove_files(paths)
        try:
            self._library.forget_files(paths)
        except IndexSaveError as error:
            messagebox.showerror("Index not saved", INDEX_NOT_SAVED.format(error=error))
        self._refresh_stats()

    def copy(self):
        paths = self._selection()
        if not paths:
            return
        destination = filedialog.askdirectory(title="Export to")
        if not destination:
            return

        def copied(report):
            # The selection is kept: the user may want to copy elsewhere or delete next.
            self._set_status(f"Copied {len(report.done)} files", theme.STATUS_OK)
            messagebox.showinfo("Copy Complete", report.summary("copied") or "Nothing to copy.")
        self._in_background(f"Copying {len(paths)} files...", lambda: file_ops.copy_files(paths, destination), copied)

    def move(self):
        paths = self._selection()
        if not paths:
            return
        destination = filedialog.askdirectory(title="Move selected images to...")
        if not destination:
            return

        def moved(report):
            self._forget(report.done)
            self._set_status(f"Moved {len(report.done)} images", theme.STATUS_OK)
            messagebox.showinfo("Move Complete", report.summary("moved") or "Nothing to move.")
        self._in_background(f"Moving {len(paths)} files...", lambda: file_ops.move_files(paths, destination), moved)

    def delete(self):
        paths = self._selection()
        if not paths:
            return
        try:
            send_to_trash = file_ops.load_trash()
        except file_ops.TrashUnavailable:
            messagebox.showerror("Missing Dependency", TRASH_MISSING_MESSAGE)
            return
        if not messagebox.askyesno(
                "Confirm Delete",
                f"Move {len(paths)} selected file(s) to the Recycle Bin?\n"
                f"Files can be restored from the Recycle Bin."):
            return
        self._in_background(f"Moving {len(paths)} files to the Recycle Bin...",
                            lambda: file_ops.trash_files(paths, send_to_trash), self._deleted)

    def _deleted(self, report):
        self._forget(report.done)
        if report.done:
            self._set_status(f"Moved {len(report.done)} files to Recycle Bin", theme.STATUS_OK)
            messagebox.showinfo("Moved to Recycle Bin",
                                f"Successfully moved {len(report.done)} file(s) to the Recycle Bin.")
        if report.errors:
            messagebox.showerror(
                "Delete Errors",
                f"{len(report.errors)} file(s) could not be deleted:\n\n"
                + "\n".join(report.errors[:MAX_DELETE_ERRORS_LISTED]))
