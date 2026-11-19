"""Index info: where the open index is, its files, what it holds and what is excluded."""
import os
from tkinter import messagebox

from vibekimus.ui.formatting import format_date, format_size

NOT_AVAILABLE = "N/A"
NONE = "None"
NO_FOLDER = "No folder selected"
NO_INDEX = "No index loaded"


def describe_file(path):
    """(size, last modified) of a file as text, NOT_AVAILABLE for what cannot be read."""
    try:
        return format_size(os.path.getsize(path)), format_date(os.path.getmtime(path), with_time=True)
    except OSError:
        return NOT_AVAILABLE, NOT_AVAILABLE


def show_index_info(library):
    counts = library.counts()
    if library.is_open:
        location = library.scope.description
        image_file, video_file = library.images.path, library.videos.path
        excluded_folders = [library.scope.absolute_path(folder) for folder in library.exclusions.folders()]
        exclusions = ", ".join(library.exclusions.sorted() + excluded_folders) or NONE
    else:
        location = NO_FOLDER
        image_file = video_file = NO_INDEX
        exclusions = NONE
    image_size, image_modified = describe_file(image_file)
    video_size, _video_modified = describe_file(video_file)
    messagebox.showinfo("Index Info", (
        f"Folder:\n  {location}\n\n"
        f"Image Cache:\n  {image_file}\n"
        f"Cache Size:  {image_size}\n"
        f"Cache Modified:  {image_modified}\n\n"
        f"Images Indexed:  {counts.images:,}\n\n"
        f"Video Cache:\n  {video_file}\n"
        f"Video Cache Size:  {video_size}\n\n"
        f"Videos Indexed:  {counts.videos:,} ({counts.frames:,} frames)\n\n"
        f"Model:  {library.model.label} ({library.model.name})\n\n"
        f"Exclusion Patterns:  {exclusions}"))
