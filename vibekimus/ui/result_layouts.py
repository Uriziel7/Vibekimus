"""How a page of results is laid out in the results view: a grid of cards, or, for a
duplicates search, a list of groups (the copy to keep, its duplicates under it).

Both page over "items" (a result or a group) and give the view their tiles (cards
or rows) for selection; the view owns pages, selection and the thumbnail loader."""
from vibekimus.duplicates import split_groups
from vibekimus.ui.duplicate_groups import DuplicateGroupView
from vibekimus.ui.justified import MAX_ASPECT, clamp_aspect, justified_rows
from vibekimus.ui.result_card import ResultCard
from vibekimus.ui.result_row import ResultRow

TILE_GAP = 4                 # between the tiles of the grid
# Pictures are loaded this much larger than the row height: rows are stretched to fill
# the width, and a picture is then cropped to its tile.
SOURCE_SCALE = 1.4
ASPECT_TOLERANCE = 0.1       # a loaded picture this much off its planned shape gets a new slot
LIST_THUMBNAIL_SCALE = 0.5   # list rows show thumbnails at half the grid size (Ctrl + wheel)…
MIN_LIST_THUMBNAIL = 72      # …but tall enough for the row's four lines of text
GROUP_GAP = 6                # between groups of the duplicates list


class GridLayout:
    """Tiles in justified rows (see justified.py): the row height follows the
    thumbnail size (Ctrl + wheel), every row fills the width exactly. Tiles are made
    at once from the sizes stored in the index and get their pictures as they load; a
    picture whose real shape differs (a photo turned upright by its EXIF tag) makes
    the page lay itself out again."""

    item_noun = "results"

    def __init__(self, view):
        self._view = view
        self._tiles = []
        self._aspects = []
        self._width = 0

    def items(self, results):
        return list(results)

    @staticmethod
    def paths(items):
        return {result.path for result in items}

    def render(self, page, on_done):
        view, row_height = self._view, self._view.thumbnail_size
        self._aspects = [clamp_aspect(result.width, result.height) for result in page]
        self._tiles = [ResultCard(view.grid_frame, result, result.path in view.selected, view.tile_callbacks,
                                  view.drag_type) for result in page]
        self._place_tiles()
        position = {id(result): index for index, result in enumerate(page)}
        box = (round(row_height * SOURCE_SCALE * MAX_ASPECT), round(row_height * SOURCE_SCALE))
        view.loader.request(page, box, lambda result, image: self._picture_ready(position[id(result)], image),
                            on_done)

    def _picture_ready(self, index, image):
        self._tiles[index].set_source(image)
        actual = clamp_aspect(*image.size)
        if abs(actual - self._aspects[index]) > ASPECT_TOLERANCE * self._aspects[index]:
            self._aspects[index] = actual
            self._place_tiles()

    def _place_tiles(self):
        if not self._tiles or not self._tiles[0].winfo_exists() or self._width <= 0:
            return   # nothing shown, or the page was cleared
        slots, height = justified_rows(self._aspects, self._width, self._view.thumbnail_size, TILE_GAP)
        for tile, slot in zip(self._tiles, slots):
            tile.place(x=slot.x, y=slot.y, width=slot.width, height=slot.height)
        self._view.grid_frame.configure(height=height)
        self._view.update_scroll_region()

    def tiles(self):
        return [tile for tile in self._tiles if tile.winfo_exists()]

    visible_tiles = tiles

    def arrange(self, width):
        """Rows as wide as the view. Returns (left, width) of the area used."""
        if width != self._width:
            self._width = width
            self._place_tiles()
        return 0, width

    def new_results(self):
        """A new search is shown."""

    def selection_changed(self, selected):
        """Cards show their own tick; nothing else to update."""


class GroupListLayout:
    item_noun = "groups"

    def __init__(self, view):
        self._view = view
        self._groups = []      # DuplicateGroupViews of the shown page
        self._open = set()     # numbers of the groups opened (kept across pages and zooming)

    @staticmethod
    def items(results):
        return [group for group in split_groups(results) if len(group) > 1]

    @staticmethod
    def paths(items):
        return {result.path for group in items for result in group}

    def _thumbnail_size(self):
        return max(MIN_LIST_THUMBNAIL, int(self._view.thumbnail_size * LIST_THUMBNAIL_SCALE))

    def render(self, page, on_done):
        view, size = self._view, self._thumbnail_size()

        def make_row(parent, result, background):
            return ResultRow(parent, result, result.path in view.selected, view.tile_callbacks, size, background,
                             view.drag_type)

        def load_thumbnails(rows, done=lambda: None):
            row_of = {id(row.result): row for row in rows}
            view.loader.request([row.result for row in rows], (size, size),
                                lambda result, image: row_of[id(result)].set_thumbnail(image), done)

        self._groups = []
        for group in page:
            shown = DuplicateGroupView(view.grid_frame, group, make_row, load_thumbnails, view.update_scroll_region,
                                       self._remember_open, view.select_paths)
            shown.show_selection(view.selected)
            shown.pack(fill="x", pady=(0, GROUP_GAP))
            if group[0].group in self._open:
                shown.toggle(animate=False)
            self._groups.append(shown)
        view.update_scroll_region()
        load_thumbnails([shown.keeper_row for shown in self._groups], on_done)

    def _remember_open(self, number, is_open):
        (self._open.add if is_open else self._open.discard)(number)

    def tiles(self):
        return [row for shown in self._groups if shown.winfo_exists() for row in shown.rows()]

    def visible_tiles(self):
        return [row for shown in self._groups if shown.winfo_exists() for row in shown.open_rows()]

    def selection_changed(self, selected):
        for shown in self._groups:
            if shown.winfo_exists():
                shown.show_selection(selected)

    @staticmethod
    def arrange(width):
        return 0, width

    def new_results(self):
        self._open.clear()
