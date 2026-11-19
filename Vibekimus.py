"""Vibekimus — AI media search. Start the program with:  python Vibekimus.py

The code lives in the `vibekimus` package; the settings are in vibekimus/settings.py.
"""
if __name__ == "__main__":
    # Imported here: the text reader's worker processes load this file too and need none of it.
    from vibekimus.ui.main_window import main
    main()
