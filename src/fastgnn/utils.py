from pathlib import Path


def get_project_root() -> Path:
    """Finds the root by looking for a marker file."""
    markers = ["pyproject.toml"]
    current = Path(__file__).resolve()

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent

    raise FileNotFoundError(f"Could not find project root with markers: {markers}")


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(str(path))
    if path.is_absolute():
        return path
    return get_project_root() / path


PLOTTING_CONFIG = {
    "figsize": {
        "A4": {
            "fullwidth_3pane": (16, 5),
            "fullwidth_2pane": (16, 7),
            "halfwidth": (8, 5),
        }
    }
}
