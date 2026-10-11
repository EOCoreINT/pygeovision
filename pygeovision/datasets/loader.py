"""DatasetLoader — unified interface for downloading and loading datasets."""
from __future__ import annotations

from pathlib import Path


class DatasetLoader:
    """Unified loader for any dataset in the registry."""
    def __init__(self, data_root: str = "~/.pygeovision/datasets"):
        self.data_root = Path(data_root).expanduser()

    def info(self, name: str) -> None:
        from pygeovision.datasets.registry import dataset_registry
        d = dataset_registry[name]
        print(f"\n{'─'*60}")
        print(f"  {d.name}")
        print(f"{'─'*60}")
        for k, v in d.to_dict().items():
            if v: print(f"  {k:<18}: {v}")

    def download(self, name: str, output_dir: str | None = None) -> Path:
        """Explain where dataset ``name`` can be obtained; it never downloads anything.

        The catalogue holds metadata only, and the few recorded ``download_url`` values
        are project pages (GitHub repositories, registration forms), not file links, so
        an automated GET would save an HTML page that looks like data.

        Raises:
            KeyError: ``name`` is not in the catalogue.
            ValueError: Always otherwise, with the page to visit when one is recorded.
        """
        from pygeovision.datasets.registry import dataset_registry
        d = dataset_registry[name]
        if d.download_url:
            where = f"Its recorded download_url is a project page, not a direct file link: {d.download_url}"
        elif d.paper_url:
            where = f"No download_url is recorded; the dataset's paper is at {d.paper_url}"
        else:
            where = (f"No download_url or paper_url is recorded for it; search the web for "
                     f"'{d.name} {d.year} dataset' to find its official page")
        raise ValueError(
            f"'{name}' has no automated download path -- the catalogue holds metadata only. {where}."
        )
