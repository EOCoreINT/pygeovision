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
        """Point to where a real dataset can actually be obtained.

        Real fix, confirmed necessary by direct inspection: this
        previously created an empty output directory and returned it
        as if a real download had happened, for the small number of
        entries that have a real download_url set -- with zero actual
        file-fetching code anywhere in the function. Checking those
        real URLs directly confirmed why: EuroSAT, BigEarthNet, DOTA,
        LEVIR-CD, and CropHarvest all point to human-facing project
        pages (GitHub repos, project websites), not direct file links.
        A real HTTP GET against these would download HTML, not the
        dataset -- worse than the previous behavior, since it would
        produce a real-looking but wrong file. No automated download is
        genuinely possible for any of these without navigating
        registration/terms/manual download links as a human, so this
        now raises clearly for every real dataset name rather than
        silently creating a misleading empty "downloaded" directory.

        Raises:
            ValueError: Always -- with the real URL to visit, whether
                or not `download_url` happens to be set (neither case
                supports a genuine automated download today).
        """
        from pygeovision.datasets.registry import dataset_registry
        d = dataset_registry[name]
        real_url = d.download_url or d.paper_url
        raise ValueError(
            f"'{name}' has no real, automated download path -- "
            f"{'its real download_url is a human-facing project page, not a direct file link' if d.download_url else 'no download_url is set for this entry'}. "
            f"Visit {real_url!r} directly to obtain it."
        )
