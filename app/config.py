from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    chunk_size: int = 120
    chunk_overlap: int = 30
    default_top_k: int = 3
    index_path: str = str(Path(__file__).resolve().parents[1] / "data" / "index.pkl")


settings = Settings()
