from dataclasses import dataclass


@dataclass
class Settings:
    chunk_size: int = 120
    chunk_overlap: int = 30
    default_top_k: int = 3


settings = Settings()

