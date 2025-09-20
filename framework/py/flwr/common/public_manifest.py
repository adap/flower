from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PublicManifest:
    public_id: str
    num_samples: int
    hashes: Optional[List[str]] = None     # optional integrity
    classes: Optional[List[str]] = None    # optional labels
