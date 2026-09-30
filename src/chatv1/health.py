from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HealthReport:
    status: str
    checkpoint_present: bool
    artifacts_dir_present: bool


def check_health(checkpoint: str = "artifacts/tiny_chatv1.pt") -> HealthReport:
    checkpoint_path = Path(checkpoint)
    artifacts_dir = checkpoint_path.parent
    present = checkpoint_path.is_file()
    return HealthReport(
        status="ok" if present else "degraded",
        checkpoint_present=present,
        artifacts_dir_present=artifacts_dir.is_dir(),
    )
