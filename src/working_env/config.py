import os
from dataclasses import dataclass

@dataclass
class EnvConfig:
    # --- Пути ---
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    
    # --- Логирование ---
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"
    
    # --- Trainer defaults ---
    accelerator: str = "auto"
    devices: str = "auto"
    log_every_n_steps: int = 10
    
    # --- Воспроизводимость ---
    seed: int = 42
    
    

    def setup_dirs(self):
        """Создает базовые папки, если их нет."""
        for d in [self.checkpoints_dir, self.logs_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

# Глобальный экземпляр
env_config = EnvConfig()