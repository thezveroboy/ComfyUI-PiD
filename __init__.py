import os
import folder_paths

# Безопасная регистрация папки для весов
_PID_MODEL_DIR = os.path.join(folder_paths.models_dir, "pid")
if os.path.isdir(_PID_MODEL_DIR):
    try:
        folder_paths.add_model_folder_path("pid", _PID_MODEL_DIR, True)
    except Exception:
        pass  # Игнорируем, если папка уже зарегистрирована

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]