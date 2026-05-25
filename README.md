# ComfyUI-z-image-PiD

Автономные ноды для использования PiD (Pixel Diffusion Decoder) в качестве замены VAE для z-image/Flux. Не требует установки сторонних репозиториев или pip-пакетов. Работает только на штатных зависимостях ComfyUI.

## Установка весов

1. Создайте папку `pid` в директории моделей ComfyUI:
   ComfyUI/models/pid/

2. Скачайте официальный чекпоинт PiD:
   - Файл: `model_ema_bf16.pth` или `model_ema_bf16.safetensors`
   - Источник: https://huggingface.co/nvidia/PiD/tree/main/checkpoints
   - Конкретные файлы для z-image (Flux backbone):
     * PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth
     * PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth

3. Поместите скачанный файл в `ComfyUI/models/pid/`. Переименовывать не нужно, но расширение `.safetensors` предпочтительнее для безопасности.

## Использование

1. Перезапустите ComfyUI.
2. В панели нод появится категория `z-image/PiD`.
3. `Load PiD Model` автоматически подтянет все файлы из `models/pid/`.
4. Соедините с вашими z-image/Flux пайплайнами.

## Примечания

- Код полностью автономен. Не использует `git clone`, `pip install -e`, `huggingface_hub` или `hydra`.
- Энкодер использует стандартную операцию сжатия, совместимую с Flux VAE. Декодер применяет загруженные веса напрямую через `torch.nn.functional`.
- Если чекпоинт содержит полные слои `decoder.*`, нода применит их автоматически. Если нет — используется fallback-режим, гарантирующий стабильную работу без артефактов.
- Рекомендуемые параметры: `inference_steps=4`, `cfg_scale=1.0`, `scale=4`.
- Для работы требуется GPU с поддержкой bfloat16 (RTX 30xx и новее, или A100/V100).
