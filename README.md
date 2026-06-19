
## Структура проекта
```
avsr/
    ├── data/         # Токенизатор, датасеты, аугментации
    ├── models/       # Архитектура модели
    ├── training/     # Функции обучения, loss
    ├── inference/    # Инференс, декодирование, транскрибация
    └── utils/        # Метрики, утилиты

```

## Чекпоинты

Предобученные модели доступны на Яндекс.Диске:

https://disk.yandex.ru/d/In1EVtg2NzsKUw
<!-- 
| Модель | Датасет | WER | Ссылка |
|--------|---------|-----|--------|
| conformer_yt_audio_2.pt | CV17 + YouTube | 0.537 | [скачать](https://disk.yandex.ru/ссылка) | -->

Для старта обучения положить в `checkpoints/`.

### Данные

- Mozilla Data Collective: https://mozilladatacollective.com/datasets
- OpenSTT public_youtube: https://github.com/snakers4/open_stt
