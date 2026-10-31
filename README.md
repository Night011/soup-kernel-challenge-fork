# Soup Kernel Challenge — Atlas Solutions

Решение ускоряет фиксированное обучение Soup 0.77.0 на Kaggle T4 с помощью собственных Triton-ядер и функций autograd. Harness подключает его через `solution.apply()`. Изменения находятся только в `solution/` и этом README.

Сохраняются Qwen2.5-1.5B-Instruct, параметры NF4, данные и порядок батчей, все 64 шага, LoRA r=16 на `q_proj`/`v_proj` с dropout 0.05, оптимизатор и расписание learning rate. Вместе с адаптером сохраняются метрики и построчные потери `rewind.jsonl`.

- Репозиторий команды: [BAITC-Hacks/hack-d5eb16fb-atlas-solutions](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions).
- Рабочий Kaggle-ноутбук: [notebook00ca772363](https://www.kaggle.com/code/hoogle57/notebook00ca772363).
- Полностью проверенный commit: [f795e0c3da4a88e1c9e907fa1df298b950682adc](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions/commit/f795e0c3da4a88e1c9e907fa1df298b950682adc).

**Статус сдачи:** ссылка на финальную публичную сохранённую версию Kaggle и её точный `COMMIT` будут зафиксированы после последнего полного замера. Рабочая ссылка выше пока не является заявлением о готовой финальной версии.

## Подтверждённые результаты

Последнее полное сравнение выполнено для `f795e0c`, каталог Kaggle `runs_full_f795e0c`. Порядок: base, solution, solution, base. Измеряется весь процесс `soup train`, включая запуск, импорты, холодную компиляцию Triton и сохранение адаптера. Для двух запусков median и среднее совпадают; ниже значения из `result.json`.

| Показатель | Base Soup | Наше решение | Отношение |
|---|---:|---:|---:|
| Весь `soup train`, два запуска | 77.72 с | 59.97 с | ускорение 1.296× |
| Пиковая видеопамять | 14348 МиБ | 2658 МиБ (2.60 ГиБ) | base/solution 5.398× |
| Eval loss на открытом наборе | 2.04966 | 2.04968 | отклонение +0.001% |

Base: 76.29 и 79.14 с; solution: 60.00 и 59.93 с. **Все 10 проверок правил пройдены**, включая 79 регрессионных тестов Soup; GPU-тесты решения — **71/71**. Минимальная доля собственных ядер в двух solution-запусках — **15.0%**, измеренная с третьего шага. Starter SwiGLU в эту долю не входит.

Одиночный запуск [23e71d6](https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions/commit/23e71d6) дал 56.92 с, 2658 МиБ и eval loss 2.04965; 71 GPU-тест прошёл. Он не устанавливает проверки, которым нужен baseline. Последующие изменения размера частей головы, подготовки компилятора и объединения residual/RMSNorm требуют своего полного сравнения. Таблица относится именно к `f795e0c`. Цель 40 с / 2500 МиБ в этом полном сравнении не достигнута.

## Как воспроизвести на Kaggle T4

### 1. Сессия и доступ

Используйте `templates/kaggle_notebook.ipynb`. В Kaggle включите **Accelerator: GPU T4 x2** и **Internet: on**. Понадобится аккаунт с подтверждённым номером телефона. Harness ограничивает измеряемый процесс одной T4, GPU 0; вторую карту решение не использует.

Для закрытого репозитория добавьте в **Add-ons → Secrets** секрет `GITHUB_TOKEN` с правом чтения основного репозитория и разрешите ноутбуку пользоваться им. Не вставляйте токен в код или вывод ячеек. Для публичного репозитория секрет не нужен. Необязательный `HF_TOKEN` помогает избежать ограничений загрузки Hugging Face.

### 2. Точный исходный код

Чтобы повторить подтверждённую таблицу, в первой кодовой ячейке шаблона задайте:

```python
REPO_URL = "https://github.com/BAITC-Hacks/hack-d5eb16fb-atlas-solutions.git"
COMMIT = "f795e0c3da4a88e1c9e907fa1df298b950682adc"
```

Пустой `COMMIT` берёт текущую ветку и не фиксирует воспроизводимый результат. Для итоговой сдачи значение должно совпадать с commit в финальной сохранённой версии ноутбука, README и `SUBMISSION.md`.

### 3. Зависимости и клонирование

Выполните штатную ячейку установки версий из шаблона:

```python
!python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
```

Затем выполните штатную ячейку клонирования и checkout. Если прежний рабочий каталог был удалён, сначала выполните `%cd /kaggle/working`, затем клонирование. Следующие команды запускаются в `/kaggle/working/repo`.

### 4. Полное сравнение

До измерения убедитесь, что в сессии нет других работающих GPU-процессов. Последовательно выполните:

```python
%cd /kaggle/working/repo
!python task/prepare_data.py
!python -m pytest solution -q -p no:cacheprovider
!python harness/run_task.py --repeats 2 --out runs_reproduce_f795e0c
!python harness/check_rules.py --result runs_reproduce_f795e0c/result.json
!python harness/score.py runs_reproduce_f795e0c/result.json
```

Для повторного сравнения используйте новое имя выходного каталога: существующие результаты harness может перезаписать. `--repeats 2` выполняет по два base и solution в установленном порядке. Eval оценивает сохранённые адаптеры обычными transformers/PEFT без кода решения.

### 5. Выходные файлы и успешный итог

- `runs_reproduce_f795e0c/result.json` — время, память, метрики и eval.
- `runs_reproduce_f795e0c/checks.json` — все десять проверок с причинами.
- `runs_reproduce_f795e0c/<base_1|base_2|solution_1|solution_2>/train.log` — подробности каждого обучения.
- В каталогах запусков — адаптеры и исходные выходы Soup, включая rewind.

Успешный итог: GPU-тесты проходят, `check_rules.py` выводит **ALL CHECKS PASSED**, включая все 79 тестов Soup. Одно наличие `result.json` не означает допуск.

При разработке `--only solution` быстрее, но не устанавливает baseline-зависимые проверки. `--profile` добавляет отдельный профилируемый запуск; его время не является временем основного solution. В финальной сохранённой версии нужно выполнить **Save & Run All** и проверить результаты именно её commit.

## Что оптимизировано

### Голова, loss и метрики

В словаре 151936 токенов. `ck_head.py` обрабатывает нужные позиции частями вместо полного буфера logits. В текущем разрабатываемом варианте часть содержит 256 строк; проверенный `f795e0c` использовал 128. cuBLAS умножает FP16 hidden states и веса с FP32 выходом. Собственное потоковое Triton-ядро получает logsumexp, NLL, энтропию и argmax. Одни и те же NLL используются для loss и rewind.

Локальный градиент hidden рассчитывается, пока logits доступны, с текущим AMP scale. Backward применяет фактический входящий масштаб и scatter, без повторного расчёта logits. Сохраняется объединение позиций, нужных attention mask и shifted labels: padding-позиция иногда предсказывает первый реальный токен. Позиции с labels идут первыми внутри головы; порядок примеров не меняется.

Collator-обёртка выводит индексы, число токенов и схему padding из того же CPU-батча до переноса на GPU. Эти метаданные удаляются из аргументов модели. GPU-скаляр знаменателя загружается ядром CE. Энтропия и точность передаются на CPU перед штатной записью TRL, без синхронизации внутри backward головы. При отсутствии CPU-метаданных остаётся GPU-подготовка.

### Замороженная голова и embedding

Точная исходная FP32-копия остаётся в RAM, compute-копия FP16 — на GPU. Для общих весов головы и embedding требуется точное восстановление FP32-значений из FP16. Собственная редукция проверяет равенство без полного временного FP32-тензора; lookup-ядро выдаёт исходные FP32 embedding-значения.

Начиная с `23e71d6`, GPU-мастер освобождается до выделения compute-копии, которая строится из небольших FP32-частей с CPU. Штатный forward без градиентов и отключение решения восстанавливают исходный мастер. Внешний eval адаптера не использует эти подмены.

Текущий проверяемый вариант также переносит обычный перевод общих весов головы из FP16 в FP32 на этап подготовки через CPU. На GPU не требуется одновременно удерживать исходную FP16-копию и новую FP32-копию. Конечные значения, точность, объект Parameter и CUDA-устройство сохраняются; исходная функция PEFT по-прежнему выполняет заморозку и настройку gradient checkpointing с прежними аргументами. Новый вариант требует собственного замера.

### NF4, MLP и сохранённые активации

Собственное ядро объединяет NF4 lookup, декодирование double-quant scales и offset с исходными codebooks. GPU-тесты требуют побитового совпадения декодирования с bitsandbytes. Парное ядро создаёт общую матрицу gate/up: две отдельные проекции заменяются одним GEMM; объединяется и GEMM градиента входа. Бюджет NF4 decode-кэша сейчас нулевой: веса восстанавливаются по необходимости и не удерживаются между использованиями. Исходные упакованные веса заморожены.

Paired MLP сохраняет узкий FP16-вход нормализации вместо двух широких gate/up активаций. Backward восстанавливает их из входа и неизменённых весов, затем использует общую матрицу для градиента входа. Память уменьшается ценой дополнительного GEMM. INT8-хранение остаётся в fallback на основе starter SwiGLU, но не является политикой paired MLP в полном замере выше.

RMSNorm сохраняет FP16-вход и исходный FP32 `rstd`. Сохранение исходного FP32-входа в FP16 вносит небольшое приближение в backward. Его влияние проверяется тестами градиентов и полным сравнением loss, метрик и eval. Побитовая эквивалентность всего обучения не заявляется.

### Padding, attention и LoRA

`integrate.py` удаляет полностью пустые крайние столбцы батча с сохранением position ids. Для подходящих батчей `ck_packed.py` держит реальные токены во всех слоях трансформера. Перед LoRA-dropout данные временно разворачиваются в исходную форму: native PyTorch dropout потребляет прежний поток случайных чисел. Нужные значения снова уплотняются для LoRA GEMM, а маска сохраняется битами.

Если shifted label требует padding-query, уплотнение отключается для батча. Для checkpoint replay, KV cache и неподдерживаемого attention предусмотрены исходные пути. Примеры не переносятся между батчами и не смешиваются.

Attention получает исходные границы примеров через cumulative sequence offsets и использует существующие PyTorch `aten._efficient_attention_forward/backward`. RoPE сохраняет оригинальные позиции токенов. Наши ядра выполняют Q/K/V packing, FP16 cast, GQA expansion и восстановление/суммирование градиентов. Сам native attention не выдаётся за собственное ядро. Быстрый путь требует нулевого attention dropout и отсутствия past cache.

Другие объединения: совместный RoPE Q/K, LoRA scale/add с сохранением порядка округлений PEFT, residual плюс RMSNorm с общим backward. В текущем варианте residual/RMSNorm объединяется и в уплотнённом трансформере.

### Холодная компиляция

Подготовка использует compile-only API Triton и инициализирует launchers/CUDA handles без запуска GPU-ядер. В текущем варианте она начинается в `solution.apply()` и совмещается с обычной загрузкой модели; до возврата из соответствующего setup все потоки завершены. Используется до четырёх compiler threads внутри измеряемого процесса. Harness задаёт свежий пустой кэш; всё время подготовки входит во внешний wall time. Эта подготовка не добавляется к заявленной доле GPU-ядер.

## Собственные ядра

| Файл в `solution/` | Ядра | Работа |
|---|---|---|
| `ck_head.py` | `_ck_cast_kernel`, `_ck_head_exact_kernel`, `_ck_embedding_kernel`, `_ck_lse_kernel`, `_ck_ce_grad_kernel`, `_ck_scatter_grad_kernel` | Cast, проверка точности, lookup, loss/метрики, CE-градиент и scatter |
| `ck_nf4.py` | `_ck_nf4_kernel`, `_ck_nf4_pair_kernel` | Точное одиночное и парное NF4-декодирование |
| `ck_rms.py` | `_ck_rms_fwd_kernel`, `_ck_rms_bwd_kernel` | RMSNorm и границы типов |
| `ck_residual_norm.py` | `_ck_residual_norm_fwd_kernel`, `_ck_residual_norm_bwd_kernel` | Residual, RMSNorm и совместный градиент |
| `ck_rope.py` | `_ck_rope_pair_kernel` | Совместный RoPE Q/K и backward |
| `ck_lora.py` | `_ck_lora_add_kernel`, `_ck_lora_grad_kernel` | LoRA scale/add и градиент |
| `ck_dropout.py` | `_ck_dropout_pack_kernel`, `_ck_dropout_bwd_kernel` | Native-маска, cast и градиент |
| `ck_activation.py` | `_ck_pack_activations_kernel` | Компактные активации для fallback |
| `ck_padding.py` | `_ck_mlp_residual_kernel`, `_ck_gather_rows_kernel`, `_ck_scatter_rows_kernel` | Перемещение уплотнённых строк и residual |
| `ck_attention.py` | `_ck_attention_pack_kernel`, `_ck_attention_unpack_grad_kernel` | Q/K/V packing, GQA/cast и градиенты |

Python-обёртка `ck_packed.py` не считается отдельным GPU-ядром. `swiglu.py` основан на примере starter kit: его `example_*`, включая адаптации хранения и strides, исключены из нашей доли. Большие GEMM и SDPA остаются работой PyTorch/cuBLAS. Неиспользуемое `_ck_ce_lse_kernel` не заявляется. Проверка точности головы относится к setup; доля обучения берётся из измеряемых harness шагов.

## Ограничения

Быстрые пути рассчитаны на замороженный NF4 Qwen2 с FP16 autocast на T4. Встроенные флаги оптимизации Soup не меняются. Не меняются allocator, выбор GPU и compiler cache harness; пакеты внутри измерения не устанавливаются. Неподдерживаемые модели, геометрии и режимы PEFT используют guarded fallback.

CPU-мастер требует RAM и начальных передач, время которых учитывается. Нулевой NF4-кэш и восстановление MLP-активаций добавляют вычисления ради памяти. FP16-входы нормализации и INT8-активации в fallback приближённые. Открытый набор не гарантирует те же время и качество на скрытых данных жюри.

Версия с меньшей памятью из полной таблицы медленнее раннего одиночного packed-запуска `98c32f4` — 53.57 с / 3400 МиБ. Для него baseline-зависимые условия не проверялись. Снижение eval loss ниже исходной модели не является целью изменения фиксированной задачи: контролируется допустимое отклонение качества.

## Источники идей и авторство

Новые ядра написаны самостоятельно. Готовые сторонние библиотеки ядер и чужие реализации ядер не импортируются и не копируются. Источники идей:

- Частичный linear/CE и локальная производная во forward: [Liger fused linear cross-entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py).
- Устойчивая online-softmax редукция: [статья FlashAttention](https://arxiv.org/abs/2205.14135).
- Объединение gate/up: [NVIDIA TensorRT-LLM post-load fusion](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/transforms/post_load_fusion.html).
- Исключение padding: [PyTorch transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).
- Интерфейс attention переменной длины: [схемы PyTorch ATen](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml), `_efficient_attention_forward/backward`; код backend-ядер не копируется.
- Выравнивание FP16 GEMM: [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
- Компактное состояние dropout: [Triton low-memory dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html). Наш путь упаковывает фактическую native-маску и сохраняет RNG.
- Семантика NF4: установленный bitsandbytes `QuantState`; границы типов: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

Формулы метрик TRL 0.29.1 и контракт chunked-step/rewind Soup сохранены. Авторство starter SwiGLU указано в исходнике. Уведомления Apache-2.0 находятся в `LICENSE` и `NOTICE`; модель и данные сохраняют свои лицензии.
