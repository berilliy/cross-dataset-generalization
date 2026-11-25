Robust Cross-Dataset Generalization of Machine Learning-Based Network Intrusion Detection Under Temporal and Distributional Shift

Финальные материалы исследования • 25 ноября 2026

Начните с research.html. Это исследовательский отчёт с таблицами и пятью встроенными рисунками. research_corrected.txt содержит тот же текст для редактирования. Численные значения и подписи можно переносить в статью. Файлы в figures доступны как PNG и векторные PDF. Методика, карта исходных данных, матрица работ literature_comparison.csv и библиография supplement/references.bib пригодны для подготовки разделов Methods и Related Work.

Что воспроизводится
• Обучение пяти базовых алгоритмов в едином трёхдоменном сравнении. 720 строк измерений с двумя порогами.
• 120 строк сравнения исходных и сокращённых признаков по календарным периодам.
• Прогноз на 123 117 строках нового внешнего стенда, 90 наборов оценок риска и 180 строк измерений. Целевая коллекция не входит в обучение.
• 36 проверок чувствительности к окну и бюджету калибровки. Все пороги, результаты и исходные оценки сохранены.
• 44 дополнительные группы оценок прежнего протокола и 28 условных сравнений моделей и источников.

Повторение на одном компьютере
1. Распакуйте архив целиком, сохранив структуру папок. Нужны Python 3.14.2 и пакеты из requirements-verified.txt. Сначала создайте виртуальное окружение и установите зависимости из корня проекта:
   python -m venv .venv
   Windows PowerShell: .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-verified.txt
   python -m ipykernel install --sys-prefix --name python3 --display-name "Python (NIDS research)"
2. Откройте notebook3ec3218ec0_corrected.ipynb в Jupyter или VS Code, выберите это окружение. Рабочая папка ядра — каталог notebook. Restart Kernel → Run All запускает обучение в режиме train. Равнозначная команда из корня:
   python source/run_notebook.py
   Скрипт сохраняет новый выполненный notebook после успешного завершения. Для сохранения выданного снимка работайте в отдельной распакованной копии. Данные включены. Сеть нужна только при первоначальной установке пакетов.
3. Пересоберите отчёт и повторите проверки из корня той же копии:
   python source/report.py
   python source/verify_new_results.py
   python source/verify_final.py
   python source/check_splits.py
   Первый скрипт создаёт figures, tables и две формы отчёта. Проверки читают run_results. Если задан NIDS_OUTPUT, перенесите результат в run_results отдельной копии перед этими шагами либо используйте стандартный каталог с самого начала. NIDS_MODE=replay воспроизводит только прежние сравнения из saved_results и по умолчанию пишет в replay_results.
4. Полный пакет уже содержит результаты этих шагов. validation.json описывает проверенный запуск. Чтобы повторить описательную проверку единиц длительности отдельно, запустите python source/feature_audit.py. В supplement/verification_new.csv — проверки внешней оценки и калибровки, в run_results/independent_export_verification.csv — базовой матрицы и календарной проверки. Для проверки вычислительной повторяемости предусмотрена отдельная команда python source/repeat_run.py. Она начинает ещё одно полное обучение с новым ядром во временной папке и сравнивает значения с run_results. Результаты этой проверки лежат в supplement/repeat_run.json и repeat_run_comparison.csv. Поправка для фиксированного порядка суммирования прогнозов Random Forest записана в protocol/computational_amendment.json. Изменение результатов этого алгоритма относительно первого запуска v3 приведено в supplement/rf_implementation_comparison.csv.

5. package_sha256.csv относится к выданным файлам до повторного запуска. Проверить копию архива после распаковки можно командой:
   python source/check_package.py

Где искать числа для статьи
RQ1 и RQ2  run_results/controlled_benchmark.csv, tables/majority_summary.csv, tables/duration_summary.csv
RQ3        tables/shift_conditions.csv, tables/shift_correlations.csv, run_results/benchmark_class_conditional_shift.csv и benchmark_attack_errors.csv
RQ4        run_results/strict_temporal_metrics.csv, calibration_sensitivity.csv, native_ablation.csv и native_feature_manifest.json
RQ5        tables/benchmark_pooling.csv, tables/external_pooling.csv, tables/external_summary.csv
Прогнозы   run_results/benchmark_predictions, native_predictions, external_scores и external_target.parquet
Ошибки     run_results/external_error_by_label.csv, native_attack_protocol_errors.csv, benchmark_attack_errors.csv
Данные     training_samples и external_data, происхождение в supplement/methods.txt и external_data/release.json
Протокол   protocol/v2_protocol.json и protocol/v3_protocol.json, записи контрольных сумм в той же папке

Запуск из интерфейса Kaggle
Локальная среда повторного обучения описана в run_results/runtime.json. Для запуска на Kaggle прикрепите полный ZIP как входной файл и импортируйте notebook. В начале рабочей копии добавьте ячейку подготовки путей. Она выбирает единственный прикреплённый файл с указанным именем, извлекает собственный архив проекта в доступный для записи каталог и делает его рабочим каталогом ядра:

from pathlib import Path
from zipfile import ZipFile
import os
archives = list(Path("/kaggle/input").rglob("Research_project_final.zip"))
assert len(archives) == 1, "Attach one project archive or specify its exact path"
with ZipFile(archives[0]) as package:
    package.extractall("/kaggle/working/nids_project")
os.chdir("/kaggle/working/nids_project/Research_final")

После этого Run All использует приложенные выборки. Если пакет данных прикреплён уже распакованным, скопируйте каталог Research_final в /kaggle/working и укажите его в os.chdir. Полное численное совпадение двух финальных запусков проверялось в локальной закреплённой среде. Версии пакетов и платформу конкретного запуска Kaggle можно сравнить по его runtime.json.

Оригинальные исходные тесты проекта были известны раньше. Внешний набор проверен после локальной фиксации нового протокола. Самостоятельное воспроизведение другой командой, проверка в действующей сети и одинаковое выделение потоков из сырых пакетов остаются за пределами этих данных.

