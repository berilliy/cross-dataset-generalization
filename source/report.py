"""Render a research dossier from the completed run. No model is fitted here."""
from pathlib import Path
import base64, html, json, re
import markdown
import numpy as np
import pandas as pd
from summarize import label, SHORT, MODELS, summarize
from feature_audit import audit

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"tables"
R=ROOT/"run_results"
S=ROOT/"supplement"
get=lambda folder,name: pd.read_csv(folder/(name+".csv"),float_precision="round_trip")
def table(frame):
    return frame.to_markdown(index=False,floatfmt=".4f",disable_numparse=True)
def fmt(x):
    return f"{x:.4f}".replace(".",",")
def count(x):
    return f"{int(x):,}".replace(",", " ")
def pct(x):
    precision = 4 if 0 < x < .0001 else 2
    return f"{100*x:.{precision}f}".replace(".",",")+"%"
def direction(r):
    return label(r.sources)+" → "+label(r.target)
def numtable(frame):
    return table(frame.apply(lambda col:col.map(fmt) if pd.api.types.is_float_dtype(col) else col))


def build():
    summarize(ROOT)
    unit=audit(ROOT)
    profiles=get(S,"duration_profiles").set_index("dataset")
    title=json.loads((ROOT/"protocol/v3_protocol.json").read_text(encoding="utf-8-sig"))["title"]
    d=get(T,"duration_summary")
    d=d[(d.model=="LightGBM")&(d.policy=="default_0.5")].sort_values(["target","scenario","sources"])
    duration=numtable(pd.DataFrame({"Источник → тест":[direction(r) for r in d.itertuples()],
        "Протокол":d.protocol,"Протокол и длительность":d.core,"Разность":d.difference}))
    b=get(R,"controlled_benchmark")
    bench=b[(b.variant=="minimal_core")&(b.policy=="default_0.5")].groupby(["target","sources","scenario","model"]).macro_f1.mean().unstack("model").reset_index().sort_values(["target","scenario","sources"])
    headline=bench[MODELS].rename(columns=SHORT).copy()
    headline.insert(0,"Источник → тест",[direction(r) for r in bench.itertuples()])
    mat=numtable(headline)
    rho=get(T,"shift_correlations")
    rho=rho.pivot(index="model",columns="policy",values="rho").loc[MODELS[1:]].reset_index()
    rho["model"]=rho.model.map(SHORT)
    corr=numtable(rho.rename(columns={"model":"Модель","default_0.5":"Порог 0,5","source_cal_fpr_1pct":"Калибровка 1%"}))
    comp=get(R,"benchmark_class_conditional_shift")
    shifts=comp.query("feature=='log_duration' and target=='TON_IOT' and sources=='CICIOT2023'").groupby("binary_label").ks.mean()
    errs=get(R,"benchmark_attack_errors").query("seed==42 and target=='TON_IOT' and sources=='CICIOT2023' and model=='LightGBM' and variant=='minimal_core' and policy=='default_0.5'").copy()
    errs["Доля ошибок"]=(errs.FP+errs.FN)/errs.n
    error_table=numtable(errs[["attack_label","n","FP","FN","Доля ошибок"]].rename(columns={"attack_label":"Метка ToN","n":"Записей"}))
    cal=get(R,"calibration_sensitivity")
    base=get(R,"strict_temporal_metrics")
    temporal=base[base.model=="LightGBM"].copy()
    temporal["Набор"]=temporal.dataset.map(label)
    temporal["Правило"]=temporal.policy.map({"default_0.5":"0,5","source_cal_fpr_1pct":"1% на калибровке"})
    temporal_table=numtable(temporal[["Набор","Правило","macro_f1","roc_auc","fpr","recall"]].rename(columns={"macro_f1":"Macro-F1","roc_auc":"ROC-AUC","fpr":"FPR","recall":"Recall"}))
    cal_show=cal[(cal.model=="LightGBM")&(cal.alpha==.01)].copy()
    cal_show["Набор"]=cal_show.dataset.map(label)
    cal_show["Окно"]=cal_show.window.map({"full":"Всё","early":"Первая половина","late":"Вторая половина"})
    cal_show["Порог"]=cal_show.threshold.map(lambda v:f"{v:.8g}".replace(".",","))
    for k,out in [("calibration_fpr","FPR в окне"),("fpr","FPR на тесте"),("recall","Recall на тесте")]:
        cal_show[out]=cal_show[k].map(pct)
    ct=table(cal_show[["Набор","Окно","benign_n","Порог","FPR в окне","FPR на тесте","Recall на тесте"]].rename(columns={"benign_n":"Нормальных"}))
    e=get(T,"external_summary")
    ep=e[(e.model=="LightGBM")&(e.variant=="minimal_core")&(e.policy=="default_0.5")].set_index("sources")
    pooled=ep.loc["UNSW_NB15+TON_IOT"]
    delta_u=pooled.macro_f1-ep.loc["UNSW_NB15"].macro_f1
    delta_t=pooled.macro_f1-ep.loc["TON_IOT"].macro_f1
    ext=e[(e.model=="LightGBM")&(e.policy=="default_0.5")].copy()
    ext["Обучение"]=ext.sources.map(label)
    ext["Признаки"]=ext.variant.map({"minimal_core":"Протокол и время","protocol_only":"Протокол"})
    external_table=numtable(ext[["Обучение","Признаки","macro_f1","fpr","recall","roc_auc","average_precision"]].rename(columns={"macro_f1":"Macro-F1","fpr":"FPR","recall":"Recall","roc_auc":"ROC-AUC","average_precision":"AP"}))
    em=get(R,"external_metrics")
    em=em[(em.model=="LightGBM")&(em.sources=="UNSW_NB15+TON_IOT")&(em.variant=="minimal_core")&(em.policy=="default_0.5")]
    exclass=get(R,"external_error_by_label")
    exclass=exclass[(exclass.key==em[em.seed.eq(42)].iloc[0].key)&(exclass.policy=="default_0.5")].copy()
    exclass["error_rate"]=exclass.errors/exclass.n
    exclass.sort_values("n",ascending=False).to_csv(T/"external_label_example.csv",index=False)
    exclass=exclass.set_index("attack_label")
    err_example=exclass.loc["DOS_SYN_Hping"];ben_example=exclass.loc["Thing_Speak"]
    # Class counts and actual rather than advertised data values come from this release.
    labels=get(R,"external_label_counts")
    benign=int(labels.loc[labels.target.eq(0),"n"].sum())
    attacks=int(labels.loc[labels.target.eq(1),"n"].sum())
    overlaps=get(T,"external_summary").query("model=='LightGBM' and variant=='minimal_core' and policy=='default_0.5'").feature_overlap
    nat=get(R,"native_ablation").query("model=='LightGBM' and policy=='default_0.5'").groupby(["dataset","variant"]).macro_f1.mean().unstack()
    m=get(T,"majority_summary").query("variant=='minimal_core' and policy=='default_0.5' and scenario!='in_domain'")
    best=m.loc[m.groupby(["target","sources"]).macro_f1.idxmax()].set_index(["target","sources"])
    loses=int((best.macro_f1<best.majority_macro_f1).sum())
    proofs=[]
    for key in [("CICIOT2023","TON_IOT"),("TON_IOT","CICIOT2023")]:
        q=best.loc[key]
        proofs.append(f"{label(key[1])} → {label(key[0])} дал {fmt(q.majority_macro_f1)} у Majority и {fmt(q.macro_f1)} у лучшего по наблюдаемому среднему алгоритма ({SHORT[q.model]})")
    v=get(R,"paired_statistical_comparisons")
    st=v[(v.experiment=="strict")&(v.target_domain=="TON_IOT")&(v.policy=="default_0.5")].iloc[0]
    cal_unsw=cal[(cal.model=="LightGBM")&(cal.dataset=="UNSW_NB15")]
    suffix=f"""
## Что показало исследование

У одного и того же классификатора результат переноса менялся с источником данных и с тем, как вычислены признаки. При одинаковых 60 000 обучающих записей объединение двух источников повысило Macro-F1 LightGBM на UNSW-NB15 по сравнению с каждым источником по отдельности. На CICIoT2023 объединение дало худший из трёх результатов. Внешняя проверка на RT-IoT2022 дала для объединённой модели Macro-F1 {fmt(pooled.macro_f1)}, для модели на UNSW-NB15 — {fmt(ep.loc['UNSW_NB15'].macro_f1)}, для модели на ToN-IoT — {fmt(ep.loc['TON_IOT'].macro_f1)}. Здесь и далее основной результат относится к порогу 0,5 и среднему по трём запускам.

Цена выбора рабочего порога особенно заметна на UNSW-NB15. LightGBM почти идеально ранжировал февральский трафик (ROC-AUC 0,9991). Порог, оценённый на нормальном трафике января с бюджетом ложных тревог 1%, в феврале пометил как атакующие 17 285 из 80 723 нормальных записей. Девять вариантов той же схемы калибровки дали тестовый FPR от {pct(cal_unsw.fpr.min())} до {pct(cal_unsw.fpr.max())}.

## Постановка и данные

Исследование оценивает бинарную классификацию завершённых сетевых записей. RQ1 — насколько меняется качество при переходе между коллекциями? RQ2 — какие модели сохраняют преимущество над простым правилом? RQ3 — связаны ли измеренные сдвиги признаков с потерей качества? RQ4 — выдерживает ли порог переход на более поздние дни? RQ5 — даёт ли объединение источников преимущество при равном объёме обучения?

| Роль | Коллекция | Записей до разбиения | Применение |
|---|---|---:|---|
| Исходный домен | UNSW-NB15 (UNSW) | 177 906 | Междоменный и календарный тест |
| Исходный домен | Network ToN-IoT (ToN) | 179 675 | Междоменный и календарный тест |
| Исходный домен | CICIoT2023 (CIC) | 195 648 | Междоменный тест |
| Внешняя проверка | RT-IoT2022 (RT) | 123 117 | Только оценка моделей из UNSW и ToN |

Первые три выборки получены детерминированным отбором по имени исходного файла и номеру строки до разбиений. Эти тесты уже изучались в предыдущих версиях проекта. RT-IoT2022 происходит из другой экспериментальной установки. Для него заранее выбраны источники, признаки, число запусков, пороги и первичное сравнение. Официальное описание UCI, включая перечень классов, было доступно до фиксации протокола. [Карточка RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022)

В междоменном эксперименте одинаковые полные векторы признаков группировали по хешу. В каждом домене группы распределили на обучение, калибровку и тест. При оценке на конкретном домене все алгоритмы и источники получали один тест. В каждом запуске отбирали 60 000 обучающих строк, при объединении — по 30 000 из каждого источника. Четыре классификатора получили одинаковые суммарные веса на каждый класс внутри источника. Majority предсказывал более частый класс исходной обучающей выборки. У объединённых источников калибровочные выборки больше, поэтому сравнение при 0,5 лучше изолирует изменение обучающих данных.

Macro-F1 — среднее F1 двух классов с одинаковым весом. FPR показывает долю нормальных записей, по которым сработала тревога, recall — долю обнаруженных атак. ROC-AUC и average precision (AP) оценивают ранжирование. F1 при фиксированной доле атак 20% пересчитывается из recall и FPR и сохранён для сравнения при одинаковой условной распространённости.

## 1. Перенос между тремя коллекциями и Majority

**Таблица 1.** Macro-F1, порог 0,5, протокол и длительность. Каждая строка — среднее по seed 42, 7 и 2026. Тесты содержат 39 104 записи для UNSW, 29 314 для ToN и 20 433 для CIC. LR — логистическая регрессия, RF — случайный лес, HGB — гистограммный градиентный бустинг.

{mat}

На UNSW объединение дало LightGBM 0,5031, на CIC — 0,2179, на ToN — 0,2950. В последнем случае объединённый результат лежит между значениями двух отдельных источников. На каждом из трёх тестов LightGBM при обучении внутри домена превзошёл свой результат после переноса с каждого из альтернативных источников. Это ответ на RQ1 и RQ5 для указанного набора признаков.

Рейтинг алгоритмов также меняется. Внутри UNSW наибольшее среднее у RF (0,7986), внутри ToN — у LR (0,7435). При переносе UNSW → CIC LR достигает 0,4252, тогда как RF — 0,2668. В {loses} из 9 междоменных условий все четыре классификатора уступили правилу Majority по среднему Macro-F1. {proofs[0]}. В обратном направлении {proofs[1]}. Majority постоянен для каждой обученной модели. При высокой доле одного класса даже постоянная тревога получает заметный Macro-F1, оставляя нормальные записи без правильных решений. Именно поэтому сравнение привязано к одной паре источника и теста. Выбор наибольшего из четырёх средних здесь описывает таблицу, а не процедуру выбора модели для применения.

![Рисунок 1. Разность Macro-F1 обученной модели и Majority в каждой междоменной задаче. Отрицательное число означает преимущество постоянного прогноза.](figures/01_majority.png)

## 2. Что измеряет длительность потока

Общая схема содержит одну величину длительности и категорию протокола. Для UNSW использовано поле dur, для ToN — duration. Для CIC использовано flow_duration. Отдельное поле Duration в CIC относится к TTL. Три индикатора кодируют TCP, UDP и ICMP, четвёртый обозначает отсутствие всех трёх. Для CIC индикатор включается при значении исходного поля выше 0,5. После пороговой обработки проверена взаимная исключительность. Длительность преобразована через log(1 + x), преобразования для обучения фиксируются по исходной выборке. Документация CIC различает два поля, но не устанавливает эквивалентность временного интервала и правил формирования потока во всех применённых выгрузках. [Описание признаков CICIoT2023, таблица 4 в разделе Feature Extraction and Data Description](https://doi.org/10.3390/s23135941)

**Таблица 2.** LightGBM на тех же строках и при том же объёме обучения. Разность — Macro-F1 с длительностью минус Macro-F1 по одному протоколу. Полная таблица пар для всех моделей, порогов и seed входит в приложение.

{duration}

Для CIC → ToN длительность повысила Macro-F1 LightGBM, для CIC → UNSW снизила его. На UNSW результат объединения даже меняет интерпретацию: по протоколу обучение на одном CIC дало больший Macro-F1, чем объединение CIC с ToN. Добавление длительности изменило их порядок. Одно и то же имя признака не устанавливает одинаковые тайм-ауты, агрегацию пакетов или определение потока. Исключение длительности проверяет зависимость результата от этого поля. При четырёх категориях протокола одинаковые векторы естественным образом встречаются во многих сетях.

![Рисунок 2. Сдвиг Macro-F1 LightGBM после добавления длительности. Каждая линия соединяет два результата на одном и том же целевом тесте.](figures/02_duration.png)

## 3. Объясняет ли величина сдвига потерю качества

Для каждой из шести одиночных и трёх объединённых задач вычислили среднюю KS-статистику между обучающей и тестовой выборками по пяти столбцам. Потеря качества — Macro-F1 при обучении внутри целевого домена минус Macro-F1 перенесённой модели, на одном тесте. Сначала усреднили повторы внутри задачи, затем рассчитали ранговую корреляцию Спирмена между девятью точками.

{corr}

При пороге 0,5 у трёх моделей коэффициент отрицателен, у LR он положителен и мал. В этих девяти условиях больший средний KS не оказался надёжным признаком большей потери качества. Агрегация придаёт четырём связанным индикаторам протокола четыре из пяти весов. Для бинарных индикаторов KS отражает изменение доли соответствующей категории. Кроме того, разные условия используют одни и те же тестовые домены. Это ограничивает и смысл самого индекса, и точность оценки связи.

![Рисунок 3. Девять условий переноса на модель. Цвет обозначает целевой домен. Точки с общим цветом используют один тест.](figures/03_shift.png)

Разбор CIC → ToN показывает, что скрывает среднее. У длительности KS равна {fmt(shifts.loc[0])} внутри нормального трафика и {fmt(shifts.loc[1])} внутри атак, в среднем по seed. При seed 42 LightGBM дал следующие ошибки на одних и тех же целевых строках. Для нормального класса это ложные тревоги, для остальных классов — пропуски атак.

{error_table}

Доли пропусков сильно различаются даже между метками dos и ddos внутри ToN. Распределения наблюдались вместе с изменениями меток, долей классов и способов регистрации. По этому сопоставлению нельзя выделить причинный вклад длительности. В приложении доступны KS внутри каждого класса и признака, JSD по интервалам источника, доли пропусков значений и выходов за исходный диапазон.

## 4. Время, калибровка и признаки стенда

Для UNSW обучение пришлось на 22 января 2015 года, калибровка — на 23 января, тест — на 18 февраля. Для ToN обучение заканчивается 24 апреля 2019 года, калибровка занимает 25 апреля, тест — 26 и 27 апреля. Записи с одинаковым полным набором исходных признаков, встретившиеся в нескольких периодах, оставлены в первом периоде. Это календарное разбиение отличается от хеш-разбиения таблиц 1–2.

**Таблица 3.** LightGBM с исходными признаками на поздних днях. Для калиброванного правила порог расположен строго выше эмпирического 99-го процентиля оценок нормальных записей.

{temporal_table}

**Таблица 4.** Один уже обученный LightGBM на набор, неизменный поздний тест, три варианта окна с бюджетом 1%. Окно делится пополам по времени между первой и последней наблюдаемой записью. Вместе с бюджетами 0,5% и 0,1% и вторым алгоритмом проверены все 36 сочетаний. Данные для каждого порога и реализации привязаны к строкам калибровки.

{ct}

UNSW предоставил всего 25 минут 24 секунды нормального трафика в калибровке. На ToN смена половины дня заметно изменила и долю тревог, и долю обнаруженных атак. При бюджете 1% порог по полному дню ToN обнаружил всего 3 из 60 284 атак, по поздней половине — 2 из 60 284. При раннем окне и бюджете 1% он обнаруживал около 78% атак ценой приблизительно 16% ложных тревог. Половины одного дня измеряют чувствительность к доступному окну. Они не дают независимых повторов будущего дня. Показанные варианты зафиксированы перед запуском расширения, после знакомства с первоначальными тестами.

![Рисунок 4. Тестовые FPR и recall по трём исходным бюджетам ложных тревог. Во всех вариантах внутри набора сохраняются одни и те же модель и тест.](figures/04_calibration.png)

Состав атак тоже менялся. В календарном тесте ToN было 39 857 записей ddos и 20 427 password. Обе метки отсутствуют в обучающем периоде. В обучении встречались dos и scanning, которых нет в тесте. Поэтому разница между датами включает смену сценария. В дополнительной проверке из обучения исключили группы технических признаков стенда. При 0,5 Macro-F1 LightGBM изменился с {fmt(nat.loc['UNSW_NB15','native'])} до {fmt(nat.loc['UNSW_NB15','conservative'])} на UNSW и с {fmt(nat.loc['TON_IOT','native'])} до {fmt(nat.loc['TON_IOT','conservative'])} на ToN. Проверка группой не устанавливает, какое из удалённых полей повлияло на результат.

## 5. Проверка на внешнем стенде RT-IoT2022

В релизе UCI {count(benign)} нормальных и {count(attacks)} атакующих записей. Все 123 117 строк оценены в исходном порядке. Модель видела только протокол и, в соответствующем варианте, логарифм длительности. Zeek FlowMeter документирует flow_duration в секундах. Метки нормального класса сопоставлены с устройствами MQTT, ThingSpeak и Wipro. В поставляемом файле нет строк Amazon Alexa, упомянутых в карточке UCI. Фактические количества по MQTT и ThingSpeak также отличаются от распределения в описании. Для подсчётов использованы строки архива с сохранённой контрольной суммой. [Авторы и релиз UCI](https://doi.org/10.24432/C5P338) · [Определения Zeek FlowMeter](https://github.com/zeek-flowmeter/zeek-flowmeter)

Первичное внешнее сравнение зафиксировано для LightGBM с длительностью и порогом 0,5. Использованы те же обученные экземпляры, которые оценивались на CIC. Объединённая модель получила {fmt(pooled.macro_f1)} Macro-F1. Внешний результат лежит между результатами одиночных источников. Разность «объединение минус UNSW» составила {fmt(delta_u)}, «объединение минус ToN» — {fmt(delta_t)}. Все 90 сочетаний источника, модели, признаков и seed оценены при обоих исходных порогах. Параметры на RT не подбирались. При исходном пороге 0,5 объединённая модель обнаруживала {pct(pooled.recall)} атак и ошибочно помечала {pct(pooled.fpr)} нормальных записей в среднем по трём запускам. Высокая AP здесь требует учёта исходной доли атак — {pct(attacks/(attacks+benign))}.

{external_table}

![Рисунок 5. Внешний тест. Две схемы признаков, три источника и все пять алгоритмов. Столбцы показывают среднее Macro-F1 по трём запускам на одних тестовых строках.](figures/05_external.png)

Одна метка DOS_SYN_Hping занимает {count(err_example.n)} строк. У объединённого LightGBM при seed 42 доля пропусков этой метки составила {pct(err_example.error_rate)}, а доля ложных тревог на нормальной активности Thing_Speak — {pct(ben_example.error_rate)}. Полная таблица ошибок включает все двенадцать меток и оба порога. Доля внешних строк с точно совпадающим полным вектором в обучении варьировала от {pct(overlaps.min())} до {pct(overlaps.max())} по трём вариантам источника в среднем по seed. Такое совпадение в пространстве протокола и длительности само по себе не идентифицирует общую сетевую сессию.

## 6. Сопоставление с предыдущими исследованиями

Cantone, Marrocco и Bria сравнили четыре коллекции и также наблюдали падение качества за пределами исходного домена. Iwanowski и соавторы связали качество переноса с выбором источника и общими классами. Здесь величина и даже направление преимущества объединения показаны на попарно одинаковых тестах и при одном бюджете обучения. [Cantone et al., IEEE Access](https://doi.org/10.1109/ACCESS.2024.3472907) · [Iwanowski et al., Applied Sciences](https://doi.org/10.3390/app15158466)

Rassam и Alasaly рассматривают происхождение средств выделения потоков как фактор смешения. В данном исследовании ту же проблему можно увидеть непосредственно в таблице 2. Основной вклад работы состоит в совместной количественной проверке семантики признаков, преимуществ перед постоянным прогнозом и переноса порога на более поздний трафик. Численные значения из работ с другими признаками, метками и разбиениями нельзя использовать для рейтинга этих моделей. Полное сопоставление десяти работ включено в literature_comparison.csv. [Rassam и Alasaly, Mathematics](https://www.mdpi.com/2227-7390/14/19/3549)

## 7. Насколько далеко распространяются выводы

Три seed характеризуют изменение обучающей подвыборки и случайности алгоритма при фиксированном целевом тесте. В календарной проверке фиксированы и обучающие строки. Нулевая наблюдаемая дисперсия у детерминированной конфигурации описывает только эту конфигурацию. Внешний стенд даёт новую точку проверки переносимости. Независимое повторение другими исследователями и разброс между производственными сетями остаются открытыми.

Для ранее заданных пар моделей и источников пересчитаны 28 условных сравнений с поправкой Холма. Например, на позднем ToN при пороге 0,5 разность Macro-F1 LightGBM и LR равна {fmt(st.macro_f1_delta)}. Условный 95%-й кластерный интервал от {fmt(st.macro_f1_delta_ci_low)} до {fmt(st.macro_f1_delta_ci_high)} получен при фиксированных моделях. В календарных тестах имеется всего 13 и 21 часовой кластер. Зависимость соседних часов и предположение об обменности ограничивают вывод. Для новых сравнений приводятся описательные различия.

Отсутствуют общая обработка сырых пакетов одним инструментом, единые ключи хостов и сессий для всех наборов, независимые повторные дни с тем же составом атак и измерения в работающей IDS. Хеш-разбиение меняет долю классов и частых векторов в тесте. Три исходных набора не дают полной пятидоменной матрицы раннего плана. Сохранённые выборки позволяют повторить именно выполненные эксперименты. Исходная очистка всех 41,35 млн строк в этом запуске не повторялась.

## Итог для статьи

На зафиксированных тестах результат объединения источников зависит от целевого домена и выбора признаков. Предсказание постоянного класса в нескольких направлениях превзошло все четыре обученных алгоритма. Средний KS по общей схеме оказался слабым объяснением потери Macro-F1. Календарные испытания показали, что хороший порядок оценок риска может сочетаться с большим числом ложных тревог либо с почти полным пропуском атак. Внешняя проверка дополняет эти наблюдения результатом на отдельно собранном лабораторном трафике. Для переноса выводов на рабочие сети потребуются независимые площадки и сопоставимое определение потока.
"""
    # Figures are embedded for a standalone reading copy, and exported separately for an article.
    zero=profiles.zero/profiles.n
    suffix=suffix.replace("**Таблица 2.**",f"В исходных выборках ровно нулевая длительность встречается у {pct(zero['CICIOT2023'])} записей CIC, {pct(zero['TON_IOT'])} записей ToN и {pct(zero['UNSW_NB15'])} записей UNSW. Среди включённых строк CIC все три исходных индикатора протокола заполнены и принимают только 0 или 1. Профили каждой колонки вынесены в [приложение по длительности](supplement/duration_profiles.csv).\n\n**Таблица 2.**")
    suffix=suffix.replace("Первичное внешнее сравнение зафиксировано",f"После оценки моделей отдельно проверены единицы длительности. Для {count(unit['eligible'])} строк с положительными длительностью и скоростью величина «число пакетов / пакетов в секунду» согласуется с flow_duration в {count(unit['agree'])} случаях. Допуск — больший из 1 микросекунды и 1% длительности. Этот контроль подтверждает внутреннюю согласованность полей релиза RT. Поля скорости и числа пакетов не входили в модель. Эквивалентность выделения потоков в CIC остаётся открытой.\n\nПервичное внешнее сравнение зафиксировано")
    materials="""
## Материалы для подготовки статьи

Для раздела Methods доступны [подробный протокол и происхождение данных](supplement/methods.txt). Все парные сравнения по признакам собраны в [таблице чувствительности к длительности](tables/duration_paired.csv), по моделям — в [таблице сравнения с Majority](tables/majority_paired.csv). Для раздела о внешней проверке доступны [средние по каждой конфигурации](tables/external_summary.csv) и [разности при объединении источников](tables/external_pooling.csv). Графики в формате PDF и PNG лежат в папке figures. Для Related Work доступны [сравнение десяти работ](literature_comparison.csv) и [библиография BibTeX](supplement/references.bib).
"""
    replication=""
    if (S/"repeat_run.json").exists():
        repeat=json.loads((S/"repeat_run.json").read_text(encoding="utf-8"))
        if repeat.get("status")=="PASS":
            replication="""
## Воспроизводимость результатов

При первом сравнении запусков Random Forest по-разному округлял сумму вероятностей деревьев. На оценках около порога этого хватало для изменения решений. После первой внешней оценки порядок суммирования зафиксировали для всех задач Random Forest. Строки обучения, деревья, правило выбора порога и основной внешний результат LightGBM сохранились. Два полных запуска исправленного notebook в отдельных ядрах на одном компьютере дали одинаковые значения во всех восьми проверенных таблицах и во всех 510 файлах прогнозов основной матрицы, календарного сравнения признаков и внешнего теста. Сохранены исходная версия протокола, [запись технической поправки](protocol/computational_amendment.json) и [результаты сравнения запусков](supplement/repeat_run_comparison.csv).
"""
    txt="# "+title+"\n"+suffix+replication+materials
    (ROOT/"research_corrected.txt").write_text(txt,encoding="utf-8")
    body=markdown.markdown(txt,extensions=["tables"])
    def embed(match):
        tag=match.group(0)
        src=re.search(r'src="([^"]+)"',tag).group(1)
        caption=re.search(r'alt="([^"]*)"',tag).group(1)
        encoded=base64.b64encode((ROOT/src).read_bytes()).decode("ascii")
        return '<figure><img src="data:image/png;base64,'+encoded+'" alt="'+caption+'"><figcaption>'+caption+'</figcaption></figure>'
    body=re.sub(r'<p><img [^>]+/?></p>',embed,body)
    styles="""body{font:17px/1.65 'Segoe UI',Arial,sans-serif;color:#1b2c36;max-width:970px;margin:3rem auto;padding:0 3vw}h1,h2{line-height:1.2;text-wrap:balance}h1{font:600 clamp(30px,4.5vw,52px)/1.1 Georgia,serif;max-width:950px}h2{margin-top:2.2em;color:#184d60}p{max-width:92ch}table{border-collapse:collapse;font-size:.88rem;width:100%;line-height:1.45}th,td{padding:.45rem .65rem;border-bottom:1px solid #dce3e6;text-align:right}th{background:#ecf3f4}td:first-child,th:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafa}figure{margin:1.6rem 0 2rem;break-inside:avoid}figure img{max-width:100%;max-height:620px;display:block;margin:auto}figcaption{font-size:.9rem;color:#34505c;margin-top:.6rem}a{color:#116478} .tbl{overflow-x:auto}@media print{body{margin:0;font-size:10pt}h1{font-size:22pt}h2{break-after:avoid}table{font-size:8pt}figure img{max-height:330px}a{color:inherit}}"""
    body=body.replace("<table>","<div class='tbl'><table>").replace("</table>","</table></div>")
    (ROOT/"research.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+styles+'</style><main>'+body+'</main></html>',encoding="utf-8")


if __name__=="__main__":
    build()

