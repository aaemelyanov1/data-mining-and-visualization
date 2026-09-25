# %% [markdown]
# # Исследовательский анализ данных SOC (ML-трек)
# **Датасет:** `iavd_incidents_raw_target.parquet`
# **Цель:** Выполнить разведочный анализ (EDA) по 7 блокам, проверить контрольные числа, визуализировать ключевые метрики и сформулировать инсайты.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")

# Загрузка данных
df = pd.read_parquet("../data/iavd_incidents_raw_target.parquet")
print(f"Датасет успешно загружен. Размер: {df.shape[0]:,} строк и {df.shape[1]} столбцов.")

# %% [markdown]
# ## 0. Общая информация о датасете
# Перед началом анализа необходимо посмотреть на структуру данных, первые строки и базовые статистики, чтобы понять природу признаков.

# %%
print("Первые 5 строк датасета ")
print(df.head().to_string())

print("\n Общая информация о типах данных и пропусках ")
df.info()

print("\n Статистики по числовым признакам ")
print(df.describe().to_string())

# %% [markdown]
# ## Блок 1. Знакомство со структурой данных
# Анализ размерности, типов данных, пропусков и структуры мультизначных полей.

# %%
# 1.1 Сколько строк и столбцов в датасете
print(f"1.1 Размер датасета: {df.shape[0]:,} строк и {df.shape[1]} столбцов.")

# 1.2 Сколько полей строкового, числового и datetime-типа
print("1.2 Распределение типов данных:")
dt_counts = df.dtypes.value_counts()
for dtype, count in dt_counts.items():
    print(f"    - {dtype}: {count} полей")

# 1.3 Какие 3 поля заполнены хуже всего
missing = df.isna().mean().sort_values(ascending=False)
print("1.3 Топ-3 поля с наибольшим процентом пропусков:")
for col, pct in missing.head(3).items():
    print(f"    - {col}: {pct:.1%}")

# 1.4 Сколько полей заполнены полностью (без NaN)
print(f"1.4 Количество полностью заполненных полей (без NaN): {(missing == 0).sum()}")

# 1.5 Сколько уникальных значений у Вердикт
print(f"1.5 Уникальных значений в 'Вердикт': {df['Вердикт'].nunique()}")

# 1.6 Сколько уникальных значений у Критичность
print(f"1.6 Уникальных значений в 'Критичность': {df['Критичность'].nunique()}")

# 1.7 Сколько карточек содержат список пар в Значение Инцидента (группировка)
col_val = 'Значение Инцидента (группировка)'
has_pipe = df[col_val].dropna().str.contains(r'\|', regex=True).mean()
print(f"1.7 Доля непустых значений с разделителем '|': {has_pipe:.1%}")

# 1.8 Максимальное число элементов списка Значение Инцидента у одной карточки
max_elements = df[col_val].dropna().str.split('~~||~~').str.len().max()
print(f"1.8 Максимальное число элементов в списке: {max_elements:,}")

# 1.9 Есть ли дубликаты EntityId
print(f"1.9 Дубликатов EntityId: {df['EntityId'].duplicated().sum()}")

# 1.10 Сколько строк даст разворот Значение Инцидента по элементам
total_elements = df[col_val].dropna().str.split('~~||~~').str.len().sum()
print(f"1.10 Строк после explode: примерно {total_elements/1e6:.1f} млн")

# 1.11 Какое поле — полный аналог Вердикт в числовом виде
print(f"1.11 Числовой аналог 'Вердикт' (поле 'verdict'): тип {df['verdict'].dtype}")

# График 1: Топ-10 полей с пропусками
plt.figure(figsize=(10, 6))
top_missing = missing.head(10) * 100
sns.barplot(x=top_missing.values, y=top_missing.index, color="salmon")
plt.title("Топ-10 полей с наибольшим процентом пропусков", fontsize=14, fontweight='bold')
plt.xlabel("Процент пропусков (%)")
plt.ylabel("Название поля")
for i, v in enumerate(top_missing.values):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 2. Жизненный цикл карточки инцидента
# Анализ вердиктов, фаз реагирования, статусов и эскалаций.

# %%
# 2.1 Перечислите значения Вердикт и частоты
verdicts = df['Вердикт'].value_counts(dropna=False)
print("2.1 Распределение вердиктов:")
for v, count in verdicts.items():
    pct = count / len(df) * 100
    print(f"    - {v}: {count:,} ({pct:.1f}%)")

# 2.2 Доля ложных срабатываний (FP)
fp_share = (df['Вердикт'] == 'False Positive').mean()
print(f"\n2.2 Доля ложных срабатываний (FP): {fp_share:.1%}")

# 2.3 Какие значения принимает Фаза реагирования
phases = df['Фаза реагирования'].value_counts(dropna=False)
print("\n2.3 Фазы реагирования:")
for p, count in phases.items():
    print(f"    - {p}: {count:,}")

# 2.4 и 2.5 Что означают фазы вида «Закрыт~~||~~Проанализирован» и самая частая комбинированная фаза
combined_phases = phases[phases.index.astype(str).str.contains('~~\|\|~~', na=False)]
if not combined_phases.empty:
    print(f"\n2.4-2.5 Комбинированные фазы означают пройденные этапы. Самая частая: '{combined_phases.idxmax()}' ({combined_phases.max():,} карточек).")

# 2.6 Какие статусы карточки в источнике (SIEM)
print("\n2.6 Статусы инцидента из SIEM:")
for s, count in df['Статус инцидента из SIEM'].value_counts(dropna=False).items():
    print(f"    - {s}: {count:,}")

# 2.7 Сколько карточек помечено как спам
spam_count = (df['Спам'] == 'True').sum()
print(f"\n2.7 Карточек, помеченных как спам: {spam_count:,}")

# 2.8 Сколько карточек эскалировано на L2 (ИСПРАВЛЕНО: добавлен вопросительный знак)
l2_col = 'Эскалировался на L2?'
l2_count = (df[l2_col] == 'True').sum()
print(f"2.8 Карточек, эскалированных на L2: {l2_count:,}")

# 2.9 В скольких карточках выполнена проверка на FP
fp_check_col = 'Ложное срабатывание'
print(f"\n2.9 Статус проверки на FP ({fp_check_col}):")
for s, count in df[fp_check_col].value_counts(dropna=False).items():
    print(f"    - {s}: {count:,}")

# 2.10 Согласуются ли Вердикт='False Positive' и флаг Ложное срабатывание='True'
ct = pd.crosstab(df['Вердикт'], df[fp_check_col], dropna=False)
print("\n2.10 Таблица сопряженности (Вердикт vs Ложное срабатывание):")
print(ct.to_string())
print("Инсайт: поля не являются полными дубликатами, флаг выставляется чаще, чем вердикт FP.")

# 2.11 Сколько карточек «Инцидент был изменен»
changed_col = 'Инцидент был изменен'
changed_count = (df[changed_col] == 'True').sum()
print(f"\n2.11 Инцидентов, которые были изменены: {changed_count:,}")

# 2.12 Каков вердикт у карточек группы ГИБ
gib_fp = df[df['Группа устранения'] == 'ГИБ']['Вердикт'].value_counts(normalize=True)
print("\n2.12 Доля вердиктов в группе ГИБ:")
for v, pct in gib_fp.items():
    print(f"    - {v}: {pct:.1%}")
print("Инсайт: доля FP в ГИБ всего 2.3%, группа работает с реальными инцидентами.")

# График 2: Распределение вердиктов
plt.figure(figsize=(10, 6))
verdicts_plot = verdicts.head(6)
sns.barplot(x=verdicts_plot.values, y=verdicts_plot.index, color="teal")
plt.title("Распределение вердиктов инцидентов", fontsize=14, fontweight='bold')
plt.xlabel("Количество инцидентов")
plt.ylabel("Вердикт")
for i, v in enumerate(verdicts_plot.values):
    pct = v / len(df) * 100
    plt.text(v + 500, i, f"{v:,} ({pct:.1f}%)", va='center')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 3. Кто и откуда: источники, аналитики, пользователи
# Анализ пользователей, хостов, аналитиков и групп устранения.

# %%
# 3.1 Сколько уникальных учётных записей в suser
print(f"3.1 Уникальных учетных записей (suser): {df['suser'].nunique():,}")

# 3.2 Сколько уникальных пользователей-жертв в Полное имя
print(f"3.2 Уникальных пользователей-жертв (Полное имя): {df['Полное имя'].nunique():,}")

# 3.3 Кто из пользователей-жертв фигурирует чаще всего
print("\n3.3 Топ-5 пользователей-жертв:")
for name, count in df['Полное имя'].value_counts().head(5).items():
    print(f"    - {name}: {count:,}")

# 3.4 Топ-3 учётные записи suser
print("\n3.4 Топ-3 учетные записи (suser):")
for user, count in df['suser'].value_counts().head(3).items():
    print(f"    - {user}: {count:,}")
print("Инсайт: в топе служебные учетные записи (root, ankey, administrator).")

# 3.5 Сколько уникальных хостов-целей
print(f"\n3.5 Уникальных хостов-целей: {df['Hostname Цели'].nunique():,}")

# 3.6 Топ-3 хоста-цели
print("\n3.6 Топ-3 хоста-цели:")
for host, count in df['Hostname Цели'].value_counts().head(3).items():
    print(f"    - {host}: {count:,}")
print("Инсайт: топ хостов - серверы конфигурации (SCCM) и дефолтные адреса.")

# 3.7 Сколько аналитиков ведут карточки (Ответственный)
print(f"\n3.7 Уникальных аналитиков (Ответственный): {df['Ответственный'].nunique()}")

# 3.8 Как распределена нагрузка между аналитиками
analyst_load = df['Ответственный'].value_counts()
print(f"3.8 Нагрузка аналитиков: максимум {analyst_load.max():,}, минимум {analyst_load.min():,}. Разброс более чем в 10 раз.")

# 3.9 Как распределены карточки по группам устранения
print("\n3.9 Распределение по группам устранения:")
for group, count in df['Группа устранения'].value_counts().items():
    pct = count / len(df) * 100
    print(f"    - {group}: {count:,} ({pct:.1f}%)")

# 3.10 Какие значения у Конвейер SIEM
print("\n3.10 Топ-5 конвейеров SIEM:")
for conv, count in df['Конвейер SIEM'].value_counts(dropna=False).head(5).items():
    print(f"    - {conv}: {count:,}")

# График 3: Топ-10 аналитиков по нагрузке
plt.figure(figsize=(10, 6))
top_analysts = analyst_load.head(10)
sns.barplot(x=top_analysts.values, y=top_analysts.index, color="coral")
plt.title("Топ-10 аналитиков по количеству обработанных карточек", fontsize=14, fontweight='bold')
plt.xlabel("Количество карточек")
plt.ylabel("Аналитик")
for i, v in enumerate(top_analysts.values):
    plt.text(v + 200, i, f"{v:,}", va='center')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 4. Время: профили активности, MTTR, SLA
# Анализ временных профилей, времени решения и SLA.

# %%
# Преобразование времени
df['Создан_dt'] = pd.to_datetime(df['Создан'], errors='coerce')

# 4.1 Каков период представления данных
print(f"4.1 Период данных: с {df['Создан_dt'].min()} по {df['Создан_dt'].max()}")

# 4.2 В каком месяце больше всего карточек
months = df['Создан_dt'].dt.to_period('M').value_counts()
top_month = months.idxmax()
print(f"\n4.2 Месяц с максимумом карточек: {top_month} ({months.max():,})")

# 4.3 В каком месяце меньше всего (среди полных месяцев)
min_month = months.idxmin()
print(f"4.3 Месяц с минимумом карточек: {min_month} ({months.min():,})")

# 4.4 Час пика и час минимума
hours = df['Создан_dt'].dt.hour.value_counts().sort_index()
print(f"\n4.4 Час пика: {hours.idxmax()}:00 ({hours.max():,} инцидентов)")
print(f"    Час минимума: {hours.idxmin()}:00 ({hours.min():,} инцидентов)")

# 4.5 Доля выходных
weekends = df['Создан_dt'].dt.dayofweek.isin([5, 6]).mean()
print(f"\n4.5 Доля инцидентов в выходные: {weekends:.1%}")

# 4.6 Медиана Время решения (часов)
time_col = 'Время решения'
median_res = df[time_col].median() / 3600
print(f"\n4.6 Медиана времени решения: {median_res:.1f} ч")

# 4.7 p90 Время решения
p90_res = df[time_col].quantile(0.9) / 3600
print(f"4.7 90-й перцентиль (p90) времени решения: {p90_res:.1f} ч (длинный хвост)")

# 4.8 Доля карточек с решением дольше 30 дней
over_30_days = (df[time_col] > 30*86400).mean()
print(f"4.8 Доля карточек с решением дольше 30 дней: {over_30_days:.1%}")

# 4.9 Медиана Время решения по L1/L2/ГИБ
print("\n4.9 Медиана времени решения по группам (в часах):")
for group, val in (df.groupby('Группа устранения')[time_col].median() / 3600).items():
    print(f"    - {group}: {val:,.1f} ч")

# 4.10 Медиана Период взятия в работу (часов)
median_take = df['Период взятия в работу'].median() / 3600
print(f"\n4.10 Медиана периода взятия в работу: {median_take:.1f} ч")

# 4.11 Ловушка. Поле задержка перед автоназначением
ts_median = df['задержка перед автоназначением'].median()
print(f"\n4.11 Ловушка! 'задержка перед автоназначением' - это unix-timestamp: {pd.to_datetime(ts_median, unit='s')}")

# 4.12 Ловушка. Единицы SLA по устранению Атаки
sla_median = df['SLA по устранению Атаки'].median()
print(f"4.12 Ловушка! SLA по устранению Атаки хранится в мс: {sla_median:,.0f} мс = {sla_median / 3.6e6:.1f} ч")

# График 4: Распределение по часам суток
plt.figure(figsize=(12, 5))
sns.barplot(x=hours.index, y=hours.values, color="skyblue")
plt.title("Распределение создания инцидентов по часам суток", fontsize=14, fontweight='bold')
plt.xlabel("Час")
plt.ylabel("Количество инцидентов")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 5. Содержательный анализ: процессы, теги, пары ключ|значение
# Анализ процессов, тегов, пар ключ-значение и долей FP.

# %%
# 5.1 Топ-8 процессов processName
print("5.1 Топ-8 процессов (processName):")
for proc, count in df['processName'].value_counts().head(8).items():
    print(f"    - {proc}: {count:,}")

# 5.2 Доля карточек, где в Команда встречается powershell
ps_share = df['Команда'].dropna().str.contains('powershell', case=False).mean()
print(f"\n5.2 Доля карточек с 'powershell' в Команде: {ps_share:.1%}")

# 5.3 Топ-5 Пиктограмма IRP
print("\n5.3 Топ-5 Пиктограмма IRP:")
for pic, count in df['Пиктограмма IRP'].value_counts().head(5).items():
    print(f"    - {pic}: {count:,}")

# 5.4 Топ-3 Теги
print("\n5.4 Топ-3 Теги:")
for tag, count in df['Теги'].value_counts().head(3).items():
    print(f"    - {tag}: {count:,}")

# 5.5 Доля карточек с комбинированными тегами (MITRE-цепочки)
comb_tags = df['Теги'].dropna().str.contains('~~||~~').mean()
print(f"\n5.5 Доля карточек с комбинированными тегами: {comb_tags:.1%}")

# 5.6, 5.7, 5.8 Разбор пар Значение Инцидента (группировка)
col_pairs = 'Значение Инцидента (группировка)'
pairs_exploded = df[col_pairs].dropna().str.split('~~||~~').explode()
pairs_split = pairs_exploded.str.split('|', n=1, expand=True)
pairs_split.columns = ['key', 'value']

print("\n5.6 Роли (значения в парах):")
for val, count in pairs_split['value'].value_counts().head(3).items():
    print(f"    - {val}: {count:,}")

print("\n5.7 Топ-3 узла-ключа:")
for key, count in pairs_split['key'].value_counts().head(3).items():
    print(f"    - {key}: {count:,}")

print(f"\n5.8 Всего пар: {len(pairs_split):,}, уникальных ключей: {pairs_split['key'].nunique():,}")

# 5.9 Какова доля FP по критичности
print("\n5.9 Доля FP по критичности:")
fp_by_crit = df.groupby('Критичность')['Вердикт'].apply(lambda x: (x=='False Positive').mean())
for crit, pct in fp_by_crit.items():
    print(f"    - {crit}: {pct:.1%}")
print("Инсайт: чем выше критичность, тем чаще алерт настоящий.")

# 5.10 Доля FP по линиям L1/L2/ГИБ
print("\n5.10 Доля FP по группам устранения:")
fp_by_group = df.groupby('Группа устранения')['Вердикт'].apply(lambda x: (x=='False Positive').mean())
for group, pct in fp_by_group.items():
    print(f"    - {group}: {pct:.1%}")

# 5.11 Сколько карточек создавал каждый конвейер srv-conv1
conv1_count = df[df['Конвейер SIEM'].str.contains('srv-conv1', na=False)].shape[0]
print(f"\n5.11 Карточек от конвейера srv-conv1: {conv1_count:,}")

# 5.12 Среднее число элементов в Значение Инцидента на карточку (медиана)
lens = df[col_pairs].dropna().str.split('~~||~~').str.len()
print(f"\n5.12 Статистика элементов в 'Значение Инцидента': среднее ≈ {lens.mean():.0f}, медиана = {lens.median():.0f}")

# График 5: Доля FP по критичности
plt.figure(figsize=(10, 6))
sns.barplot(x=fp_by_crit.values * 100, y=fp_by_crit.index, color="lightgreen")
plt.title("Доля ложных срабатываний (FP) по уровню критичности", fontsize=14, fontweight='bold')
plt.xlabel("Доля FP (%)")
plt.ylabel("Критичность")
for i, v in enumerate(fp_by_crit.values * 100):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 6. Качество данных
# Анализ дубликатов, пропусков, выбросов и единиц измерения.

# %%
# 6.1 Есть ли дубликаты карточек (EntityId)
print(f"6.1 Дубликатов EntityId: {df['EntityId'].duplicated().sum()}")

# 6.2 Сколько карточек с пустым Комментарий
empty_comments = df['Комментарий'].fillna("").eq("").sum()
print(f"6.2 Пустых комментариев: {empty_comments:,}")

# 6.3 Поле Ошибки заполнено
errors_filled = df['Ошибки'].notna().sum()
print(f"6.3 Заполнено поле 'Ошибки': {errors_filled}")

# 6.4 Сколько NaN у Эскалировался на L2
l2_nan = df['Эскалировался на L2?'].isna().sum()
print(f"6.4 NaN у 'Эскалировался на L2': {l2_nan:,}")

# 6.5 NaN vs «Не указан» у Вердикт
verdict_nan = df['Вердикт'].isna().sum()
verdict_unspecified = (df['Вердикт'] == 'Не указан').sum()
print(f"6.5 'Вердикт' NaN: {verdict_nan} | 'Не указан': {verdict_unspecified:,}")
print("Инсайт: пропуск и осознанное значение 'Не указан' - это разные состояния.")

# 6.6 Найдите значения в разных регистрах
print("\n6.6 Примеры разного регистра в suser (топ-3):")
for user, count in df['suser'].str.lower().value_counts().head(3).items():
    print(f"    - {user}: {count:,}")

# 6.7 Максимум Время решения в часах и правдоподобно ли
max_hours = df['Время решения'].max() / 3600
print(f"\n6.7 Максимальное 'Время решения': {max_hours:,.1f} ч (технический выброс, неправдоподобно)")

# 6.8 Единицы SLA по устранению Атаки (медиана 3 600 000)
sla_median = df['SLA по устранению Атаки'].median()
print(f"6.8 SLA медиана: {sla_median:,.0f} мс = {sla_median / 3.6e6:.1f} ч")

# График 6: Распределение времени решения (логарифмическая шкала)
plt.figure(figsize=(10, 6))
# Убираем нули и явные технические нули для логарифмической шкалы
time_to_plot = df[time_col][df[time_col] > 0] / 3600
sns.histplot(time_to_plot, bins=50, log_scale=True, color="purple")
plt.title("Распределение времени решения инцидентов (логарифмическая шкала)", fontsize=14, fontweight='bold')
plt.xlabel("Время решения (часы, log scale)")
plt.ylabel("Количество инцидентов")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Блок 7. Мост к ML
# Подготовка целевой переменной, анализ утечек, кардинальности и выбор признаков.

# %%
# 7.1 Насколько сбалансирован целевой класс (FP)
df['y'] = (df['Вердикт'] == 'False Positive').astype(int)
fp_balance = df['y'].mean()
print(f"7.1 Баланс классов: {fp_balance:.3f} (FP) / {1 - fp_balance:.3f} (остальные)")

# 7.2 Альтернативный кандидат в таргеты
alt_target = (df['Ложное срабатывание'] == 'True').sum()
print(f"\n7.2 Альтернативный кандидат в таргеты: флаг 'Ложное срабатывание' (True: {alt_target:,})")

# 7.3 Поля с высокой кардинальностью
print("\n7.3 Поля с высокой кардинальностью (Топ-4):")
for col, count in df.nunique().sort_values(ascending=False).head(4).items():
    print(f"    - {col}: {count:,}")

# 7.4 Числовые признаки «из коробки»
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n7.4 Числовые признаки ({len(num_cols)}): {num_cols}")

# 7.5 Строк после explode Значение Инцидента
print(f"\n7.5 Строк после explode 'Значение Инцидента': примерно {total_elements/1e6:.1f} млн")

# 7.6 Признак — почти прямой таргет (утечка)
print("\n7.6 Потенциальные утечки таргета:")
print("    - флаг 'Ложное срабатывание'")
print("    - текст 'Решение'")

# 7.7 Как использовать поля-флаги
bool_cols = [c for c in df.columns if df[c].isin(['True', 'False']).any()]
print(f"\n7.7 Найдено булевых признаков (True/False): {len(bool_cols)}. Пропуски (NaN) тоже несут сигнал.")

# 7.8 Поля для TF-IDF/эмбеддингов
print("\n7.8 Поля для TF-IDF/эмбеддингов:")
print("    - 'Алерт', 'Описание инцидента', 'Команда', 'CMDLINE'")

# 7.9 Как стратифицировать выборку
print(f"\n7.9 Стратификация выборки: по целевой переменной 'y' (сохранит долю FP {fp_balance:.1%}).")

# 7.10 Поля-идентификаторы для исключения
print("\n7.10 Поля-идентификаторы для исключения из модели:")
print("    - EntityId")
print("    - verdict (дубликат таргета)")

# График 7: Баланс классов
plt.figure(figsize=(6, 6))
class_counts = df['y'].value_counts()
labels = ['False Positive (1)', 'Остальные (0)']
colors = ['#ff9999', '#66b3ff']
plt.pie(class_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
plt.title("Баланс классов для ML-модели", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
