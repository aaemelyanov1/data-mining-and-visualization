"""Сборка эскиза обзора алертов SOC: схема кодировок + обоснование.

Все числа взяты из реального прогона ml/1.5_eda_soc_incidents.py
(iavd_incidents_raw_target.parquet, 240 295 карточек).
Запуск:  python sketch/make_sketch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

WIN, HGT = 11.69, 8.27
INK, GREY, LIGHT, BG = "#1f2933", "#9aa5b1", "#d7dce2", "#f5f7fa"
ACCENT, MUTED = "#b44d12", "#c9cfd6"

fig = plt.figure(figsize=(WIN, HGT), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

overflow = []


def txtw(s, size):
    """Оценка ширины строки в долях ширины рисунка.
    Коэффициент 0.58 — эмпирический для кириллицы в DejaVu Sans
    (латиница даёт ~0.50, кириллица заметно шире)."""
    return len(s) * 0.58 * size / 72.0 / WIN


def text(x, y, s, size=8.0, color=INK, weight="normal", ha="left", va="center",
         limit=None, tag="", zorder=5):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
            zorder=zorder)
    w = txtw(s, size)
    x0 = x if ha == "left" else (x - w if ha == "center" else x - w)
    bad = False
    if limit is None:
        bad = False
    elif ha == "left":
        bad = x0 + w > limit + 1e-9
    elif ha == "right":
        bad = x0 < limit - 1e-9
    else:
        bad = x0 < limit - 1e-9 or x0 + w > limit + 1e-9
    if limit is not None and bad:
        overflow.append(f"{tag or s[:34]!r}: x={x0:.3f}..{x0 + w:.3f} вне {limit:.3f}")


def wrap(s, size, width):
    cpl = max(8, int(width * WIN * 72 / (0.58 * size)))
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if len(t) <= cpl:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def box(x, y, w, h, title, sub=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        lw=1.1, edgecolor=INK, facecolor="white", zorder=2))
    text(x + 0.013, y + h - 0.024, title, 9.4, INK, "bold", limit=x + w, tag="title")
    if sub:
        text(x + 0.013, y + h - 0.048, sub, 7.1, "#52606d", limit=x + w, tag="sub")


def marker(x, y, n, color=INK):
    ax.text(x, y, n, fontsize=7.0, weight="bold", color="white", va="center",
            ha="center", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.26", facecolor=color, edgecolor="none"))


# ------------------------------------------------------------------ заголовок
text(0.045, 0.958, "ЭСКИЗ: обзор алертов SOC", 16.5, INK, "bold")
text(0.045, 0.928,
     "Что требуется от эскиза: (1) что выделяется преаттентивно, (2) как группируется, "
     "(3) чем кодируется Вердикт.", 8.0, "#52606d", limit=0.955, tag="subtitle")
text(0.045, 0.906,
     "Прямоугольники — области группировки. Кружки с номерами — ссылки на обоснование внизу.",
     7.4, GREY, limit=0.955, tag="subtitle2")

# ------------------------------------------------------------------ A. шапка
ax.add_patch(Rectangle((0.045, 0.745), 0.91, 0.132, facecolor=BG, edgecolor=LIGHT, zorder=1))
text(0.058, 0.848, "A. Объём потока за период", 9.4, INK, "bold", limit=0.94, tag="A-t")
text(0.058, 0.822, "240 295 карточек   ·   1 756 эскалаций на L2   ·   "
                   "медиана взятия в работу 24,5 ч", 7.9, INK, limit=0.94, tag="A-s")

bars = [("Критическая", 4079), ("Высокая", 45510), ("Средняя", 158159), ("Низкая", 32497)]
bx0, bw, mx = 0.185, 0.20, 158159
for i, (name, val) in enumerate(bars):
    yy = 0.812 - i * 0.019
    text(bx0 - 0.007, yy, name, 7.0, "#52606d", ha="right")
    ax.add_patch(Rectangle((bx0, yy - 0.0055), bw * val / mx, 0.011,
                           facecolor=INK if i == 0 else GREY, edgecolor="none", zorder=3))
    text(bx0 + bw * val / mx + 0.006, yy, f"{val:,}".replace(",", " "), 6.7, "#52606d")
marker(0.162, 0.803, "1")

# ------------------------------------------------------------------ B. матрица
BX, BY, BW, BH = 0.045, 0.360, 0.525, 0.360
box(BX, BY, BW, BH, "B. Матрица «Критичность → Вердикт» — ядро экрана",
    "строки упорядочены по шкале критичности; в ячейке — число карточек")
hdr = BY + BH - 0.070
cx_r, cx_f, cx_s = BX + 0.115, BX + 0.245, BX + 0.390
for cx, lab in [(cx_r, "не-FP"), (cx_f, "FP"), (cx_s, "доля FP")]:
    text(cx, hdr, lab, 7.9, INK, "bold", ha="center")
ax.plot([BX + 0.013, BX + BW - 0.013], [hdr - 0.015, hdr - 0.015], color=LIGHT, lw=1, zorder=2)

rows = [("Критическая", 3060, 1019), ("Высокая", 17449, 28061),
        ("Средняя", 45525, 112634), ("Низкая", 23573, 8924)]
for i, (name, real, fp) in enumerate(rows):
    yy = hdr - 0.043 - i * 0.047
    hot = name == "Критическая"
    if hot:
        ax.add_patch(Rectangle((BX + 0.009, yy - 0.019), BW - 0.022, 0.038,
                               facecolor="#fdf1e7", edgecolor=ACCENT, lw=1.2, zorder=2))
    text(BX + 0.017, yy, name, 8.3, INK, "bold" if hot else "normal", zorder=4)
    text(cx_r, yy, f"{real:,}".replace(",", " "), 8.3, INK, "bold" if hot else "normal",
         ha="center", zorder=4)
    text(cx_f, yy, f"{fp:,}".replace(",", " "), 8.3, MUTED, ha="center", zorder=4)
    share = fp / (real + fp)
    text(cx_s, yy, f"{share:.1%}", 8.3, ACCENT if share < 0.3 else "#52606d",
         "bold" if share < 0.3 else "normal", ha="center", zorder=4)

yy = hdr - 0.043 - 4 * 0.047
ax.plot([BX + 0.013, BX + BW - 0.013], [yy + 0.019, yy + 0.019], color=LIGHT, lw=1, zorder=2)
text(BX + 0.017, yy, "Отсутствует", 7.5, GREY)
text(cx_r, yy, "13", 7.5, GREY, ha="center")
text(cx_f, yy, "37", 7.5, GREY, ha="center")
text(cx_s, yy, "74,0 %", 7.5, GREY, ha="center")
text(BX + 0.017, yy - 0.024,
     "вынесено за шкалу: всего 50 записей (0,02 %) — уровня нет", 7.0, GREY, limit=BX + BW)
marker(BX + BW - 0.028, hdr - 0.090, "2")
marker(cx_f, hdr - 0.211, "3")

# ------------------------------------------------------------------ C. внимание
CX, CY, CW, CH = 0.60, 0.575, 0.355, 0.145
box(CX, CY, CW, CH, "C. «Требуют внимания»", "сортировка не по критичности, а по P(реальный) x цена ошибки")
ax.add_patch(Rectangle((CX + 0.013, CY + 0.010), CW - 0.026, 0.070,
                       facecolor="#fdf1e7", edgecolor=ACCENT, lw=1.2, zorder=2))
text(CX + 0.021, CY + 0.067, "ПРЕАТТЕНТИВНАЯ ЦЕЛЬ", 7.0, ACCENT, "bold", limit=CX + CW)
text(CX + 0.021, CY + 0.047, "Критическая + не-FP = 3 060 карточек, чистота 75,0 %",
     8.0, INK, "bold", limit=CX + CW)
ax.text(CX + 0.021, CY + 0.037,
        wrap("размер точки пропорционален объёму; насыщенность одна на все "
             "true-положительные", 6.9, CW - 0.048),
        fontsize=6.9, color="#52606d", va="top", zorder=5)
marker(CX + CW - 0.030, CY + 0.113, "4", ACCENT)

# ------------------------------------------------------------------ D. время
DX, DY, DW, DH = 0.60, 0.360, 0.355, 0.190
box(DX, DY, DW, DH, "D. Срез по времени — добирает контекст",
    "доля карточек по часам суток: пик 8:00 (19 118), минимум 0:00 (3 591)")

HOURLY = [3591, 3997, 5410, 4649, 5531, 12380, 18240, 18756, 19118, 16753,
          11500, 16210, 15451, 13672, 12620, 8216, 7501, 6294, 7074, 8515,
          8103, 6401, 5671, 4642]
hours = list(range(24))

# Сверяем встроенные числа с данными, если parquet доступен: в эскизе не должно
# быть ни одной цифры, взятой «на глаз».
try:
    import pandas as _pd
    from pathlib import Path as _Path

    _p = _Path(__file__).resolve().parents[1] / "data" / "iavd_incidents_raw_target.parquet"
    if _p.exists():
        _d = _pd.to_datetime(_pd.read_parquet(_p, columns=["Создан"])["Создан"],
                             errors="coerce")
        _live = _d.dt.hour.value_counts().sort_index()
        _live = [int(_live.get(h_, 0)) for h_ in hours]
        if _live != HOURLY:
            print("ВНИМАНИЕ: часовые значения в скрипте разошлись с parquet:")
            print("  скрипт:", HOURLY)
            print("  parquet:", _live)
        else:
            print("часовые значения сверены с parquet: совпадают")
except Exception as _e:
    print(f"parquet недоступен ({type(_e).__name__}), часовые значения из скрипта")

gx, gw, gy, gh = DX + 0.016, DW - 0.032, DY + 0.058, 0.068
vmax = max(HOURLY)
for i, (h_, v_) in enumerate(zip(hours, HOURLY)):
    slot = gw / len(hours)
    px = gx + i * slot
    ax.add_patch(Rectangle((px, gy), slot - 0.0012, gh * v_ / vmax,
                           facecolor=ACCENT if h_ == 8 else GREY, edgecolor="none", zorder=3))
    if h_ % 4 == 0:
        text(px + slot / 2 - 0.0006, gy - 0.013, str(h_), 6.0, "#52606d", ha="center")
text(DX + 0.016, gy - 0.030, "часы суток, 0 = полночь; подпись каждые 4 ч", 6.6,
     "#52606d", limit=DX + DW - 0.026)
marker(DX + DW - 0.030, gy - 0.030, "5")

# ------------------------------------------------------------------ обоснование
ax.plot([0.045, 0.955], [0.330, 0.330], color=LIGHT, lw=1)
text(0.045, 0.308, "ОБОСНОВАНИЕ КОДИРОВОК", 8.4, INK, "bold")

legend = [
    ("1", "Количество алертов — длина полосы на общей шкале. По Кливленду–МакГиллу "
          "позиция и длина — самые точные каналы, интенсивность цвета — худший."),
    ("2", "Критичность — порядковая шкала, кодируется позицией строки, а не пятью "
          "оттенками: оттенок различим плохо и зависит от монитора."),
    ("3", "Вердикт: 7 значений сворачиваются до бинарных FP / не-FP — это операционное "
          "решение. FP = 62,7 % (большинство) приглушён."),
    ("4", "Преаттентивная цель — единственный элемент, который обязан находиться без "
          "чтения подписей. Всё остальное вторичные каналы."),
    ("5", "Время задаёт ритм SOC: суточный цикл и доля выходных (12,8 %) видятся "
          "преаттентивно. Это контекст, а не сигнал тревоги."),
]

# Раскладка легенды в три колонки. Высота каждого пункта измеряется по факту
# отрисовки: у пунктов разное число строк, фиксированный шаг их накладывал.
COL_X = [0.045, 0.345, 0.645]
TXT_W = 0.256
LEG_Y = 0.286
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def place(n, s, x, y):
    body = wrap(s, 6.8, TXT_W)
    if txtw(max(body.split("\n"), key=len), 6.8) > TXT_W + 0.003:
        overflow.append(f"легенда {n} шире колонки")
    marker(x + 0.008, y, n, ACCENT if n == "4" else INK)
    t = ax.text(x + 0.022, y + 0.012, body, fontsize=6.8, color="#33414e",
                va="top", zorder=6)
    fig.canvas.draw()
    return t.get_window_extent(renderer=renderer).height / fig.bbox.height


row0 = [place(*legend[i], COL_X[i], LEG_Y) for i in range(3)]
y1 = LEG_Y - max(row0) - 0.014
row1 = [place(*legend[3 + i], COL_X[i], y1) for i in range(2)]
top_legend = y1 - max(row1) - 0.014

text(0.045, top_legend, "ГРУППИРОВКА (законы гештальта)", 7.8, INK, "bold")
ax.text(0.045, top_legend - 0.016,
        wrap("Близость — карточки одного узла рядом. Сходство — одинаковый вердикт даёт "
             "одинаковый вид. Общая область — каждая панель замкнута рамкой. "
             "Приглушение вместо перечёркивания.", 7.0, 0.910),
        fontsize=7.0, color="#52606d", va="top", zorder=6)
ax.text(0.045, top_legend - 0.064,
        wrap("ЧЕГО НЕТ: оттенков «красный/зелёный» (≈8 % мужчин не различают, а это рабочий "
             "инструмент SOC), 3D, датчиков, сетки. Данные: 240 295 карточек, "
             "контрольные числа 1.1, 2.2, 4.4, 5.9.", 7.0, 0.910),
        fontsize=7.0, color=ACCENT, va="top", zorder=6)

if top_legend - 0.110 < 0.018:
    overflow.append(f"нижний блок уходит ниже полотна: {top_legend - 0.110:.3f}")

fig.savefig("sketch/1_eskiz_obzora_alertov.png", dpi=170, facecolor="white")
print("saved: sketch/1_eskiz_obzora_alertov.png")
if overflow:
    print("ПРЕДУПРЕЖДЕНИЕ — текст выходит за границы:")
    for o in overflow:
        print("  -", o)
else:
    print("проверка вёрстки: переполнений нет")