
'''
🚀 3Filoff EDA Toolkit

viz.py v.0.4 • Вспомогательные функции для визуализации и EDA

оттестирована 2025 26 10

upd : 
   • plot_phik_correlation - автомаштабировании с подстройкой под DPI

'''

import re
import os
import shap
from pathlib import Path
from itertools import combinations
from typing import Optional, Set, Union, List, Dict, DefaultDict, Literal, Callable, Any, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, skew, kurtosis, ttest_ind, mannwhitneyu
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

# Вспомогательные инструменты Jupyter
from IPython.display import display
from pandas.io.formats.style import Styler



# Определяем корень датасетов - с fallback на '../datasets' (относительно utils/)
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./../datasets"))

# Глобальные справочники - инициализируем пустыми, но допускаем переопределение
CSV_PATHS = {}

# Справочник описаний датасетов по их имени
# Используется для генерации отчётов и контекстной документации
DATASET_DESCRIPTIONS = {}

# Единый справочник описаний всех колонок в проекте
# Используется для генерации отчётов и контекстной документации
COLUMN_DESCRIPTIONS = {}




# •••••••••• ФУНКЦИИ 3Filoff ••••••••••


# ## **Вспомогательные функции**

# Напишем функции для выполнения рутинных операций, чтобы:
# - избежать загромождения ячеек юпитер ноутбука повторяющимся кодом
# - сделать код более читаемым и структурированным
# - упростить поддержку и обновление кода в будущем
# 
# Эти функции позволят нам быстро и удобно выполнять часто используемые операции с данными, не дублируя код в разных частях ноутбука








#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


"""
Модуль визуализации для EDA: красивые таблицы в ноутбуке + экспорт в Markdown.

Основная функция - `display_table` - поддерживает три режима:
- notebook: стилизованная HTML-таблица для анализа,
- markdown: компактная pipe-таблица для копирования в отчёты,
- both: одновременно и то, и другое.

Переключение через функцию:
    from utils.viz import set_output_mode
    set_output_mode("markdown")
"""


# Глобальный режим вывода: "notebook", "markdown", или "both"
EDA_OUTPUT_MODE: str = "notebook"



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def _print_compact_markdown(df: pd.DataFrame) -> None:
    """
    Печатает компактную Markdown-таблицу без кавычек, с настоящими переносами строк.
    NaN и пустые значения отображаются как пустые ячейки.
    """
    if df.empty:
        print("⚠️ Пустой датафрейм")
        return

    # Сначала заменяем NaN на пустую строку, потом приводим к str
    df_str = df.fillna("").astype(str)
    headers = "|".join(df_str.columns)
    separator = "|".join(["---"] * len(df_str.columns))
    rows = ["|".join(row) for row in df_str.values]
    lines = [f"|{headers}|", f"|{separator}|"] + [f"|{row}|" for row in rows]
    print("\n".join(lines))


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def set_global_styles(
    dpi: int = 150,
    palette: str = "rocket",
    grid_color: str = "#E0E0E0",
    font_family: str = "Sans",
    apply_pandas_display: bool = True,
    apply_seaborn_style: bool = True,
    apply_matplotlib_rc: bool = True,
) -> None:
    """
    Настраивает глобальные параметры визуализации и отображения данных.
    
    Описание:
        Единая точка конфигурации для pandas, seaborn и matplotlib.
        Особенности:
        - Лёгкая сетка и округлые графики,
        - Читаемые шрифты (Arial),
        - Палитра 'rocket' (градиент от тёмного к яркому),
        - Минималистичные подписи без перегрузки.

    Параметры:
        dpi                  : int  • разрешение графиков (по умолчанию 150)
        palette              : str  • палитра seaborn для графиков (по умолчанию 'tab10')
        grid_color           : str  • цвет сетки (по умолчанию '#c5d1e0' - серо-голубой)
        font_family          : str  • семейство шрифтов (по умолчанию 'DejaVu Sans')
        apply_pandas_display : bool • применять ли настройки pandas (по умолчанию True)
        apply_seaborn_style  : bool • применять ли стиль seaborn (по умолчанию True)
        apply_matplotlib_rc  : bool • применять ли rcParams matplotlib (по умолчанию True)

    Возвращаемое значение:
        None - применяет настройки глобально
    """
    # 1. Pandas: читаемость
    if apply_pandas_display:
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.expand_frame_repr', False)

    # 2. Seaborn: стиль
    if apply_seaborn_style:
        sns.set_style("whitegrid", {
            'axes.facecolor': 'white',
            'grid.color': grid_color,
            'grid.linewidth': 0.7,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#555555',
            'ytick.color': '#555555',
            'font.family': font_family
        })
        sns.set_palette(palette)

    # 3. Matplotlib: внешний вид
    if apply_matplotlib_rc:
        plt.rcParams.update({
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,

            'font.family': font_family,
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'normal',
            'axes.labelpad': 4.0,
            'axes.titlepad': 6.0,

            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'xtick.color': '#555555',
            'ytick.color': '#555555',

            'lines.linewidth': 1.4,
            'lines.markersize': 4,

            'patch.edgecolor': 'white',
            'patch.linewidth': 0.8,

            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    # 💎 Вывод настроек
    print(f"🎨 Стили обновлены: | DPI={dpi} | Palette='{palette}' | Grid='{grid_color}'")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def set_output_mode(mode: str, verbose: bool = True) -> None:
    """
    Устанавливает глобальный режим вывода для display_table.

    Параметры
   -------
    mode : str
        Один из: "notebook", "markdown", "both".
    verbose : bool, по умолчанию True
        Если True - выводит понятное подтверждение с эмодзи.
    """
    global EDA_OUTPUT_MODE
    if mode not in ("notebook", "markdown", "both"):
        raise ValueError("mode must be one of: 'notebook', 'markdown', 'both'")
    EDA_OUTPUT_MODE = mode
    if verbose:
        mode_labels = {
            "notebook": "👁️ notebook • стилизованные таблицы для анализа",
            "markdown": "📋 markdown • чистый формат для копирования",
            "both": "👁️ + 📋 both • и стилизованные, и копируемо"
        }
        print(f"🚀 EDA-среда инициализирована: {mode_labels[mode]}")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def display_table(
    df: pd.DataFrame,
    rows: Optional[int] = 5,
    float_precision: int = 3,
    max_header_length: int = 20,
    styler_func: Optional[Callable[[Styler], Styler]] = None,
    as_markdown: Optional[bool] = None,
    mode: Optional[str] = None
) -> None:
    """
    Отображает датафрейм в одном из трёх режимов: для анализа, для копирования или для обоих целей сразу.

    Цель:
        Предоставить единый интерфейс для просмотра данных, который:
        - в интерактивной среде (Jupyter) показывает профессионально стилизованную таблицу,
        - при необходимости генерирует чистую Markdown-таблицу, готовую к вставке в отчёты,
        - позволяет совмещать оба подхода для максимального удобства.

    Режимы работы:
        - "notebook" (по умолчанию):
            Ограничивает вывод, обрезает длинные заголовки, применяет цветовую схему,
            выравнивание и кастомную стилизацию. Идеален для разведочного анализа.
        - "markdown":
            Печатает компактную pipe-таблицу с полными названиями колонок и форматированием чисел.
            Результат можно сразу копировать в GitHub, Notion, Obsidian и др.
        - "both":
            Сначала отображает стилизованную таблицу, затем - Markdown-версию с разделителем.
            Удобен при демонстрации результатов и одновременном предоставлении копируемого формата.

    Параметры:
        df : pd.DataFrame
            Датафрейм для отображения.
        rows : int or None, по умолчанию 5
            Максимальное количество строк для вывода. Если None - выводятся все строки.
        float_precision : int, по умолчанию 3
            Количество знаков после запятой для вещественных чисел.
        max_header_length : int, по умолчанию 20
            Максимальная длина заголовка колонки в режиме "notebook" (длинные обрезаются с '...').
            В режимах "markdown" и "both" (для Markdown-части) игнорируется - используются полные названия.
        styler_func : Optional[Callable[[Styler], Styler]], по умолчанию None
            Функция для дополнительной стилизации в режиме "notebook"
            (например, подсветка выбросов, градиенты).
            Не применяется в Markdown-режиме.
        as_markdown : Optional[bool], по умолчанию None
            Для обратной совместимости:
                True → mode="markdown",
                False → mode="notebook".
            Игнорируется, если задан параметр `mode`.
        mode : Optional[str], по умолчанию None
            Явное указание режима вывода. Возможные значения:
                "notebook", "markdown", "both".
            Приоритет параметров: mode > as_markdown > EDA_OUTPUT_MODE.

    Возвращаемое значение:
        None. Результат выводится напрямую в ячейку Jupyter.

    Примеры:
        >>> display_table(df)  # красивая таблица
        >>> display_table(df, mode="markdown")  # только Markdown
        >>> display_table(df, mode="both")  # и то, и другое

        # Глобальное переключение
        >>> set_output_mode("both")
        >>> display_table(df)
    """
    if df.empty:
        print("⚠️ Пустой датафрейм")
        return
    
    df = df.copy()
    df.columns = df.columns.astype(str)
    if hasattr(df.index, 'name'):  # не сломаем MultiIndex без необходимости
        df.index = df.index.astype(str)

    # Определение режима с учётом приоритетов
    if mode is not None:
        use_mode = mode
    elif as_markdown is not None:
        use_mode = "markdown" if as_markdown else "notebook"
    else:
        use_mode = EDA_OUTPUT_MODE

    if use_mode not in ("notebook", "markdown", "both"):
        raise ValueError("mode must be one of: 'notebook', 'markdown', 'both'")

    # Режим "both": рекурсивный вызов двух режимов
    if use_mode == "both":
        display_table(
            df, rows=rows, float_precision=float_precision,
            max_header_length=max_header_length, styler_func=styler_func,
            mode="notebook"
        )
        #print("\n" + "•" * 20 + " markdown " + "•" * 20 + "\n")
        print()
        display_table(
            df, rows=rows, float_precision=float_precision,
            max_header_length=max_header_length, styler_func=styler_func,
            mode="markdown"
        )
        return

    # Ограничиваем строки (или оставляем все, если rows=None)
    df_limited = df if rows is None else df.head(rows)
    if df_limited.empty:
        print("⚠️ Нет строк для отображения")
        return

    # Режим "markdown"
    if use_mode == "markdown":
        df_out = df_limited.copy()
        for col in df_out.select_dtypes(include=['number']).columns:
            if is_float_dtype(df[col]):
                df_out[col] = df_out[col].map(
                    lambda x: f"{x:.{float_precision}f}" if pd.notna(x) else ""
                )
            elif is_integer_dtype(df[col]):
                df_out[col] = df_out[col].map(
                    lambda x: f"{x:d}" if pd.notna(x) else ""
                )
        _print_compact_markdown(df_out)
        return

    # Режим "notebook"
    df_display = df_limited.copy()

    # Обрезка и уникальность заголовков - только для ноутбука
    truncated_columns = []
    seen = {}
    for col in df.columns:
        if isinstance(col, str) and len(col) > max_header_length:
            truncated = col[:max_header_length - 3] + "..."
        else:
            truncated = col

        if truncated in seen:
            seen[truncated] += 1
            unique_truncated = f"{truncated}.{seen[truncated]}"
        else:
            seen[truncated] = 0
            unique_truncated = truncated

        truncated_columns.append(unique_truncated)

    df_display.columns = truncated_columns

    original_to_truncated = dict(zip(df.columns, truncated_columns))
    numeric_cols_orig = df.select_dtypes(include=['number']).columns.tolist()
    text_cols_orig = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols_display = [original_to_truncated[col] for col in numeric_cols_orig if col in df.columns]
    text_cols_display = [original_to_truncated[col] for col in text_cols_orig if col in df.columns]

    styled = df_display.style

    # Форматирование чисел - БЕЗ разделителей разрядов
    if numeric_cols_display:
        format_dict = {}
        for orig_col in numeric_cols_orig:
            disp_col = original_to_truncated[orig_col]
            if is_float_dtype(df[orig_col]):
                format_dict[disp_col] = f"{{:.{float_precision}f}}"
            elif is_integer_dtype(df[orig_col]):
                format_dict[disp_col] = "{}"  # целые без запятых
            else:
                format_dict[disp_col] = "{:.2f}"
        styled = styled.format(format_dict)

    # Выравнивание
    if numeric_cols_display:
        styled = styled.set_properties(subset=numeric_cols_display, **{'text-align': 'right', 'font-family': 'tahoma'})
    if text_cols_display:
        styled = styled.set_properties(subset=text_cols_display, **{'text-align': 'left', 'font-family': 'tahoma'})

    # Базовые стили таблицы
    styled = styled.set_table_styles([
        {
            'selector': 'th:not(.row_heading)',
            'props': [
                ('background-color', '#ffffff !important'),
                ('color', "#7c213e"),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'thead, thead th, thead td, th.col_heading',
            'props': [
                ('background', 'transparent !important'),
                ('background-color', 'transparent !important'),
                ('border', 'none !important')
            ]
        },
        {
            'selector': 'th.col_heading',
            'props': [
                ('text-align', 'left'),
                ('font-family', 'tahoma'),
                ('font-weight', '400'),
                ('background-color', 'transparent'),
                ('padding', '8px 6px'),
                ('font-size', '11px'),
                ('color', "#5b7485")
            ]
        },
        {
            'selector': 'th.row_heading',
            'props': [
                ('background-color', "#dfe6eb"),
                ('border', '1px solid #758c9b'),
                ('font-family', 'tahoma'),
                ('text-align', 'right'),
                ('padding', '4px 6px'),
                ('font-size', '11px'),
                ('color', "#576c7b")
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('font-family', 'tahoma'),
                ('border', '1px solid #a2b3be'),
                ('padding', '4px 6px'),
                ('font-size', '11px')
            ]
        }
    ])

    if styler_func is not None:
        styled = styler_func(styled)

    from IPython.display import display
    display(styled)









#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••






def standardize_column_names(
    df: pd.DataFrame,
    verbose: bool = False,
    handle_camel_case: bool = True,
    show_summary: bool = False
) -> None:
    """
    Приводит названия колонок датафрейма к snake_case напрямую (in-place).
    
    Цель: подготовить датафрейм к дальнейшему анализу и ML в соответствии
    с общепринятыми стандартами именования признаков.
    
    Преобразования (если handle_camel_case=True):
        'Heart Rate'         → 'heart_rate'
        'CK-MB'              → 'ck_mb'
        'Systolic BP'        → 'systolic_bp'
    
    Параметры:
        df: pd.DataFrame
            Датафрейм, колонки которого будут переименованы на месте.
        verbose: bool, по умолчанию False
            Выводить ли информацию о выполненных преобразованиях.
        handle_camel_case: bool, по умолчанию True
            Преобразовывать ли CamelCase → snake_case.
        show_summary: bool, по умолчанию False
            Показывать ли краткую сводку по датафрейму после переименования
    
    Возвращает:
        None. Изменения применяются непосредственно к переданному датафрейму.
    
    Исключения:
        ValueError: если после преобразования возникают дублирующиеся имена колонок.
    
    Примеры:
        >>> standardize_column_names(df)
        >>> standardize_column_names(df, verbose=True)
        >>> standardize_column_names(df, verbose=True, show_summary=True)
    
    Зависимости:
        import re
    """
    def _to_snake_case(name: str, handle_camel: bool = True) -> str:
        # Шаг 1: заменяем все недопустимые символы на подчёркивания
        s1 = re.sub(r'[^a-zA-Z0-9]+', '_', name)
        # Шаг 2: (опционально) преобразуем CamelCase → snake_case
        if handle_camel:
            s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        else:
            s2 = s1
        # Шаг 3: приводим к нижнему регистру и убираем множественные/краевые подчёркивания
        s3 = re.sub(r'_+', '_', s2.lower()).strip('_')
        return s3 if s3 else "unnamed_column"

    original_cols = list(df.columns)
    new_cols = [_to_snake_case(col, handle_camel=handle_camel_case) for col in original_cols]

    # 🔒 Проверка на коллизии имён
    if len(new_cols) != len(set(new_cols)):
        duplicates = [col for col in set(new_cols) if new_cols.count(col) > 1]
        raise ValueError(
            f"Коллизия имён колонок после стандартизации. Дубликаты: {duplicates}. "
            "Проверьте исходные названия - они могут давать одинаковый snake_case."
        )

    # Применяем изменения
    df.columns = new_cols

    # 🖨️ Опциональный вывод
    if show_summary:
        from utils.viz import dataset_profile  # локальный импорт, чтобы избежать circular dependency
        dataset_profile(df, report='head')
    
    if verbose:
        total_cols = len(df.columns)
        changed = [(orig, new) for orig, new in zip(original_cols, new_cols) if orig != new]
        unchanged = total_cols - len(changed)

        if not changed:
            print("📌 Колонки уже соответствуют snake_case - изменений не требуется")
        else:
            if unchanged == 0:
                print(f"🔤 Все {total_cols} колонок приведены к snake_case:")
            else:
                print(f"🔤 {len(changed)} из {total_cols} колонок приведены к snake_case "
                      f"(остальные {unchanged} уже в корректном формате):")

            rename_df = pd.DataFrame(changed, columns=["Исходное имя", "Новое имя"])
            display_table(rename_df, rows=len(rename_df))
            print('')




#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# 🧹 Вспомогательные функции для аудита категориальных данных
#
# Этот блок содержит утилиты для выявления несогласованности в категориальных признаках:
# - _find_typo_groups: находит группы похожих строк (расстояние Левенштейна ≤ max_distance)
# - _levenshtein_distance: измеряет схожесть строк (для поиска опечаток и дублей)
# - _normalize_text: приводит текст к стандартному виду (регистр, ё/е, знаки)
# - _is_likely_numeric: определяет, не являются ли строковые категории на самом деле числами

# _find_typo_groups: Находит группы похожих строк (расстояние Левенштейна ≤ max_distance)
def _find_typo_groups(values: Set[str], max_distance: int = 2) -> List[List[str]]:
    """
    Находит группы похожих строк (расстояние Левенштейна ≤ max_distance).
    
    Параметры:
        values: Set[str] - уникальные значения для анализа
        max_distance: int - максимальное расстояние для группировки
    
    Возвращаемое значение:
        List[List[str]] - список групп похожих значений; каждая группа содержит ≥2 элемента
    """
    if len(values) < 2:
        return []
    
    # Нормализуем и сопоставляем
    normalized_to_original = {}
    for val in values:
        normalized = _normalize_text(str(val))
        if normalized not in normalized_to_original:
            normalized_to_original[normalized] = []
        normalized_to_original[normalized].append(val)
    
    # Находим похожие нормализованные значения
    normalized_values = list(normalized_to_original.keys())
    used = set()
    groups = []
    
    for i, norm_val1 in enumerate(normalized_values):
        if norm_val1 in used:
            continue
            
        current_group = normalized_to_original[norm_val1].copy()
        used.add(norm_val1)
        
        for j in range(i + 1, len(normalized_values)):
            norm_val2 = normalized_values[j]
            if norm_val2 in used:
                continue
            if _levenshtein_distance(norm_val1, norm_val2) <= max_distance:
                current_group.extend(normalized_to_original[norm_val2])
                used.add(norm_val2)
        
        if len(set(current_group)) > 1:
            groups.append(current_group)
    
    return groups

def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Вычисляет расстояние Левенштейна между двумя строками.
    
    Описание: Возвращает минимальное число операций вставки, удаления или замены символа,
              необходимых для превращения s1 в s2. Используется для поиска похожих строк
              и обнаружения опечаток.
    
    Параметры:
        s1: str - первая строка для сравнения
        s2: str - вторая строка для сравнения
    
    Возвращаемое значение:
        int - расстояние Левенштейна (неотрицательное целое число)
    """
    
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def _normalize_text(text: str) -> str:
    """
    Базовая нормализация текста для сравнения категориальных значений.
    
    Описание: Приводит к нижнему регистру, заменяет 'ё' на 'е',
              удаляет мягкий/твёрдый знаки, заменяет спецсимволы на пробелы,
              сохраняет буквы, цифры и пробелы. Удаляет лишние пробелы.
    """
    text = text.lower()
    text = text.replace('ё', 'е')
    text = text.replace('ь', '').replace('ъ', '')
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text, flags=re.IGNORECASE)
    return ' '.join(text.split())


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def _is_likely_numeric(series: pd.Series) -> bool:
    """
    Определяет, является ли серия преимущественно числом в строковом представлении.
    
    Параметры:
        series: pd.Series - серия для проверки (может содержать пропуски или строки)
    
    Возвращаемое значение:
        bool - True, если более 90% непустых значений можно преобразовать во float
    """
    
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    numeric_count = 0
    for val in non_null:
        try:
            float(val)
            numeric_count += 1
        except ValueError:
            pass
    return numeric_count / len(non_null) > 0.9


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# _bytes_to_human_readable: Преобразует размер в байтах в человекочитаемую строку с единицами измерения
def _bytes_to_human_readable(size_bytes: int) -> str:
    """
    Преобразует размер в байтах в человекочитаемую строку с единицами измерения.
    
    Параметры:
        size_bytes: int - размер в байтах (должен быть неотрицательным)
    
    Возвращаемое значение:
        str - строка вида '1.23 КБ', '456.0 байт', '2.1 ГБ'
    """
    
    # Определяем единицы измерения
    units = ['байт', 'КБ', 'МБ', 'ГБ']
    size = float(size_bytes)
    unit_index = 0
    
    # Пока размер больше 1024 и не достигли максимальной единицы
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Округляем до двух знаков после запятой
    return f"{round(size, 2)} {units[unit_index]}"


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# label_for_dataset: Определяет название и описание датасета по объекту DataFrame
from typing import Optional, Union, Literal
import pandas as pd

def label_for_dataset(
    df: pd.DataFrame,
    separator: Optional[str] = None,
    format: Literal["tuple", "string"] = "tuple"
) -> Union[tuple[str, str], str]:
    """
    Определяет название и описание датасета по объекту DataFrame.

    Поведение:
        - format="tuple" (по умолчанию): возвращает (имя, описание_с_separator).
        - format="string": возвращает готовую строку "имя описание_с_separator".

    Особенности:
        - Если описание отсутствует, возвращается только имя (без лишних пробелов).
        - Автоматически ищет датафрейм в глобальных переменных Jupyter Notebook.
        - Использует справочник DATASET_DESCRIPTIONS для получения описания.

    Параметры:
        df : pd.DataFrame
            Объект датафрейма.
        separator : Optional[str], по умолчанию None
            Как форматировать описание:
                - None → " описание"
                - "•" → " • описание"
                - "()" → " (описание)"
                - "[...]" → " [...] описание"
        format : {"tuple", "string"}, по умолчанию "tuple"
            Формат возвращаемого значения.

    Возвращает:
        tuple[str, str] или str - в зависимости от format.
    """
    dataset_descriptions = globals().get("DATASET_DESCRIPTIONS", {})
    
    # Попытка получить globals из Jupyter
    search_space = globals()  # fallback
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            search_space = ipython.user_global_ns
    except Exception:
        pass

    for name, obj in search_space.items():
        if obj is df:
            raw_description = dataset_descriptions.get(name, "").strip()
            
            # Если описания нет - возвращаем только имя
            if not raw_description:
                if format == "tuple":
                    return name, ""
                else:
                    return name

            # Форматируем описание, если оно есть
            if separator is None:
                formatted_desc = f" {raw_description}"
            elif len(separator) == 1:
                formatted_desc = f" {separator} {raw_description}"
            elif len(separator) == 2:
                formatted_desc = f" {separator[0]}{raw_description}{separator[1]}"
            else:
                formatted_desc = f" {separator} {raw_description}"

            if format == "tuple":
                return name, formatted_desc
            else:
                return name + formatted_desc

    # Датафрейм не найден
    fallback_name = "неизвестный_датасет"
    if format == "tuple":
        return fallback_name, ""
    else:
        return fallback_name


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# label_for_column: Возвращает имя колонки и её описание из глобального справочника COLUMN_DESCRIPTIONS.
def label_for_column(
    col: str,
    separator: Optional[str] = None,
    format: Literal["tuple", "string"] = "tuple"
) -> Union[tuple[str, str], str]:
    """
    Возвращает подпись колонки из глобального справочника COLUMN_DESCRIPTIONS.

    Поведение:
        - format="tuple" (по умолчанию): возвращает (имя, описание_с_separator).
        - format="string": возвращает готовую строку "имя описание_с_separator".

    Параметры:
        col : str
            Имя колонки.
        separator : Optional[str], по умолчанию None
            Как форматировать описание:
                - None - " описание"
                - "•" - " • описание"
                - "()" - " (описание)"
                - "[...]" - " [...] описание"
        format : {"tuple", "string"}, по умолчанию "tuple"
            Формат возвращаемого значения.

    Возвращает:
        tuple[str, str] или str - в зависимости от format.
    """
    column_descriptions = globals().get("COLUMN_DESCRIPTIONS", {})
    
    raw_description = column_descriptions.get(col, "")
    if not isinstance(raw_description, str):
        raw_description = str(raw_description).strip()
    raw_description = raw_description.strip()
    
    col_name = col
    if not raw_description:
        formatted_desc = ""
    else:
        if separator is None:
            formatted_desc = f" {raw_description}"
        elif len(separator) == 1:
            formatted_desc = f" {separator} {raw_description}"
        elif len(separator) == 2:
            formatted_desc = f" {separator[0]}{raw_description}{separator[1]}"
        else:
            formatted_desc = f" {separator} {raw_description}"

    if format == "tuple":
        return col_name, formatted_desc
    else:  
        return col_name + formatted_desc


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# _fix_decimal_comma: Исправляет числовые строки с десятичной запятой («14,2» - 14.2) в датафрейме
def _fix_decimal_comma(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
        Исправляет числовые строки с десятичной запятой («14,2» - 14.2) в датафрейме.
        
        Описание:
            Для строковых колонок пытается заменить запятые на точки и преобразовать в число.
            Колонка преобразуется, только если доля валидных чисел ≥ threshold.
            После преобразования:
                - если все значения целые и нет пропусков - тип Int64,
                - иначе - float64 (с поддержкой NaN).
            Используется на этапе очистки данных перед EDA или ML.

        Параметры:
            df: pd.DataFrame - входной датафрейм
            threshold: float - минимальная доля валидных чисел для преобразования (0.0-1.0, по умолчанию 0.9)

        Возвращаемое значение:
            pd.DataFrame - копия с исправленными колонками; оригинальный df не изменяется

        Примечания:
            - Не затрагивает нестроковые колонки.
            - Не преобразует колонки с долей чисел < threshold (например, смесь текста и чисел).
            - Использует nullable-тип Int64 для целых без пропусков.
    """
    
    # Выбираем только строковые колонки
    object_cols = df.select_dtypes(include=['object']).columns
    
    cols_to_convert = []
    total_rows = len(df)
    
    # Проверка: замена запятых и попытка парсинга без циклов
    for col in object_cols:
        # Подготавливаем данные
        cleaned = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
        numeric_series = pd.to_numeric(cleaned, errors='coerce')
        valid_count = numeric_series.notna().sum()
        
        if valid_count / total_rows >= threshold:
            cols_to_convert.append(col)
    
    # Если ничего не нашли - завершаем без преобразований
    if not cols_to_convert:
        print("✔️ все псевдочисловые колонки проверены (преобразование не требуется)")
        return df
    
    # Выводим отчёт о найденных столбцах
    print("\n🛠️ Автоматическое преобразование столбцов `object` содержащих числа в `float64`\n")
    
    # Подготовка данных для преобразования
    df_updated = df.copy()
    
    for col in cols_to_convert:
        # Подготавливаем данные
        cleaned = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
        
        # Показываем статистику
        numeric_series = pd.to_numeric(cleaned, errors='coerce')
        valid_count = numeric_series.notna().sum()
        print(f"   🔎 {col}: {valid_count}/{total_rows} (совпадение: {valid_count/total_rows:.1%})")
        
        # Преобразуем с обработкой ошибок
        try:
            df_updated[col] = pd.to_numeric(cleaned, errors='coerce')
        except Exception as e:
            print(f"   ⚠️ Ошибка при преобразовании {col}: {e}")
            continue

        # Попытка понизить тип до целочисленного, если возможно
        if (
            df_updated[col].notna().all() and
            (df_updated[col] % 1 == 0).all()
        ):
            try:
                df_updated[col] = df_updated[col].astype('Int64')
            except Exception as e:
                print(f"   ⚠️ Не удалось привести {col} к Int64: {e}")
    
    print()
    
    # Показываем результаты преобразования
    for col in cols_to_convert:
        if col in df_updated.columns:  # Проверяем, что колонка не была удалена
            dtype_name = df_updated[col].dtype.name
            emoji_map = {
                'int8': '1️⃣', 'int16': '1️⃣', 'int32': '1️⃣', 'int64': '1️⃣',
                'uint8': '1️⃣', 'uint16': '1️⃣', 'uint32': '1️⃣', 'uint64': '1️⃣',
                'float16': '🔢', 'float32': '🔢', 'float64': '🔢',
                'object': '📦', 'datetime64': '📅', 'category': '🏷️'
            }
            dtype_display = f"{emoji_map.get(dtype_name, '🚨')} {dtype_name}"
            print(f"   📌 {col} 🔨 преобразовано в {dtype_display}")
    
    print(f"\n✔️ все `object` столбцы ({len(cols_to_convert)}) содержащие числа, преобразованы в `float64`")
    return df_updated


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# _detect_numerical_issue: Определяет проблемы в числовой серии для отчётов
def _detect_numerical_issues(series: pd.Series, total_rows: int) -> List[str]:
    """
    Выявляет типичные проблемы в числовой колонке для отчётов о качестве данных.
    
    Описание:
        Анализирует серию на наличие:
        - пропусков (>0%),
        - выбросов (>5% по IQR),
        - сильной асимметрии (|skewness| > 1.5),
        - почти константных значений,
        - подозрительно больших максимумов (медиана > 0, максимум > медианы × 3 и > 1000).
        Возвращает список интерпретируемых сообщений о проблемах.
        Используется в audit_numerical и dataset_profile для единообразной диагностики.

    Параметры:
        series: pd.Series - числовой признак для анализа
        total_rows: int - общее число строк в датафрейме (для расчёта % пропусков и выбросов)

    Возвращаемое значение:
        List[str] - список сообщений о проблемах; пустой список, если проблем нет
    """
    issues = []
    n_total = total_rows
    n_missing = series.isna().sum()
    
    if n_missing > 0:
        missing_pct = n_missing / n_total * 100
        issues.append(f"пропусков: {missing_pct:.1f}%")
    
    clean_series = series.dropna()
    if clean_series.empty:
        return issues

    # Выбросы
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = ((clean_series < lower_bound) | (clean_series > upper_bound)).sum()
    outliers_pct = n_outliers / n_total * 100
    if outliers_pct > 5:
        issues.append(f"выбросов: {outliers_pct:.1f}%")

    # Асимметрия
    skewness = clean_series.skew()
    if not pd.isna(skewness) and abs(skewness) > 1.5:
        issues.append(f"асимметрия: {skewness:.2f}")

    # Почти константный
    if clean_series.nunique() == 1:
        issues.append("почти константный")

    # Подозрительный максимум
    max_val = clean_series.max()
    median_val = clean_series.median()
    if median_val > 0 and max_val > median_val * 3 and max_val > 1000:
        issues.append(f"подозрительный максимум: {int(max_val)}")

    return issues


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# _format_number: Форматирует числовое значение для красивого отображения в таблица
def _format_number(x, precision: int) -> str:
    """
    Форматирует числовое значение для красивого отображения в таблицах.
    
    Описание:
        Преобразует число в строку с разделителями тысяч и заданной точностью:
        - Целые числа (включая float, представляющие целые значения) выводятся без десятичной точки.
        - Вещественные числа округляются до указанного количества знаков после запятой.
        - Пропущенные значения (NaN) отображаются как строка "NaN".
        Результат всегда включает разделители тысяч (например, 1,234,567).

    Параметры:
        x: числовое значение или np.nan - значение для форматирования
        precision: int - количество знаков после запятой для вещественных чисел

    Возвращаемое значение:
        str - отформатированная строка, готовая к отображению в ячейке таблицы
    """
    if pd.isna(x):
        return "NaN"
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()):
        return f"{int(x):,}"
    else:
        return f"{x:,.{precision}f}"


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# preview: Универсальный предпросмотр датафрейма.
def preview(
    df: pd.DataFrame,
    mode: Literal["auto", "head", "sample", "info", "full"] = "auto",
    n: Optional[int] = None,
    float_precision: int = 3,
    max_header_length: int = 20,
    cmap: Optional[str] = None,
    col: Optional[str] = None,
    random_state: Optional[int] = 42
) -> None:
    """
    Универсальный предпросмотр датафрейма.
    
    Описание:
        Заменяет print(df), df.head(), df.info(), audit_categorical_frequencies в EDA:
        - 'auto': имитирует print(df) - head(5) + tail(5) с разделителем (по умолчанию)
        - 'head': первые n строк
        - 'sample': случайные n строк (воспроизводимо при random_state)
        - 'info': структура, типы, пропуски (как dataset_profile в режиме 'short')
        - 'full': весь датафрейм (только если ≤1000 строк)
        - col="breed": анализ частот категорий (аналог audit_categorical_frequencies)
        Использует display_table, COLUMN_DESCRIPTIONS, DATASET_DESCRIPTIONS.

    Параметры:
        df: pd.DataFrame - датафрейм для предпросмотра
        mode: Literal["auto", "head", "sample", "info", "full"] - режим отображения
        n: Optional[int] - количество строк (для 'head', 'sample'); если None - зависит от режима
        float_precision: int - знаки после запятой
        max_header_length: int - макс. длина заголовка
        cmap: Optional[str] - палитра для градиента
        col: Optional[str] - имя категориальной колонки для анализа частот
        random_state: Optional[int] - фиксация случайности для sample (по умолчанию 42)

    Возвращаемое значение:
        None - вывод через display_table или print
    """
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return
    
    # Валидация параметра mode
    if mode in ("info"):
        dataset_profile(df, report="short")


    if mode in ("auto", "head", "sample", "full"):
        dataset_profile(df, report="summary")

        # Анализ отдельной колонки
        if col is not None:
            if col not in df.columns:
                print(f"❌ Колонка '{col}' не найдена в датафрейме")
                return

            series = df[col]
            n_total = len(series)
            n_unique = series.nunique()

            col_name, col_desc = label_for_column(col, separator="•")
            col_label = f"{col_name}{col_desc}" if col_desc else col_name

            print(f"\n🎹 Частота значений '{col_label}'")
            print(f"📐 Общее количество строк: {n_total:,} × {n_unique} групп")

            value_counts = series.value_counts(sort=True, ascending=False)
            result = pd.DataFrame({
                'Значение': value_counts.index,
                'Количество строк': value_counts.values,
                'Процент от общего числа строк': (value_counts / n_total * 100).round(3)
            }).reset_index(drop=True)

            styler_func = None
            if cmap is None:
                cmap = 'YlGn'
            styler_func = lambda s: s.background_gradient(
                subset=["Процент от общего числа строк"], 
                cmap=cmap
            )

            display_table(
                result,
                rows=len(result),
                float_precision=3,
                max_header_length=1000,
                styler_func=styler_func
            )
            return

        n_rows, n_cols = df.shape

        # Режим full
        if mode == "full":
            if n_rows > 1000:
                print(f"⚠️ Слишком много строк ({n_rows}) для режима 'full'. Используйте 'head' или 'sample'.")
                return
            display_table(df, rows=n_rows, float_precision=float_precision, max_header_length=max_header_length)
            return

        # Режим auto - имитация print(df)
        if mode == "auto":
            head_tail_n = n if n is not None else 5
            if n_rows <= 2 * head_tail_n:
                display_table(df, rows=n_rows, float_precision=float_precision, max_header_length=max_header_length)
            else:
                head_df = df.head(head_tail_n)
                tail_df = df.tail(head_tail_n)

                # Вспомогательная функция форматирования
                def _format_number(x, precision: int) -> str:
                    if pd.isna(x):
                        return "NaN"
                    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()):
                        return f"{int(x):,}"
                    else:
                        return f"{x:,.{precision}f}"

                # Форматируем числовые колонки
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in head_df.columns:
                        head_df = head_df.copy()
                        head_df[col] = head_df[col].apply(lambda x: _format_number(x, float_precision))
                    if col in tail_df.columns:
                        tail_df = tail_df.copy()
                        tail_df[col] = tail_df[col].apply(lambda x: _format_number(x, float_precision))

                # Создаём строку-разделитель с оригинальным индексом
                separator_index = "⋮"
                separator_row = pd.DataFrame(
                    [["⋮"] * len(df.columns)],
                    columns=df.columns,
                    index=[separator_index]
                )

                # Объединяем БЕЗ сброса индекса - сохраняем оригинальные индексы
                preview_df = pd.concat([head_df, separator_row, tail_df], ignore_index=False)

                print(f"\n📋 Первые и последние {head_tail_n} строк датасета:")
                display_table(
                    preview_df,
                    rows=len(preview_df),
                    float_precision=3,  # уже отформатировано вручную!
                    max_header_length=max_header_length
                )
            return

        # Режимы head / sample
        if n is None:
            n = 10

        if mode == "sample":
            sample_size = min(n, n_rows)
            df_to_show = df.sample(n=sample_size, random_state=random_state)
            rows_to_show = sample_size
        else:  # "head"
            df_to_show = df.head(n)
            rows_to_show = min(n, n_rows)

        # Стилизация
        styler_func = None
        if cmap is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                styler_func = lambda s: s.background_gradient(
                    subset=[col for col in numeric_cols if col in df_to_show.columns],
                    cmap=cmap
                )

        display_table(
            df_to_show,
            rows=rows_to_show,
            float_precision=float_precision,
            max_header_length=max_header_length,
            styler_func=styler_func
        )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# fn_safe_convert_to_datetime • Безопасно преобразует столбец в datetime, сохраняя информацию о временных зонах.
def dataset_convert_datetime(
    df: pd.DataFrame,
    date_column: str,
    convert_to_utc: bool = True
) -> bool:
    """
    Безопасно преобразует указанный столбец датафрейма в тип datetime с диагностикой и обработкой временных зон.

    Описание:
        Функция пытается преобразовать строковый или частично валидный столбец в datetime.
        Автоматически обнаруживает наличие временных зон и при необходимости удаляет их (конвертация в naive UTC).
        При ошибке выводит первые 10 некорректных значений и оставляет датафрейм без изменений.

    Параметры:
        df: pd.DataFrame - целевой датафрейм (модифицируется на месте)
        date_column: str - имя столбца для преобразования
        convert_to_utc: bool - если True, удаляет временную зону после преобразования, оставляя локальное время в UTC

    Возвращаемое значение:
        bool - True при успешном преобразовании, False при обнаружении ошибок
    """

    # Проверка существования столбца
    if date_column not in df.columns:
        print(f"❌ столбец '{date_column}' не найден")
        return False

    # Сохраняем исходный тип и пропуски
    original_dtype = df[date_column].dtype
    initial_na_count = df[date_column].isna().sum()
    print(f"• Исходный тип: {original_dtype}")
    print(f"• Пропусков: {initial_na_count}")

    # Копируем исходный ряд для анализа
    original_series = df[date_column].copy()

    # Определяем наличие временных зон - векторизованно, без циклов
    # Пробуем преобразовать в datetime с utc=True - если есть зоны, они сохранятся
    test_series = pd.to_datetime(original_series, errors='coerce', utc=True)
    has_timezone = test_series.dt.tz is not None

    try:
        # Преобразуем с учётом временных зон
        converted_series = pd.to_datetime(original_series, errors='raise', utc=True)

        if convert_to_utc and has_timezone:
            print("⚠️ обнаружены даты с временными зонами - конвертируем в локальное время (UTC убран)")
            converted_series = converted_series.dt.tz_localize(None)
        elif convert_to_utc and not has_timezone:
            print("ℹ️ даты без временной зоны - оставлены как есть (UTC не применяется)")
        elif not convert_to_utc and has_timezone:
            print("ℹ️ сохраняем временные зоны - не преобразуем в UTC")

        # Применяем результат
        df[date_column] = converted_series

        print(f"✔️ столбец '{date_column}' успешно преобразован в datetime")
        print(f"💾 новый тип: {df[date_column].dtype}")
        return True

    except (ValueError, TypeError) as e:
        print(f"\n🚨 обнаружены некорректные значения в '{date_column}'")
        print("📝 Проблемные значения (первые 10):")

        # Ищем невалидные значения векторизованно - без циклов
        invalid_mask = pd.to_datetime(original_series, errors='coerce').isna()
        invalid_values = original_series[invalid_mask & original_series.notna()].head(10)
        
        for idx, val in invalid_values.items():
            print(f"  строка {idx}: [{val}]")

        if len(invalid_values) == 0:
            print("  (не найдено - возможно, ошибка в структуре данных)")

        print("\n❌ преобразование отменено - исправьте данные перед повторной попыткой")
        return False


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# load_dataset: Загружает датасет с автоматической очисткой и диагностикой качества
# UPD: decimal: str = '.',
def load_dataset(
    dataset_name: str,
    file_path: Optional[Union[str, Path]] = None,
    sep: str = ',',
    decimal: str = '.',
    drop_duplicates: bool = False,
    auto_audit_numeric: bool = True,
    replace_whitespace_with_nan: bool = False
) -> pd.DataFrame:
    """
    Загружает датасет с автоматической очисткой и диагностикой качества.
    Описание:
        Умная замена pd.read_csv() для EDA и ML:
        - Поддерживает различные разделители десятичных знаков через параметр decimal,
        - Опционально заменяет значения из одних пробелов на NaN (если replace_whitespace_with_nan=True),
        - Удаляет лишние пробелы в строках,
        - Анализирует дубликаты и даёт рекомендации,
        - Проверяет пропуски и потенциально числовые колонки,
        - Выводит интерпретируемый отчёт о состоянии данных.
        Использует глобальный справочник CSV_PATHS для поиска файлов по имени.
    Параметры:
        dataset_name: str - имя датасета (ключ в глобальном CSV_PATHS)
        file_path: Optional[Union[str, Path]] - путь к файлу (если не указан - берётся из CSV_PATHS)
        sep: str - разделитель в CSV (по умолчанию ',')
        decimal: str - разделитель десятичных знаков (по умолчанию '.')
        drop_duplicates: bool - удалить полные дубликаты строк (по умолчанию False)
        auto_audit_numeric: bool - проверять потенциально числовые колонки (по умолчанию True)
        replace_whitespace_with_nan: bool - заменять строки из одних пробелов на NaN (по умолчанию False)
    Возвращаемое значение:
        pd.DataFrame - очищенный и готовый к анализу датафрейм
    Исключения:
        KeyError - если dataset_name отсутствует в CSV_PATHS
        FileNotFoundError - если файл не найден
        UnicodeDecodeError - при проблемах с кодировкой
    """
    # 1. Определяем путь к файлу
    if file_path is None:
        # Ищем CSV_PATHS как атрибут текущего модуля
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, 'CSV_PATHS'):
            raise RuntimeError(
                "Глобальная переменная 'CSV_PATHS' не найдена в модуле viz. "
                "Создайте её в ноутбуке: `import utils.viz; utils.viz.CSV_PATHS = {...}`"
            )
        CSV_PATHS = getattr(current_module, 'CSV_PATHS')
        if dataset_name not in CSV_PATHS:
            available = ', '.join(CSV_PATHS.keys())
            raise KeyError(
                f"🚨 Датасет '{dataset_name}' не найден в CSV_PATHS | "
                f"📢 Доступные датасеты: {available}"
            )
        file_path = CSV_PATHS[dataset_name]
    file_path = str(file_path)
    if not isinstance(file_path, str):
        raise TypeError(f"🚨 'file_path' должен быть строкой или Path, получено {type(file_path).__name__}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ файл не найден: {os.path.abspath(file_path)}")
    # 2. Загружаем данные
    try:
        df = pd.read_csv(file_path, sep=sep, decimal=decimal)
        df = df.infer_objects()
        original_rows = len(df)
        original_cols = len(df.columns)
        missing_cols = df.columns[df.isna().any()].tolist()
        # Анализ дубликатов - вычисляем ОДИН РАЗ
        duplicates_full = df[df.duplicated(keep=False)]
        num_duplicates_total = len(duplicates_full)
        unique_duplicate_groups = len(duplicates_full.drop_duplicates())
        # Вывод информации о файле
        file_size = _bytes_to_human_readable(os.path.getsize(file_path))
        print(f"💾 файл '{os.path.basename(file_path)}' успешно загружен!")
        print(f"     🧠 Память          : {file_size}")
        print(f"     📐 Размер датасета : {original_rows} строк × {original_cols} колонок")
        # 3. Анализ дубликатов
        candidate_ids = []
        if num_duplicates_total > 0:
            duplicate_ratio = num_duplicates_total / original_rows
            print(f"\n🚨 Обнаружено {num_duplicates_total} полных дубликатов строк ({duplicate_ratio:.1%})")
            print(f"     🔢 Уникальных комбинаций: [ {unique_duplicate_groups} ]")
            # Проверка: появится ли уникальный ID после удаления?
            df_clean_test = df.drop_duplicates()
            for col in df_clean_test.columns:
                if df_clean_test[col].nunique() == len(df_clean_test) and df_clean_test[col].nunique() > 1:
                    candidate_ids.append(col)
            if candidate_ids:
                ids_str = ", ".join(f"`{col}`" for col in candidate_ids[:2])
                print(f"     🔍 После удаления дубликатов колонка {ids_str} станет уникальной - вероятно, это идентификатор 🆔")
                print("     💡 Удаление дубликатов безопасно: данные не потеряют смысловую уникальность")
            elif duplicate_ratio > 0.5:
                print("     ⚠️ Высокая доля повторов (>50%) - вероятно, технические дубликаты загрузки")
                print("     💡 Удаление рекомендуется")
            elif original_rows < 100:
                print("     💡 Датасет маленький - повторы могут быть валидными")
            else:
                print("     💡 Повторы могут быть как ошибками, так и валидными данными")
            if not drop_duplicates:
                if len(duplicates_full) <= 50:
                    display_table(duplicates_full, len(duplicates_full))
                else:
                    display_table(duplicates_full, 10)
                print("📢 для автоматического удаления: запусти с drop_duplicates=True")
        else:
            print("\n✔️ полных дубликатов строк не обнаружено")
        # Удаление дубликатов (если запрошено)
        if drop_duplicates and num_duplicates_total > 0:
            df = df.drop_duplicates(keep='first').reset_index(drop=True)
            cleaned_rows = len(df)
            removed = original_rows - cleaned_rows
            print(f"\n🧹 Автоматическое удаление дубликатов строк: [ {removed} ] строк удалено 🗑️")
            if candidate_ids:
                print(f"     ✔️ Безопасно: после очистки колонки {', '.join(f'`{c}`' for c in candidate_ids[:2])} обеспечивает уникальность")
            print(f"     📐 Итоговый размер: {cleaned_rows} строк × {len(df.columns)} колонок\n")
        # 4. Умная проверка числовых колонок
        if auto_audit_numeric:
            potential_numeric_cols = []
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue
                try:
                    # Пробуем преобразовать в число
                    test_series = non_null.astype(str).str.replace(',', '.', regex=False)
                    numeric_test = pd.to_numeric(test_series, errors='coerce')
                    valid_count = numeric_test.notna().sum()
                    total_count = len(non_null)
                    valid_ratio = valid_count / total_count if total_count > 0 else 0
                    if valid_ratio >= 0.9:  # Только если ≥90% чисел
                        potential_numeric_cols.append({
                            'col': col,
                            'valid_count': valid_count,
                            'total_count': total_count,
                            'valid_ratio': valid_ratio
                        })
                except (ValueError, TypeError):
                    continue
            if potential_numeric_cols:
                print(f"📢 Потенциально числовые колонки:")
                for item in potential_numeric_cols:
                    print(f"       • {item['col']}: числовых строк {item['valid_count']} из {item['total_count']} [ {item['valid_ratio']:.1%} ]")
                print()
                # Рекомендации по decimal
                if decimal == '.':
                    has_comma_values = any(
                        ',' in str(val) 
                        for col_item in potential_numeric_cols 
                        for val in df[col_item['col']].dropna().head(3)
                        if isinstance(val, str)
                    )
                    if has_comma_values:
                        print("   💡 Совет: используйте decimal=',' для загрузки чисел с запятыми\n")
            else:
                print("✔️ все числовые колонки загружены корректно")

        # 5. Очистка пробелов и опциональная замена "только пробелов" на NaN
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            whitespace_counts = pd.Series(index=string_cols, dtype='int')
            for col in string_cols:
                only_spaces = df[col].astype(str).str.match(r'^\s*$')
                whitespace_counts[col] = only_spaces.sum()

            # Выводим информацию о пробельных строках ВСЕГДА
            problematic_cols = whitespace_counts[whitespace_counts > 0]
            if not problematic_cols.empty:
                if replace_whitespace_with_nan:
                    print("🧹 автоматическая ЗАМЕНА ПРОБЕЛОВ НА NaN:")
                    for col in problematic_cols.index:
                        count = whitespace_counts[col]
                        total = len(df)
                        pct = (count / total) * 100
                        print(f"   • {col}: {count} ({pct:.2f}%) → NaN")
                    # Выполняем замену
                    for col in problematic_cols.index:
                        only_spaces = df[col].astype(str).str.match(r'^\s*$')
                        df.loc[only_spaces, col] = np.nan
                else:
                    print("⚠️ Обнаружены ячейки, содержащие только пробелы:")
                    for col in problematic_cols.index:
                        count = whitespace_counts[col]
                        total = len(df)
                        pct = (count / total) * 100
                        print(f"     • {col}: {count} ({pct:.2f}%)")
                    #print("   💡 для автозамены на NaN, используй - replace_whitespace_with_nan=True")
            else:
                print("✔️ ячеек, содержащих только пробелы не обнаружено")

            # Удаляем лишние пробелы у всех строковых колонок (всегда)
            for col in string_cols:
                df[col] = df[col].str.strip()
        else:
            print("✔️ строковых колонок нет - очистка пробелов не требуется")
        # 6. Отчёт о пропусках
        if missing_cols:
            print("⚠️ колонки с пропусками:")
            for col in missing_cols:
                pct = df[col].isna().sum() / len(df) * 100
                print(f"     • {col}: {df[col].isna().sum()} ({pct:.2f}%)")
        else:
            print("✔️ все колонки без пропусков")
        return df
    # 7. Обработка ошибок: ПРОБРАСЫВАЕМ исключение
    except FileNotFoundError:
        print(f"❌ файл не найден: {os.path.abspath(file_path)}")
        raise
    except pd.errors.EmptyDataError:
        print("❌ файл пустой")
        raise
    except UnicodeDecodeError:
        print("❌ ошибка кодировки. Попробуйте encoding='utf-8' или 'cp1251'")
        raise
    except Exception as e:
        print(f"❌ ошибка при загрузке: {str(e)}")
        raise


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# dataset_profile: Выводит профиль датафрейма - структурированный отчёт о его структуре и качестве
def dataset_profile(
    df: pd.DataFrame,
    report: Literal["head", "summary", "cols", "short", "full"] = "head"
) -> None:
    """
    Выводит профиль датафрейма - структурированный отчёт о его структуре и качестве.
    
    Описание:
        Универсальный инструмент для быстрого осмотра данных перед EDA или ML.
        Поддерживает 5 уровней детализации - от краткого заголовка до статистики по колонкам.
        Использует глобальные справочники DATASET_DESCRIPTIONS и COLUMN_DESCRIPTIONS
        для автоматической подписи датасетов и признаков.

    Параметры:
        df: pd.DataFrame - датафрейм для профилирования
        report: Literal["head", "summary", "cols", "short", "full"] - уровень детализации:
            - "head"   : только имя и описание датасета
            - "summary": + память, размер, типы признаков
            - "cols"   : + список колонок с описаниями
            - "short"  : как "summary" + колонки с эмодзи-типами
            - "full"   : как "short" + пропуски, кардинальность, выбросы

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука
    """
    if report not in ("head", "summary", "cols", "short", "full"):
        raise ValueError(
            f"Некорректное значение report='{report}'. "
            f"Допустимые значения: 'head', 'summary', 'cols', 'short', 'full'"
        )

    # Типизация колонок
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = set(df.select_dtypes(include=["object", "category"]).columns)
    datetime_cols = set(df.select_dtypes(include=["datetime64"]).columns)
    boolean_cols = set(df.select_dtypes(include=["boolean"]).columns)
    string_cols = set(df.select_dtypes(include=["string"]).columns)
    other_cols = (
        set(df.columns) 
        - numeric_cols - categorical_cols - datetime_cols 
        - boolean_cols - string_cols
    )

    # Метаданные
    dataset_name, dataset_desc = label_for_dataset(df, separator='•')
    n_rows, n_cols = df.shape
    memory_kb = df.memory_usage(deep=True).sum() / 1024

    # Основная информация (всегда)
    print(f"🗃️ Датасет {dataset_name}{dataset_desc}")

    # Память и размер (summary, short, full)
    if report in ("summary", "short", "full"):
        print(f"     🧠 Память                   : {memory_kb:.1f} KB")
        print(f"     📐 Размер датасета          : {n_rows} строк × {n_cols} колонок")
        if numeric_cols:
            print(f"     🔢 Числовых признаков       : {len(numeric_cols)}")
        if categorical_cols:
            print(f"     🏷️ Категориальных признаков : {len(categorical_cols)}")
        if datetime_cols:
            print(f"     📅 Признаков даты/времени   : {len(datetime_cols)}")
        if boolean_cols:
            print(f"     ✅ Булевых признаков        : {len(boolean_cols)}")
        if string_cols:
            print(f"     🔤 Текстовых признаков      : {len(string_cols)}")
        if other_cols:
            print(f"     ⚠️ Прочих признаков         : {len(other_cols)}")
            print(f"        Типы: {', '.join(str(df[col].dtype) for col in other_cols)}")
        if n_cols == 0:
            print("     ⚠️ Датасет не содержит колонок")

    # Список колонок и статистика
    if report in ("cols", "short", "full"):
        print("\n🎹 Колонки датасета:")

        type_config = {
            'numeric':      ('🔢', "числовой"),
            'categorical':  ('🏷️', "категориальный"),
            'datetime':     ('📅', "дата/время"),
            'boolean':      ('✅', "булев"),
            'string':       ('🔤', "строка"),
            'other':        ('📦', "прочий")
        }

        col_type_map = {}
        for col in df.columns:
            if col in numeric_cols:
                col_type_map[col] = 'numeric'
            elif col in categorical_cols:
                col_type_map[col] = 'categorical'
            elif col in datetime_cols:
                col_type_map[col] = 'datetime'
            elif col in boolean_cols:
                col_type_map[col] = 'boolean'
            elif col in string_cols:
                col_type_map[col] = 'string'
            else:
                col_type_map[col] = 'other'

        for col in df.columns:
            col_type = col_type_map[col]
            emoji, _ = type_config[col_type]
            
            # Получаем подпись ОДИН РАЗ
            if report == "cols":
                col_name, desc = label_for_column(col, separator="-")
                print(f"     • [ {col_name} ]{desc}")
            elif report == "short":
                col_name, desc = label_for_column(col, separator="-")
                print(f"     {emoji} {col_name}{desc}")
            elif report == "full":
                col_name, desc = label_for_column(col, separator="•")
                parts = [desc]

                # Пропуски - для ВСЕХ типов колонок
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    pct = n_missing / len(df) * 100
                    parts.append(f" ⚠️ пропусков: {n_missing} ({pct:.1f}%)")

                # Кардинальность - для категориальных-подобных
                if col_type in ('categorical', 'string', 'boolean'):
                    n_unique = df[col].nunique()
                    parts.append(f" 💎 [групп: {n_unique}]")
                    if len(df) > 0 and n_unique / len(df) > 0.5:
                        parts.append(" ⚠️ высокая кардинальность")

                # Проблемы - ТОЛЬКО для числовых (через единый детектор)
                if col_type == 'numeric':
                    issues = _detect_numerical_issues(df[col], len(df))
                    if issues:
                        parts.append(f"\n         📌 {' • '.join(issues)}")

                print(f"     {emoji} {col_name}{''.join(parts)}")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# dataset_quick_audit: Проводит быстрый скрининг качества данных перед углублённым анализом
# dataset_quick_audit: Проводит быстрый скрининг качества данных перед углублённым анализом
def dataset_quick_audit(
    df: pd.DataFrame,
    report: Literal["head", "summary", "short"] = "summary",
    outlier_iqr_multiplier: float = 1.5,
    extreme_iqr_multiplier: float = 3.0,
    detect_outliers: bool = True,
    detect_extremes: bool = True,
) -> None:
    """
    Проводит быстрый скрининг качества данных перед углублённым анализом.
    
    Описание:
        Комплексная, но лёгкая проверка датафрейма на типичные проблемы:
        - дубликаты строк,
        - пропуски по колонкам,
        - потенциальные идентификаторы,
        - выбросы и экстремумы (IQR × outlier_iqr_multiplier, IQR × extreme_iqr_multiplier),
        - асимметрию распределений,
        - почти константные признаки,
        - сильные корреляции (>0.7),
        - проблемы в категориальных признаках (мусор, дисбаланс).
        Использует dataset_profile для базовой информации.

    Новые параметры:
        outlier_iqr_multiplier : float, по умолчанию 1.5
            Множитель IQR для определения выбросов.
        extreme_iqr_multiplier : float, по умолчанию 3.0
            Множитель IQR для определения экстремальных значений.
        detect_outliers : bool, по умолчанию True
            Выполнять ли проверку на выбросы.
        detect_extremes : bool, по умолчанию True
            Выполнять ли проверку на экстремальные значения.

    Параметры:
        df: pd.DataFrame - датафрейм для диагностики
        report: Literal["head", "summary", "short"] - уровень детализации базовой информации:
            - "head": только имя датасета,
            - "summary": + память и размер,
            - "short": + типы признаков

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука

    Примеры:
        >>> dataset_quick_audit(df)  # как раньше
        >>> dataset_quick_audit(df, outlier_iqr_multiplier=2.0)  # мягче
        >>> dataset_quick_audit(df, detect_extremes=False)  # без экстремумов
    """
    # Валидация параметра report
    if report not in ("head", "summary", "short"):
        raise ValueError(
            f"Некорректное значение report='{report}'. "
            f"Допустимые значения: 'head', 'summary', 'short'"
        )

    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    # Основная информация о датасете
    n_rows, n_cols = df.shape
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Диагностика качества данных датасета\n")

    dataset_profile(df, report=report)
    print()
    
    # Анализ дубликатов
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        pct = n_duplicates / n_rows * 100
        print(f"🚨 Дублированные строки: {n_duplicates} ({pct:.2f}%)")
    else:
        print("✔️ Дубликатов строк нет")

    # Анализ пропусков
    missing_any = False
    for col in df.columns:
        col_name, col_desc = label_for_column(col, separator='•')
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            pct = n_missing / n_rows * 100
            print(f"🚨 Пропущенные значения в {col_name}{col_desc}: {n_missing} ({pct:.2f}%)")
            missing_any = True
    if not missing_any:
        print("✔️ Пропусков нет")

    # Потенциальные идентификаторы
    id_candidates = []
    for col in df.columns:
        n_uniq = df[col].nunique()
        n_total = len(df)
        pct_unique = n_uniq / n_total * 100

        # Пропускаем float-колонки с нецелыми значениями
        if pd.api.types.is_float_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and not (non_null % 1 == 0).all():
                continue  # Нецелые float - плохой кандидат на ID

        if pct_unique >= 95.0:
            # Приоритет 1: имя колонки
            name_score = 1.0 if any(k in col.lower() for k in ['id', 'key', 'code', 'uid']) else 0.0
            # Приоритет 2: тип данных
            type_score = 1.0 if df[col].dtype in ['object', 'int64', 'int32'] else 0.5
            # Приоритет 3: уникальность
            unique_score = pct_unique / 100

            score = name_score * 3 + type_score * 2 + unique_score  # взвешенная сумма

            col_name, col_desc = label_for_column(col, separator='•')
            status = "⚠️ (не уникален!)" if n_uniq < n_total else ""
            info = f"{col_name}{col_desc} ({n_uniq} уникальных, {pct_unique:.1f}%){status}"
            id_candidates.append({
                'col': col,
                'info': info,
                'score': score,
                'is_duplicate': n_uniq < n_total
            })

    if id_candidates:
        # Сортируем по скору
        best = max(id_candidates, key=lambda x: x['score'])
        print(f"🆔 Потенциальный идентификатор: {best['info']}")
        if best['is_duplicate']:
            print(f"     📌 Колонка содержит дубликаты - проверьте данные.")
    else:
        print("✔️ Потенциальных идентификаторов не обнаружено")

    # Анализ числовых признаков
    if numeric_columns:
        # Вспомогательная функция для проверки подозрительного максимума
        def _has_suspicious_max(series: pd.Series) -> bool:
            if series.empty:
                return False
            median_val = series.median()
            max_val = series.max()
            return median_val > 0 and max_val > median_val * 3 and max_val > 1000

        # Выбросы (IQR × outlier_iqr_multiplier)
        if detect_outliers:
            outliers_any = False
            for col in numeric_columns:
                col_name, col_desc = label_for_column(col, separator='•')
                valid_data = df[col].dropna()
                if len(valid_data) == 0 or valid_data.nunique() <= 20:
                    continue

                q1, q3 = valid_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - outlier_iqr_multiplier * iqr
                upper_bound = q3 + outlier_iqr_multiplier * iqr
                n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_pct = n_outliers / n_rows * 100

                if n_outliers > 0:
                    print(f"🔶 Выбросы в {col_name}{col_desc}: {n_outliers} ({outlier_pct:.1f}%)")
                    outliers_any = True
                    if _has_suspicious_max(valid_data):
                        print(f"     📢 Подозрительно большое значение: {int(valid_data.max()):,}")
                else:
                    if _has_suspicious_max(valid_data):
                        print(f"🔶 Подозрительно большое значение в {col_name}{col_desc}: {int(valid_data.max()):,}")
                        outliers_any = True

            if not outliers_any:
                print("✔️ Выбросов и подозрительных значений нет")

        # Экстремумы (IQR × extreme_iqr_multiplier)
        if detect_extremes:
            extremes_any = False
            for col in numeric_columns:
                col_name, col_desc = label_for_column(col, separator='•')
                valid_data = df[col].dropna()
                if len(valid_data) == 0 or valid_data.nunique() <= 20:
                    continue

                q1, q3 = valid_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - extreme_iqr_multiplier * iqr
                upper_bound = q3 + extreme_iqr_multiplier * iqr
                n_extremes = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if n_extremes > 0:
                    pct = n_extremes / n_rows * 100
                    print(f"💥 Экстремальные значения в {col_name}{col_desc}: {n_extremes} ({pct:.1f}%)")
                    extremes_any = True
                    if _has_suspicious_max(valid_data):
                        print(f"     📢 Подозрительно большое значение: {int(valid_data.max()):,}")
                else:
                    if _has_suspicious_max(valid_data):
                        print(f"💥 Подозрительно большое значение в {col_name}{col_desc}: {int(valid_data.max()):,}")
                        extremes_any = True

            if not extremes_any:
                print("✔️ Экстремальных значений и подозрительных максимумов нет")

        # Асимметрия
        skew_any = False
        for col in numeric_columns:
            col_name, col_desc = label_for_column(col, separator='•')
            valid_data = df[col].dropna()
            n = len(valid_data)
            
            if n == 0:
                continue
            
            skew_value = stats.skew(valid_data) if n > 2 else np.nan
            
            if pd.isna(skew_value) or abs(skew_value) <= 0.1:
                continue
                
            # Определяем направление и символ
            if skew_value > 0:
                if abs(skew_value) > 1.0:
                    symbol = "▶▶"
                    strength = "сильно"
                elif abs(skew_value) > 0.5:
                    symbol = "▶"
                    strength = ""
                else:
                    symbol = "▷"
                    strength = "слабо"
                direction = "правосторонняя"
            else:
                if abs(skew_value) > 1.0:
                    symbol = "◀◀"
                    strength = "сильно"
                elif abs(skew_value) > 0.5:
                    symbol = "◀"
                    strength = ""
                else:
                    symbol = "◁"
                    strength = "слабо"
                direction = "левосторонняя"
            
            # Формируем строку
            strength_text = f" {strength}" if strength else ""
            print(f"⚖️ Асимметрия в {col_name}{col_desc}: {skew_value:.2f} {symbol}{strength_text} {direction}")
            skew_any = True

        if not skew_any:
            print("✔️ Значимой асимметрии нет")


        # Почти константные признаки
        near_constant_any = False
        for col in numeric_columns:
            col_name, col_desc = label_for_column(col, separator='•')
            n_unique = df[col].nunique()
            if n_unique == 1:
                print(f"🔇 Почти константный признак {col_name}{col_desc}: все значения одинаковы")
                near_constant_any = True
            elif n_unique == 2 and len(df) > 10:
                top2_sum = df[col].value_counts().nlargest(2).sum()
                if top2_sum / len(df) > 0.99:
                    print(
                        f"🔇 Почти константный признак {col_name}{col_desc}: "
                        f"99%+ значений сосредоточено в двух категориях"
                    )
                    near_constant_any = True
        if not near_constant_any:
            print("✔️ Почти константных признаков нет")

        # Сильные корреляции
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            high_corr_found = False
            for i in range(len(numeric_columns)):
                for j in range(i + 1, len(numeric_columns)):
                    r = corr_matrix.iloc[i, j]
                    if 0.7 < abs(r) < 1.0:
                        col1, col2 = numeric_columns[i], numeric_columns[j]
                        col1_name, col1_desc = label_for_column(col1, separator='•')
                        col2_name, col2_desc = label_for_column(col2, separator='•')
                        print(f"🔗 Сильная корреляция: '{col1_name}'{col1_desc} ▸ {r:.3f} ◂ '{col2_name}'{col2_desc}")
                        high_corr_found = True
            if not high_corr_found:
                print("✔️ Сильных корреляций нет")
        else:
            print("✔️ Сильных корреляций нет (недостаточно числовых признаков)")
    else:
        print("✔️ Числовых признаков нет - пропускаем анализ распределений и корреляций")

    # Анализ категориальных признаков
    if categorical_columns:
        problem_lines = []
        clean_lines = []
        
        for col in categorical_columns:
            col_name, col_desc = label_for_column(col, separator='•')
            n_unique = df[col].nunique()
            n_total = len(df[col])
            issues = []

            # Проверяем непустые строки на разные проблемы
            non_null_series = df[col].dropna().astype(str)
            if len(non_null_series) > 0:
                # 1. Проверяем строки из одних пробелов
                whitespace_only_mask = non_null_series.str.strip().eq('')
                n_whitespace_only = whitespace_only_mask.sum()
                if n_whitespace_only > 0:
                    pct_whitespace = n_whitespace_only / n_total * 100
                    issues.append(f"строки из пробелов: {n_whitespace_only} ({pct_whitespace:.2f}%)")

                # 2. Проверяем мусорные значения (исключая уже учтённые пробелы)
                non_whitespace_series = non_null_series[~whitespace_only_mask]
                if len(non_whitespace_series) > 0:
                    junk_mask = non_whitespace_series.str.lower().isin(["null", "n/a", "nan", "none"])
                    n_junk = junk_mask.sum()
                    if n_junk > 0:
                        pct_junk = n_junk / n_total * 100
                        issues.append(f"мусорные значения ('null', 'n/a', etc.): {n_junk} ({pct_junk:.2f}%)")

            # 3. Проверяем другие проблемы
            if n_unique == 1:
                issues.append("только одно значение - можно удалить")
            elif n_total > 0:
                top_freq = df[col].value_counts().iloc[0]
                top_pct = top_freq / n_total * 100
                if top_pct > 95:
                    top_val = df[col].value_counts().index[0]
                    issues.append(f"сильный дисбаланс: '{top_val}' - {top_pct:.1f}%")

            full_name = f"{col_name}{col_desc}"
            if issues:
                problem_lines.append(f"⚠️ {full_name}: {', '.join(issues)}")
            else:
                clean_lines.append(f"💎 уникальных значений в {full_name}: {n_unique}")

        # Сначала выводим проблемы
        if problem_lines:
            for line in problem_lines:
                print(line)
        else:
            print("✔️ Проблем в категориальных признаках не обнаружено")

        # Потом - чистые категории (только если они есть)
        if clean_lines:
            for line in clean_lines:
                print(f'{line}')
    else:
        print("✔️ Категориальных признаков нет")

#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••



# dataset_overview: Даёт полный обзор структуры датафрейма и рекомендаций по подготовке к ML
def dataset_overview(
    df: pd.DataFrame,
    report: Literal["summary", "ML"] = "summary",
    show_rows: Optional[int] = None,
    cmap: str = "summer",
    max_unique_values: int = 10
) -> None:
    """
    Даёт полный обзор структуры датафрейма и рекомендаций по подготовке к ML.

    Описание:
        Расширенная версия dataset_profile с фокусом на практические рекомендации:
        - анализ дубликатов и пропусков,
        - классификация признаков (уникальные, бинарные, категориальные и т.д.),
        - советы по кодированию и обработке,
        - рекомендации по масштабированию для ML (в режиме "ML").
        Используется как финальная проверка перед началом EDA или ML.

    Параметры:
        df: pd.DataFrame - датафрейм для анализа
        report: Literal["summary", "ML"] - уровень детализации:
            - "summary": структура, дубликаты, пропуски, типы признаков,
            - "ML": как "summary" + колонка "Масштабирование" с рекомендациями
        show_rows: Optional[int] - показать случайную выборку указанного размера
        cmap: str - цветовая палитра для градиента уникальных значений

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука
    """
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    # Валидация параметра report
    if report not in ("summary", "ML"):
        raise ValueError(
            f"Некорректное значение report='{report}'. "
            f"Допустимые значения: 'summary', 'ML'"
        )
    
    df_name, df_desc = label_for_dataset(df, separator="•")

    print(f'Предварительный анализ датасета {df_name}\n')

    # Основная информация о датасете
    n_rows, n_cols = df.shape
    dataset_profile(df, report='summary')
    print()

    # Анализ дубликатов
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        total_rows = len(df)
        duplicate_ratio = duplicates / total_rows
        df_clean = df.drop_duplicates()
        unique_combos = len(df_clean)

        print(f"🚨 Повторяющиеся строки: {duplicates} из {total_rows} ({duplicate_ratio:.1%})")
        print(f"     🔢 Уникальных комбинаций признаков: {unique_combos}")

        # Проверка: появился ли уникальный идентификатор после удаления дубликатов?
        candidate_ids = []
        for col in df_clean.columns:
            # Проверка уникальности теперь работает для всех типов данных
            if df_clean[col].nunique() == len(df_clean) and df_clean[col].nunique() > 1:
                candidate_ids.append(col)

        if candidate_ids:
            ids_list = []
            for c in candidate_ids[:3]:
                col_name, col_desc = label_for_column(c, separator="()")
                ids_list.append(f"{col_name}{col_desc}")
            ids_str = ", ".join(ids_list)
            more = f" и ещё {len(candidate_ids) - 3}" if len(candidate_ids) > 3 else ""
            print(f"     🔍 После удаления дубликатов обнаружен потенциальный ID: {ids_str}{more}")
            print(f"     💡 Рекомендуется проверить эту колонку как уникальный идентификатор")
        elif duplicates == total_rows - 1 and unique_combos == 1:
            print(f"   ⚠️ Все строки идентичны - возможно, ошибка загрузки или пустой файл")
        elif duplicate_ratio > 0.5:
            print(f"   ⚠️ Высокая доля повторов (>50%) - вероятно, технические дубликаты")
            print(f"   🧹 Рекомендуется удалить")
        elif total_rows < 100:
            print(f"     💡 Датасет маленький ({total_rows} строк) - повторы могут быть валидными")
            print(f"     🧐 Проверьте контекст перед удалением")
        else:
            print(f"     💡 Повторы могут быть как ошибками, так и валидными данными")
            print(f"     🧐 Проверьте наличие уникального ID или контекст данных")

    else:
        print("✔️ Полных дубликатов строк не обнаружено")

    # Общие замечания по структуре
    if n_rows < 100:
        print(f"\n⚠️ ВНИМАНИЕ: датасет маленький ({n_rows} строк)")
    elif n_rows > 10000:
        print(f"\n🚀 ВНИМАНИЕ: датасет большой ({n_rows} строк)")

    # Анализ типов признаков и формирование рекомендаций
    info_data = []
    max_unique_values = 10  # можно вынести в параметр функции
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_total = len(df)
        ratio = n_unique / n_total

        # СБОР ПРОБЛЕМ (как у тебя)
        problems = []
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing_pct = n_missing / n_total * 100
            problems.append(f"🚨 пропуски: {n_missing} ({missing_pct:.1f}%)")

        if dtype == "object":
            whitespace_mask = df[col].astype(str).str.match(r'^\s*$') & df[col].notna()
            n_whitespace = whitespace_mask.sum()
            if n_whitespace > 0:
                whitespace_pct = n_whitespace / n_total * 100
                problems.append(f"⚠️ пробелы: {n_whitespace} ({whitespace_pct:.1f}%)")

            non_null = df[col].dropna().astype(str)
            junk_mask = non_null.str.lower().isin(['null', 'n/a', 'nan', 'none'])
            if junk_mask.any():
                problems.append("мусор: 'null', 'n/a' и подобное")

        if dtype in ("object", "category") and n_unique <= 50:
            top_freq = df[col].value_counts().iloc[0]
            top_pct = top_freq / len(df) * 100
            if top_pct > 95:
                problems.append(f"⚖️ дисбаланс: {top_pct:.1f}%")

        if dtype == "object" and n_unique > 50 and ratio < 0.8:
            problems.append("высокая кардинальность")

        if dtype == "object" and ratio > 0.95:
            problems.append(f"потенциально ID: {n_unique}/{n_total} уникальных")

        problems_str = " ".join(problems) if problems else ""

        # КЛАССИФИКАЦИЯ ПРИЗНАКОВ
        if n_unique == n_total:
            if dtype == "object":
                feature_type = "🆔 уникальный"
                recommendation = "использовать как 🆔"
            else:
                feature_type = "📏 уникальный количественный"
                recommendation = "уникальные значения - не подходит для модели"
        elif n_unique == 2:
            # Бинарный признак (важно!)
            feature_type = "💊 бинарный"
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0}:
                recommendation = "оставить как 0/1"
            elif unique_vals <= {'Male', 'Female'}:
                recommendation = "кодировать как 0/1"
            else:
                recommendation = "проверить значения и кодировать"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype in ['int8', 'int16', 'int32', 'int64'] and n_unique <= 20:
                feature_type = "🔢 дискретный"
                recommendation = "можно использовать как категориальный или числовой"
            else:
                feature_type = "🔢 непрерывный"
                recommendation = "использовать как есть, проверить масштабирование"
        elif dtype == "category":
            feature_type = "🏷️ категориальный"
            recommendation = "использовать как есть"
        elif dtype == "object":
            if n_unique <= 2:
                feature_type = "💊 бинарный"
                recommendation = "кодировать как 0/1"
            elif n_unique <= 20:
                feature_type = "🏷️ категориальный (низкая кардинальность)"
                recommendation = "преобразовать в категориальный"
            elif n_unique <= 50:
                feature_type = "🏷️ категориальный (средняя)"
                recommendation = "Target Encoding / CatBoost"
            else:
                feature_type = "📖 высококардинальный"
                recommendation = "Hashing, CatBoost, или NLP"
        else:
            feature_type = "❓ неизвестный"
            recommendation = "проверить тип данных вручную"

        # УНИКАЛЬНЫЕ ЗНАЧЕНИЯ (если мало)
        unique_vals_sample = df[col].dropna().unique()
        if len(unique_vals_sample) <= max_unique_values:
            try:
                unique_vals_sorted = sorted(unique_vals_sample, key=str)
            except:
                unique_vals_sorted = unique_vals_sample
            unique_vals_str = ", ".join(map(str, unique_vals_sorted))
        else:
            try:
                sample_vals = sorted(unique_vals_sample[:max_unique_values], key=str)
            except:
                sample_vals = unique_vals_sample[:max_unique_values]
            unique_vals_str = ", ".join(map(str, sample_vals)) + ", ..."

        # Эмодзи для типа данных
        dtype_emoji = {
            "int8": "1️⃣", "int16": "1️⃣", "int32": "1️⃣", "int64": "1️⃣",
            "uint8": "1️⃣", "uint16": "1️⃣", "uint32": "1️⃣", "uint64": "1️⃣",
            "float16": "🔢", "float32": "🔢", "float64": "🔢",
            "object": "📦", "datetime64[ns]": "📅", "category": "🏷️"
        }
        dtype_display = f"{dtype_emoji.get(dtype, '🚨')} {dtype}"

        # Получение описания колонки
        col_name, col_desc = label_for_column(col, separator='')

        # Рекомендация по масштабированию
        scaling_recommendation = "-"
        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 1:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                scaling_recommendation = "не требуется (бинарный)"
            else:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                has_outliers = n_outliers > 0.05 * n_rows

                min_val, max_val = df[col].min(), df[col].max()
                if min_val >= 0 and max_val <= 1 and (max_val - min_val) <= 1:
                    scaling_recommendation = "не требуется (уже в [0,1])"
                elif has_outliers:
                    scaling_recommendation = "стандартизация (осторожно: выбросы!)"
                elif (max_val - min_val) > 100:
                    scaling_recommendation = "стандартизация (широкий диапазон)"
                else:
                    scaling_recommendation = "нормализация или стандартизация"
        else:
            scaling_recommendation = "не применимо"

        info_data.append({
            "Колонка": col_name,
            "Описание": col_desc,
            "Уникальных": n_unique,
            "Тип данных": f"{dtype_emoji.get(dtype, '🚨')} {dtype}",
            "Тип признака": feature_type,
            "Проблемы": problems_str,
            "Значения": unique_vals_str,
            "Рекомендация": recommendation,
            "Масштабирование": scaling_recommendation
        })

    # Отображение таблицы
    info_df = pd.DataFrame(info_data)
    display_columns = list(info_df.columns) if report == "ML" else [col for col in info_df.columns if col not in ["Масштабирование","Рекомендация"]]

    display_table(
        info_df[display_columns],
        rows=len(info_df),
        float_precision=0,
        styler_func=lambda s: s.background_gradient(subset=["Уникальных"], cmap=cmap)
    )

    # Случайная выборка
    if show_rows is not None:
        sample_size = min(show_rows, n_rows)
        print(f"\n🎲 Случайные строки ({sample_size}) из датасета {df_name}:")
        display_table(df.sample(n=sample_size, random_state=42), max_header_length=8, rows=sample_size)
        print('')



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••




def handle_duplicates(
    df: pd.DataFrame,
    action: Literal["check", "clean"] = "check",
    id_col: Optional[str] = None,
    show_samples: int = 0
) -> Optional[pd.DataFrame]:
    """
    Универсальная очистка дубликатов с поддержкой ID-колонки, поиском кандидатов и проверкой конфликтов.

    Workflow:
        1. Удаляет полные дубликаты строк (все колонки, включая id).
        2. Если задан id_col:
            a. Удаляет дубликаты по id_col + признаки.
            b. Назначает id_col как индекс (только при action='clean').
            c. Проверяет конфликты: один id → разные значения в колонках.
            d. Удаляет дубликаты по признакам (индекс не учитывается).
        3. Если id_col не задан - ищет потенциальные идентификаторы (≥95% уникальности).

    Параметры:
        df : pd.DataFrame
        action : {"check", "clean"}, по умолчанию "check"
        id_col : Optional[str], по умолчанию None
        show_samples : int, по умолчанию 0

    Возвращает:
        pd.DataFrame - если action='clean'
        None - если action='check'
    """
    if df.empty:
        print("⚠️ Пустой датафрейм")
        return None

    dataset_profile(df, report='summary')
    current_df = df.copy()

    # ШАГ 1: Полные дубликаты строк (все колонки, включая id)
    total_before_full = len(current_df)
    dup_full = current_df.duplicated(keep=False)
    n_dup_full = dup_full.sum()
    if n_dup_full > 0:
        pct_full = n_dup_full / total_before_full * 100
        print(f"🕵️ Найдено полных дубликатов строк: {n_dup_full} ({pct_full:.3f}%)")
        if show_samples > 0:
            print(f"\n📋 Примеры полных дубликатов:")
            display_table(current_df[dup_full].head(show_samples), rows=show_samples)
        
        if action == "clean":
            current_df = current_df.drop_duplicates(keep='first').copy()
            print(f"✔️ Удалено полных дубликатов. Осталось {len(current_df)} строк\n")
    else:
        print("✔️ Полных дубликатов строк не найдено\n")

    # ШАГ 2: Работа с id_col
    final_id_col = id_col
    if id_col is not None:
        col_info = label_for_column(id_col, separator='•', format="string")
        if id_col not in current_df.columns:
            print(f"⚠️ Колонка {id_col} не найдена - пропускаем шаг с ID\n")
            final_id_col = None
        else:
            #2a: Дубликаты по id_col (без учёта признаков)
            total_before_id = len(current_df)
            n_unique_ids = current_df[id_col].nunique()
            n_total_ids = len(current_df)
            n_dup_ids = n_total_ids - n_unique_ids
            
            if n_dup_ids > 0:
                pct_dup_ids = n_dup_ids / total_before_id * 100
                print(f"🕵️ Найдено дубликатов по {col_info} (повторяющиеся ID): {n_dup_ids} ({pct_dup_ids:.3f}%)")
                if show_samples > 0:
                    duplicated_ids = current_df[current_df.duplicated(subset=[id_col], keep=False)][id_col].unique()[:show_samples]
                    sample_df = current_df[current_df[id_col].isin(duplicated_ids)].head(show_samples * 2)
                    print(f"\n📋 Примеры строк с повторяющимися ID:")
                    display_table(sample_df.reset_index(drop=True), rows=len(sample_df))
                
                if action == "clean":
                    current_df = current_df.drop_duplicates(subset=[id_col], keep='first').copy()
                    print(f"✔️ Удалены дубликаты по {col_info}. Осталось {len(current_df)} строк\n")
            else:
                print(f"✔️ Дубликатов по {col_info} (повторяющиеся ID) не найдено\n")

            #2b: Назначаем индекс ТОЛЬКО если action="clean"
            if action == "clean":
                current_df = current_df.set_index(id_col)
                print(f"🆔 Колонка {col_info} назначена как индекс\n")
            else:
                print(f"🔍 Режим check: анализ конфликтов по {col_info} без изменения индекса\n")

            #2c: Проверка конфликтов по ID (анализируется всегда, но по-разному для check/clean)
            print("🕵️ Проверка конфликтов по ID (один ID → разные значения):")
            conflict_cols = []
            total_ids = current_df[id_col].nunique() if action == "check" else current_df.index.nunique()

            for col in current_df.columns:
                if action == "check":
                    # Для check: используем оригинальную колонку id
                    n_conflict_ids = (current_df.groupby(id_col)[col].nunique() > 1).sum()
                else:
                    # Для clean: индекс уже id_col
                    n_conflict_ids = (current_df.groupby(current_df.index)[col].nunique() > 1).sum()

                if n_conflict_ids > 0:
                    conflict_cols.append((col, n_conflict_ids))

            if conflict_cols:
                print(f"   ⚠️ Найдены конфликты в {len(conflict_cols)} колонках:")
                for col, n_ids in conflict_cols:
                    pct_conflict = n_ids / total_ids * 100 if total_ids > 0 else 0
                    print(f"      • {col}: {n_ids} ID ({pct_conflict:.3f}%) имеют разные значения")

                if show_samples > 0:
                    print(f"\n📋 Примеры конфликтов:")
                    example_rows = []
                    for col, _ in conflict_cols[:2]:
                        if action == "check":
                            conflict_ids = current_df.groupby(id_col)[col].nunique()
                            conflict_ids = conflict_ids[conflict_ids > 1].index[:show_samples]
                            for id_val in conflict_ids:
                                examples = current_df[current_df[id_col] == id_val]
                                if len(examples) <= 5:
                                    example_rows.append(examples)
                        else:
                            conflict_ids = current_df.groupby(current_df.index)[col].nunique()
                            conflict_ids = conflict_ids[conflict_ids > 1].index[:show_samples]
                            for id_val in conflict_ids:
                                examples = current_df.loc[[id_val]]
                                if len(examples) <= 5:
                                    example_rows.append(examples)
                        if len(example_rows) >= show_samples:
                            break

                    if example_rows:
                        examples_df = pd.concat(example_rows).head(show_samples * 3)
                        display_table(examples_df.reset_index(drop=(action == "clean")), rows=len(examples_df))

                # --- Только для clean: обработка конфликтов ---
                if action == "clean":
                    print(f"   🧹 Обработка конфликтов: оставляем первое значение для каждого ID...")
                    current_df = current_df.groupby(current_df.index).first().copy()
                    print(f"   ✔️ Конфликты разрешены. Осталось {len(current_df)} строк\n")
            else:
                print("    ✔️ Конфликтов по ID не обнаружено")
            print()

            #2d: Дубликаты по признакам (индекс не учитывается)
            total_before_features = len(current_df)
            dup_features = current_df.duplicated(keep=False)
            n_dup_features = dup_features.sum()
            if n_dup_features > 0:
                pct_features = n_dup_features / total_before_features * 100
                print(f"🕵️ Найдено дубликатов по признакам: {n_dup_features} ({pct_features:.3f}%)")
                if show_samples > 0:
                    print(f"\n📋 Примеры:")
                    display_df = current_df[dup_features].reset_index().head(show_samples)
                    display_table(display_df, rows=len(display_df))
                
                if action == "clean":
                    current_df = current_df.drop_duplicates(keep='first').copy()
                    print(f"    ✔️ Удалено дубликатов по признакам. Осталось {len(current_df)} строк\n")
            else:
                print("✔️ Дубликатов по признакам не найдено\n")

    # ШАГ 3: Поиск потенциальных идентификаторов (если id_col не задан)
    if final_id_col is None:
        print("🕵️ Поиск потенциальных идентификаторов...")
        total = len(current_df)
        
        if total == 0:
            print("      • Датасет пуст\n")
        else:
            candidates = []
            for col in current_df.columns:
                n_uniq = current_df[col].nunique()
                pct_unique = n_uniq / total * 100

                # Пропускаем float-колонки с нецелыми значениями
                if pd.api.types.is_float_dtype(current_df[col]):
                    non_null = current_df[col].dropna()
                    if len(non_null) > 0 and not (non_null % 1 == 0).all():
                        continue  # Нецелые float - плохой кандидат на ID

                if pct_unique >= 95.0:
                    # Приоритет 1: имя колонки
                    name_score = 1.0 if any(k in col.lower() for k in ['id', 'key', 'code', 'uid']) else 0.0
                    # Приоритет 2: тип данных
                    type_score = 1.0 if current_df[col].dtype in ['object', 'int64', 'int32'] else 0.5
                    # Приоритет 3: уникальность
                    unique_score = pct_unique / 100

                    score = name_score * 3 + type_score * 2 + unique_score  # взвешенная сумма

                    candidates.append({
                        'col': col,
                        'n_unique': n_uniq,
                        'pct_unique': pct_unique,
                        'score': score
                    })

            if candidates:
                # Сортируем по скору
                best = max(candidates, key=lambda x: x['score'])
                status = "⚠️ (не уникален!)" if best['n_unique'] < total else ""
                print(f"     💎 Найден потенциальный идентификатор: '{best['col']}' "
                      f"({best['n_unique']} уникальных, {best['pct_unique']:.3f}%){status}")
                if best['n_unique'] < total:
                    print(f"     📌 Колонка содержит дубликаты - проверьте данные.")
                # --- Сообщение пользователю ---
                print(f"💡 Рекомендация: используйте handle_duplicates(df, id_col='{best['col']}', action='clean') для очистки по ID.")
            else:
                print("     💎 Потенциальных идентификаторов не найдено\n")

    # Возврат результата
    if action == "clean":
        return current_df
    else:
        return None


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# audit_numerical: Анализирует качество числовых колонок в одном датафрейме и выводит отчёт в стиле EDA.
def audit_numerical(
    df: pd.DataFrame,
    report: Literal["summary", "full"] = "full",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    cmap: str = "Oranges"
) -> None:
    """
    Проводит аудит качества числовых признаков в датафрейме.
    
    Описание:
        Выявляет типичные проблемы в числовых колонках:
        - пропуски (>0%),
        - выбросы (>5% по IQR),
        - сильную асимметрию (|skewness| > 1.5),
        - почти константные признаки,
        - подозрительно большие максимумы.
        Выводит интерпретируемый отчёт и сводную таблицу метрик.

    Параметры:
        df: pd.DataFrame - датафрейм для аудита
        report: Literal["summary", "full"] - уровень детализации:
            - "summary": только чеклист проблем,
            - "full": + сводная таблица,
        include: Optional[List[str]] - анализировать ТОЛЬКО эти колонки
        exclude: Optional[List[str]] - исключить эти колонки из анализа
        cmap: str - цветовая палитра для градиента в таблице

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука
    """

     # Валидация параметра report
    if report not in ("summary", "full"):
        raise ValueError(
            f"Некорректное значение report='{report}'. "
            f"Допустимые значения: 'summary', 'full'"
        )

    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    # 1. Определяем числовые колонки с фильтрацией
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if include is not None:
        numeric_cols = [col for col in numeric_cols if col in include]
    if exclude is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    if not numeric_cols:
        print("✔️ Нет числовых колонок для анализа")
        return

    # 2. Получаем описание датафрейма
    df_name, df_desc = label_for_dataset(df, separator="•")
    df_label = f"{df_name}{df_desc}" if df_desc else df_name

    print("Анализ качества числовых данных\n")
    print(f"🗃️ Датафрейм: {df_label}")
    print(f"🔢 Числовых колонок: {len(numeric_cols)}\n")

    print(f"📋 Чеклист:")
    issues_found = False
    all_metric_records = []

    for col in sorted(numeric_cols):
        col_name, col_desc = label_for_column(col, separator="•")
        full_col_name = f"{col_name}{col_desc}"

        series = df[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0

        clean_series = series.dropna()
        if clean_series.empty:
            outliers_pct = 0.0
            skewness = np.nan
        else:
            # Выбросы по IQR
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
            n_outliers = outlier_mask.sum()
            outliers_pct = (n_outliers / n_total * 100) if n_total > 0 else 0
            
            # Асимметрия (skewness)
            skewness = clean_series.skew()

        # Определяем проблемы через единый детектор
        issue_details = _detect_numerical_issues(series, n_total)
        has_issues = len(issue_details) > 0

        if has_issues:
            issues_found = True
            print(f"    🚨 {full_col_name}")
            print(f"         📌 {' • '.join(issue_details)}")
        else:
            print(f"    ✔️ {full_col_name} 💎 качество данных хорошее")

        # Определяем тип асимметрии для интерпретации
        if pd.isna(skewness):
            skew_type = "Н/Д"
        elif abs(skewness) <= 0.5:
            skew_type = "≈ симметричная"
        elif skewness > 0.5:
            skew_type = "правосторонняя ▶"
        else:  # skewness < -0.5
            skew_type = "◀ левосторонняя"

        # Сохраняем метрики для таблицы
        all_metric_records.append({
            "Колонка": full_col_name,
            "Пропуски (%)": missing_pct,
            "Выбросы (%)": outliers_pct,
            "Асимметрия": skewness if not pd.isna(skewness) else np.nan,
            "Тип асимметрии": skew_type,  # ← новая колонка
            "Среднее": clean_series.mean() if not clean_series.empty else np.nan,
            "Медиана": clean_series.median() if not clean_series.empty else np.nan,
            "Std": clean_series.std() if not clean_series.empty else np.nan,
            "Минимум": clean_series.min() if not clean_series.empty else np.nan,
            "Максимум": clean_series.max() if not clean_series.empty else np.nan
        })

    if not issues_found:
        print("\n✔️ Все числовые колонки прошли проверку качества!")
        return

    # 3. Сводная таблица
    if report in ("full") and all_metric_records:
        print(f"\n📋 Сводная таблица качества (всего: {len(all_metric_records)} колонок):")
        quality_df = pd.DataFrame(all_metric_records)

        display_table(
            quality_df,
            rows=len(quality_df),
            float_precision=3,
            max_header_length = 1000,
            styler_func=lambda s: s.background_gradient(subset=["Выбросы (%)", "Асимметрия"], cmap=cmap,
            #low=0.1,   # даже минимальные значения будут средней яркости
            #high=0.3   # максимальные - тёмные, но не чёрные
            )
        )
        

#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# report_numerical_consistency: Аудит распределения и статистических различий числовых признаков между несколькими датафреймами и выводит отчёт в стиле EDA.
def report_numerical_consistency(
    dataframes: List[pd.DataFrame],
    report: Literal["min", "cols", "full"] = "full",
    plot: bool = False 
) -> None:
    """
    Аудит распределения и статистических различий числовых признаков между несколькими датафреймами

    Позволяет выявить отклонения в распределении, наличие выбросов и статистически значимые различия между источниками данных.
    Используется для проверки согласованности данных в разных выборках, например, в A/B-тестах или при объединении данных из разных источников.

    Параметры:
        dataframes: list[pd.DataFrame] - список датафреймов для сравнения
        col: str - имя колонки для анализа
        dataset_labels: list[str] - метки для каждого датафрейма
        plot: bool - флаг для включения визуализации и статистических тестов
        metric: str - тип метрики для отображения (например, "Выбросы (%)", "Пропуски (%)")

    Возвращаемое значение:
        pd.DataFrame - таблица с метриками для каждого датафрейма
    """
    if not dataframes:
        print("⚠️ Нет датафреймов для анализа")
        return

    # 1. Определяем метки датасетов
    dataset_labels = []
    for df in dataframes:
        name, desc = label_for_dataset(df, separator="•")
        label = f"{name}{desc}"
        dataset_labels.append(label)

    # 2. Находим общие ЧИСЛОВЫЕ колонки
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns &= set(df.columns)
    
    if not common_columns:
        print("🔍 Нет общих колонок между датафреймами")
        return

    numerical_common = set()
    for col in common_columns:
        is_numeric = all(
            pd.api.types.is_numeric_dtype(df[col])
            for df in dataframes
        )
        if is_numeric:
            numerical_common.add(col)

    if not numerical_common:
        print("✔️ Нет общих числовых колонок для анализа")
        return

    # 3. Подготовка к анализу
    print("📊 Анализ согласованности числовых данных между датафреймами\n")

    EMOJI_NUMBERS = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    print("🗃️ Источники данных:")
    for i, label in enumerate(dataset_labels):
        emoji_num = EMOJI_NUMBERS[i+1] if i+1 < len(EMOJI_NUMBERS) else f"{i+1}"
        print(f"{emoji_num} {label}")
    
    print(f'\n📋 Чеклист:')
    all_metric_records = []
    issues_found = False

    # 4. Анализ каждой числовой колонки
    for col in sorted(numerical_common):
        col_name, col_desc = label_for_column(col, separator="•")
        full_col_name = f"{col_name}{col_desc}"

        # Собираем статистики по каждому датафрейму
        stats_per_df = []
        has_issues = False

        for i, df in enumerate(dataframes):
            series = df[col]
            n_total = len(series)
            n_missing = series.isna().sum()
            missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0

            # Работаем только с непропущенными значениями
            clean_series = series.dropna()
            if clean_series.empty:
                stats = {
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "missing_pct": missing_pct,
                    "outliers_pct": 0.0
                }
            else:
                # Выбросы по методу IQR
                Q1 = clean_series.quantile(0.25)
                Q3 = clean_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
                n_outliers = outlier_mask.sum()
                outliers_pct = (n_outliers / n_total * 100) if n_total > 0 else 0

                stats = {
                    "mean": clean_series.mean(),
                    "median": clean_series.median(),
                    "std": clean_series.std(),
                    "min": clean_series.min(),
                    "max": clean_series.max(),
                    "missing_pct": missing_pct,
                    "outliers_pct": outliers_pct
                }

            stats_per_df.append(stats)

            # Сохраняем для сводной таблицы
            all_metric_records.append({
                "Колонка": full_col_name,
                "Метрика": "Среднее",
                "Источник": dataset_labels[i],
                "Значение": stats["mean"]
            })
            all_metric_records.append({
                "Колонка": full_col_name,
                "Метрика": "Медиана",
                "Источник": dataset_labels[i],
                "Значение": stats["median"]
            })
            all_metric_records.append({
                "Колонка": full_col_name,
                "Метрика": "Std",
                "Источник": dataset_labels[i],
                "Значение": stats["std"]
            })
            all_metric_records.append({
                "Колонка": full_col_name,
                "Метрика": "Пропуски (%)",
                "Источник": dataset_labels[i],
                "Значение": stats["missing_pct"]
            })
            all_metric_records.append({
                "Колонка": full_col_name,
                "Метрика": "Выбросы (%)",
                "Источник": dataset_labels[i],
                "Значение": stats["outliers_pct"]
            })

        # Определяем, есть ли проблемы
        means = [s["mean"] for s in stats_per_df if pd.notna(s["mean"])]
        medians = [s["median"] for s in stats_per_df if pd.notna(s["median"])]

        if means and medians:
            mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
            median_cv = np.std(medians) / np.mean(medians) if np.mean(medians) != 0 else 0

            outlier_pcts = [s["outliers_pct"] for s in stats_per_df]
            max_outliers = max(outlier_pcts) if outlier_pcts else 0
            min_outliers = min(outlier_pcts) if outlier_pcts else 0

            if mean_cv > 0.2 or median_cv > 0.2 or (max_outliers > 5 and max_outliers - min_outliers > 10):
                has_issues = True
                issues_found = True

        if has_issues:
            print(f"🚨 {full_col_name} 📢 обнаружены расхождения")
            for i, stats in enumerate(stats_per_df):
                if pd.notna(stats["mean"]):
                    print(f"     📌 {dataset_labels[i][:20]}: среднее={stats['mean']:.2f}, выбросы={stats['outliers_pct']:.1f}%")
        else:
            print(f"✔️ {full_col_name} 💎 распределения схожи")

    if not issues_found and not all_metric_records:
        print("\n✔️ Все числовые колонки полностью согласованы!")
        return

    # 5. Вывод результатов
    if report in ("cols", "full") and all_metric_records:
        print(f"\n📋 Сводная таблица метрик (всего: {len(all_metric_records)} записей):")
        metrics_df = pd.DataFrame(all_metric_records)
        try:
            pivot_df = metrics_df.pivot_table(
                index=["Колонка", "Метрика"],
                columns="Источник",
                values="Значение",
                aggfunc="first"
            ).reset_index()
            display_table(pivot_df, rows=20, max_header_length=20)
        except Exception:
            display_table(metrics_df, rows=15, max_header_length=20)

    # 6. Детальный анализ (матрицы и визуализация)
    if report == "full":
        print(f"\nДетальный анализ по проблемным колонкам:")
        for col in sorted(numerical_common):
            col_name, col_desc = label_for_column(col, separator="•")
            full_col_name = f"{col_name}{col_desc}"

            # Матрица метрик
            matrix_rows = []
            metrics = ["Среднее", "Медиана", "Std", "Min", "Max", "Пропуски (%)", "Выбросы (%)"]
            
            for metric in metrics:
                row = {"Метрика": metric}
                for i, df in enumerate(dataframes):
                    series = df[col].dropna()
                    if series.empty:
                        val = "-"
                    else:
                        if metric == "Среднее":
                            val = series.mean()
                        elif metric == "Медиана":
                            val = series.median()
                        elif metric == "Std":
                            val = series.std()
                        elif metric == "Min":
                            val = series.min()
                        elif metric == "Max":
                            val = series.max()
                        elif metric == "Пропуски (%)":
                            missing_pct = (df[col].isna().sum() / len(df) * 100)
                            val = missing_pct
                        elif metric == "Выбросы (%)":
                            Q1 = series.quantile(0.25)
                            Q3 = series.quantile(0.75)
                            IQR = Q3 - Q1
                            bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                            outliers = series[(series < bounds[0]) | (series > bounds[1])]
                            val = (len(outliers) / len(df) * 100)
                        else:
                            val = "-"
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            if metric in ["Пропуски (%)", "Выбросы (%)"]:
                                val = f"{val:.1f}"
                            else:
                                val = f"{val:.2f}"
                    row[dataset_labels[i]] = val
                matrix_rows.append(row)

            print(f"\n🎹 Колонка: {full_col_name}")
            matrix_df = pd.DataFrame(matrix_rows)
            display_table(matrix_df, rows=len(matrix_rows), max_header_length=25)

            # Визуализация и статистика (ТОЛЬКО если plot=True и >=2 датафреймов)
            if plot and len(dataframes) >= 2:
                try:
                    from scipy.stats import probplot, levene, ttest_ind, f_oneway

                    # Собираем данные для визуализации
                    plot_data = []
                    groups = []
                    for i, df in enumerate(dataframes):
                        clean_vals = df[col].dropna()
                        groups.append(clean_vals.values)
                        for val in clean_vals:
                            plot_data.append({
                                "Значение": val,
                                "Источник": dataset_labels[i]
                            })
                    
                    if not plot:
                        continue

                    plot_df = pd.DataFrame(plot_data)
                    
                    # 1. Гистограммы + KDE
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 2, 1)
                    sns.histplot(
                        data=plot_df,
                        x="Значение",
                        hue="Источник",
                        kde=True,
                        alpha=0.6,
                        stat="count"
                    )
                    plt.title(f"Распределение: {full_col_name}", fontsize=11)
                    plt.xlabel("Значение")
                    plt.ylabel("Количество")
                    plt.grid(True, linestyle='--', alpha=0.5)

                    # 2. QQ-plot
                    plt.subplot(1, 2, 2)
                    colors = sns.color_palette("husl", len(dataframes))
                    for i, (df_label, group) in enumerate(zip(dataset_labels, groups)):
                        if len(group) > 0:
                            probplot(group, dist="norm", plot=plt)
                            lines = plt.gca().get_lines()
                            if lines:
                                lines[-1].set_color(colors[i])
                                lines[-2].set_color(colors[i])
                    plt.title("QQ-plot (нормальность)", fontsize=11)
                    plt.legend([label[:15] for label in dataset_labels], fontsize=8)

                    plt.tight_layout()
                    plt.show()

                    # 3. Статистические тесты
                    valid_groups = [g for g in groups if len(g) > 1]
                    if len(valid_groups) >= 2:
                        # Levene's test
                        w_stat, p_levene = levene(*valid_groups, center='median')
                        print(f"     📊 Levene’s test (равенство дисперсий): p = {p_levene:.4f}")
                        if p_levene < 0.05:
                            print(f"        ⚠️  Дисперсии значимо различаются (гетероскедастичность)")
                        else:
                            print(f"        ✔️ Дисперсии статистически равны (гомоскедастичность)")

                        # t-test или ANOVA
                        if len(valid_groups) == 2:
                            t_stat, p_ttest = ttest_ind(valid_groups[0], valid_groups[1], equal_var=False)
                            print(f"     📊 Welch’s t-test (средние): p = {p_ttest:.4f}")
                            p_vals = [p_levene, p_ttest]
                        else:
                            f_stat, p_anova = f_oneway(*valid_groups)
                            print(f"     📊 ANOVA (средние): p = {p_anova:.4f}")
                            p_vals = [p_levene, p_anova]
                        
                        if any(p < 0.05 for p in p_vals):
                            print(f"        ⚠️  Обнаружены статистически значимые различия")
                        else:
                            print(f"        ✔️ Статистически значимых различий не обнаружено")

                except Exception as e:
                    print(f"     ❌ Ошибка визуализации/статистики: {str(e)}")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# audit_categorical: Проводит аудит согласованности значений в категориальных признаках
def audit_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_distance: int = 2,
    min_frequency: int = 1,
    cmap: str = "Oranges"
) -> None:
    """
    Проводит аудит согласованности значений в категориальных признаках.
    
    Описание:
        Выявляет группы похожих значений (расстояние Левенштейна ≤ max_distance),
        которые, вероятно, являются несогласованной записью одной сущности:
        - опечатки ("Москва" vs "Москава"),
        - разный регистр ("Apple" vs "apple"),
        - сокращения ("St." vs "Street").
        Автоматически исключает колонки, похожие на числовые.
        Выводит интерпретируемый отчёт и сводную таблицу проблемных групп.

    Параметры:
        df: pd.DataFrame - датафрейм для аудита
        columns: Optional[List[str]] - колонки для анализа; если None - все категориальные
        max_distance: int - максимальное расстояние Левенштейна для группировки (по умолчанию 2)
        min_frequency: int - минимальная частота значения для участия в анализе (по умолчанию 1)
        cmap: str - цветовая палитра для градиента в таблице

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука
    """
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    # Определяем колонки для анализа
    if columns is None:
        columns = []
        for col in df.columns:
            # Рассматриваем только object и category
            if df[col].dtype.name in ('object', 'category'):
                # Исключаем колонки, похожие на числовые
                if not _is_likely_numeric(df[col]):
                    columns.append(col)

    # Фильтруем существующие колонки
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        print("✔️ Нет категориальных колонок для анализа согласованности")
        return

    # Получаем описание датафрейма
    df_name, df_desc = label_for_dataset(df, separator="•")
    df_label = f"{df_name}{df_desc}" if df_desc else df_name

    print("Анализ согласованности категориальных данных\n")
    print(f"🗃️ Датафрейм             : {df_label}")
    print(f"🏷️ Категориальных колонок: {len(columns)}\n")

    print("📋 Чеклист:")
    issues_found = False
    all_typo_records = []

    for col in sorted(columns):
        col_name, col_desc = label_for_column(col, separator="•")
        full_col_name = f"{col_name}{col_desc}"

        # Получаем частоты значений
        value_counts = df[col].dropna().astype(str).value_counts()
        value_counts = value_counts[value_counts >= min_frequency]
        
        has_issues = False
        typo_groups = []

        if len(value_counts) > 1:
            # Нормализуем значения для сравнения
            normalized_to_original = {}
            for val, freq in value_counts.items():
                normalized = _normalize_text(str(val))
                if normalized not in normalized_to_original:
                    normalized_to_original[normalized] = []
                normalized_to_original[normalized].append((val, freq))

            # Находим похожие нормализованные значения
            normalized_values = list(normalized_to_original.keys())
            used = set()
            
            for i, norm_val1 in enumerate(normalized_values):
                if norm_val1 in used:
                    continue
                    
                current_group = normalized_to_original[norm_val1].copy()
                used.add(norm_val1)
                
                # Сравниваем со всеми последующими
                for j in range(i + 1, len(normalized_values)):
                    norm_val2 = normalized_values[j]
                    if norm_val2 in used:
                        continue
                    if _levenshtein_distance(norm_val1, norm_val2) <= max_distance:
                        current_group.extend(normalized_to_original[norm_val2])
                        used.add(norm_val2)
                
                # Если группа содержит более одного оригинального значения - проблема
                if len(set(orig for orig, _ in current_group)) > 1:
                    has_issues = True
                    typo_groups.append(current_group)

        if has_issues:
            issues_found = True
            total_problematic = sum(len(group) for group in typo_groups)
            print(f"    🚨 {full_col_name} 📢 несогласованных значений: {total_problematic}")
            
            # Выводим до 3 групп
            for group in typo_groups[:3]:
                originals = [f"{orig}" for orig, freq in group]
                print(f"         📄 {originals}")
            
            # Собираем данные для таблицы
            for group in typo_groups:
                for orig, freq in group:
                    all_typo_records.append({
                        "Колонка": full_col_name,
                        "Значение": orig,
                        "Частота": freq,
                        "Группа": ", ".join([f"{o}" for o, f in group])
                    })
        else:
            print(f"    ✔️ {full_col_name} 💎 качество данных хорошее")

    if not issues_found:
        print("\n✔️ Все категориальные колонки прошли проверку на согласованность!")
        return

    # Сводная таблица
    print(f"\n📋 Сводная таблица несогласованных значений (всего: {len(all_typo_records)}):")
    typo_df = pd.DataFrame(all_typo_records)
    display_table(
        typo_df, 
        rows=15, 
        max_header_length=25, 
        styler_func=lambda s: s.background_gradient(subset=["Частота"], cmap=cmap)
    )
    
    print("\n🛠️ Рекомендации:")
    print("     • Проверьте группы похожих значений на несогласованность")
    print("     • Рассмотрите унификацию написания через словарь замен")



# 3_analyze_category_frequencies • Анализирует частоты значений в категориальной колонке.
def audit_categorical_frequencies (
    df: pd.DataFrame,
    col: str,
    cmap: str = 'YlGn',
    show_dataset_info: bool = True,
    force_categorical: bool = True,
    sort_by_value: Optional[Literal["asc", "desc", None]] = None 
) -> None:
    """  Optional[Literal["asc", "desc"]]
    Анализирует частоты значений в категориальной колонке и выводит интерактивную таблицу.
    
    Описание:
        Функция строит таблицу с частотным анализом категориальной колонки, включая:
        - Значение категории
        - Количество строк с этим значением
        - Процент от общего числа строк
        
        Таблица отображается через display_table с цветовой градиентной подсветкой
        по процентному столбцу. Поддерживает анализ числовых колонок как категорий.
    
    Особенности:
        • Автоматическое определение категориальных колонок (по типу и числу уникальных значений)
        • Цветовая подсветка процентов через cmap (YlGn, Reds, viridis и др.)
        • Сортировка: по частоте (по умолчанию), по значению (возрастание/убывание)
        • Интеграция с глобальными справочниками (DATASET_DESCRIPTIONS, COLUMN_DESCRIPTIONS)
        • Защита от анализа не-категориальных колонок (с возможностью принудительного анализа)
    
    Параметры:
        df: pd.DataFrame - целевой датафрейм
        col: str - имя анализируемой колонки
        cmap: str - цветовая карта для градиентной подсветки (по умолчанию 'YlGn')
        show_dataset_info: bool - показывать информацию о датасете (по умолчанию False)
        force_categorical: bool - анализировать колонку как категориальную, даже если она числовая (по умолчанию False)
        sort_by_value: Literal["asc", "desc", None] - порядок сортировки:
            - "asc": по возрастанию значений
            - "desc": по убыванию значений  
            - None: по убыванию частоты (по умолчанию)
    
    Возвращаемое значение:
        None (выводит отчёт через display_table)
    
    Примеры:
        >>> audit_categorical_frequencies (df, "breed")
        >>> audit_categorical_frequencies (df, "age_category", sort_by_value="asc")
        >>> audit_categorical_frequencies (df, "numeric_code", force_categorical=True)
    """

    # Валидация параметра sort_by_value
    if sort_by_value not in ("asc", "desc", None):
        raise ValueError(
            f"Некорректное значение sort_by_value='{sort_by_value}'. "
            f"Допустимые значения: 'asc', 'desc'"
        )

    # Проверка колонки
    if col not in df.columns:
        print(f"❌ Колонка '{col}' не найдена в DataFrame.")
        return None
    
    series = df[col]
    n_unique = series.nunique()
    n_total = len(series)

    # Авто-определение: похоже ли на категорию?
    is_categorical_by_type = pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)
    is_few_unique = n_unique <= 25 or (n_unique / n_total) < 0.05
    is_likely_categorical = is_categorical_by_type or (pd.api.types.is_numeric_dtype(series) and is_few_unique)

    if not force_categorical and not is_likely_categorical:
        print(f"⚠️ Колонка '{col}' (тип: {series.dtype}) не похожа на категориальную. "
              f"Уникальных значений: {n_unique} из {n_total}. "
              f"Используйте force_categorical=True, чтобы проигнорировать проверку.")
        return None

    # Получение метаданных датасета
    dataset_key, dataset_desc = label_for_dataset(df, separator='•')

    # расчёт частот
    value_counts = series.value_counts(sort=True, ascending=False)  # Сначала - по частоте
    total_rows = len(df)

    # Формирование таблицы
    result = pd.DataFrame({
        'Значение': value_counts.index,
        'Количество строк': value_counts.values,
        'Процент от общего числа строк': (value_counts / total_rows * 100).round(3)
    }).reset_index(drop=True)

    # Сортировка по значению (опционально)
    if sort_by_value == 'asc':
        result = result.sort_values(by='Значение', ascending=True).reset_index(drop=True)
    elif sort_by_value == 'desc':
        result = result.sort_values(by='Значение', ascending=False).reset_index(drop=True)
    # Если sort_by_value == None - остаётся сортировка по частоте (как было)

    # Информация о датасете
    if show_dataset_info:
        print(f"🗃️ Датасет '{dataset_key}'{dataset_desc}")
        n_rows, n_cols = df.shape
        memory_kb = df.memory_usage(deep=True).sum() / 1024

    col, desc = label_for_column(col, separator="•")

    # Вывод заголовка
    header = f"🎹 Частота значений '{col}'{desc}"
    print(header)
    print(f"📐 Общее количество строк {total_rows:,} × {n_unique} групп")

    if not is_likely_categorical:
        print(f"\n⚠️ Колонка '{col}' (тип: {series.dtype}) не похожа на категориальную.")

    # Визуализация
    max_pct = result['Процент от общего числа строк'].max()

    display_table(
        result,
        rows=len(result),
        float_precision=3,
        max_header_length = 1000,
        styler_func=lambda s: s.background_gradient(subset=["Процент от общего числа строк"], cmap=cmap)
    )

    return None


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# audit_categorical_cross: Проводит кросс-аудит согласованности категориальных признаков между датафреймами
def audit_categorical_cross(
    dataframes: List[pd.DataFrame],
    report: Literal["min", "diff", "full"] = "full",
) -> None:
    """
    Проводит кросс-аудит согласованности категориальных признаков между датафреймами.
    
    Описание:
        Сравнивает категориальные колонки, присутствующие хотя бы в двух датафреймах:
        - выявляет значения, присутствующие не во всех источниках,
        - находит несогласованную запись одной сущности (опечатки, регистр, сокращения),
        - показывает, в каких датафреймах встречаются расхождения.
        Использует глобальные справочники DATASET_DESCRIPTIONS и COLUMN_DESCRIPTIONS
        для автоматической подписи источников и признаков.

    Параметры:
        dataframes: List[pd.DataFrame] - список датафреймов для сравнения
        report: Literal["min", "diff", "full"] - уровень детализации:
            - "min": только сводка и рекомендации,
            - "diff": + таблица расхождений,
            - "full": + детальные матрицы по каждой проблемной колонке

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука
    """
    # Валидация параметра report
    if report not in ("min", "diff", "full"):
        raise ValueError(
            f"Некорректное значение report='{report}'. "
            f"Допустимые значения: 'min', 'diff', 'full'"
        )
    
    if not dataframes:
        print("⚠️ Нет датафреймов для анализа")
        return

    # Проверка на пустые датафреймы
    non_empty_dfs = [df for df in dataframes if not df.empty]
    if not non_empty_dfs:
        print("⚠️ Все датафреймы пустые")
        return
    dataframes = non_empty_dfs

    # 1. Определяем метки датасетов
    dataset_labels = []
    for df in dataframes:
        name, desc = label_for_dataset(df, separator="•")
        label = f"{name}{desc}"
        dataset_labels.append(label)

    # 2. Собираем, в каких датафреймах есть каждая колонка
    col_to_dfs: DefaultDict[str, List[Tuple[int, pd.DataFrame]]] = defaultdict(list)
    for idx, df in enumerate(dataframes):
        for col in df.columns:
            col_to_dfs[col].append((idx, df))

    # Выбираем колонки, присутствующие хотя бы в двух датафреймах
    candidate_cols = {col for col, info in col_to_dfs.items() if len(info) >= 2}

    if not candidate_cols:
        print("🔍 Нет колонок, присутствующих хотя бы в двух датафреймах")
        return

    # 3. Фильтруем только категориальные (object/category) и не числовые
    categorical_common = set()
    for col in candidate_cols:
        dfs_with_col = [df for _, df in col_to_dfs[col]]
        # Проверяем, что во всех этих датафреймах тип - object или category
        is_object_or_cat = all(
            df[col].dtype.name in ('object', 'category')
            for df in dfs_with_col
        )
        if is_object_or_cat:
            is_numeric = any(_is_likely_numeric(df[col]) for df in dfs_with_col)
            if not is_numeric:
                categorical_common.add(col)

    if not categorical_common:
        print("✔️ Все общие категориальные колонки согласованы - расхождений нет")
        return

    # 4. Подготовка к анализу
    print("Анализ согласованности категориальных данных между датафреймами\n")

    EMOJI_NUMBERS = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    print("🗃️ Источники данных:")
    for i, label in enumerate(dataset_labels):
        emoji_num = EMOJI_NUMBERS[i+1] if i+1 < len(EMOJI_NUMBERS) else f"{i+1}"
        print(f"    {emoji_num} {label}")
    
    print(f'\n📋 Чеклист:')
    all_diff_records = []

    # 5. Анализ каждой подходящей колонки
    for col in sorted(categorical_common):
        # Получаем только датафреймы и индексы, где есть эта колонка
        relevant_info = col_to_dfs[col]  # list of (original_idx, df)
        relevant_dfs = [df for _, df in relevant_info]
        relevant_indices = [idx for idx, _ in relevant_info]
        relevant_labels = [dataset_labels[idx] for idx in relevant_indices]

        # Собираем значения из всех релевантных датафреймов
        raw_sets: List[Set[str]] = []
        for df in relevant_dfs:
            vals = set(df[col].dropna().astype(str))
            raw_sets.append(vals)

        all_raw = set().union(*raw_sets)
        common_raw = set.intersection(*raw_sets) if raw_sets else set()
        diff_raw = all_raw - common_raw

        col_name, col_desc = label_for_column(col, separator="•")
        full_col_name = f"{col_name}{col_desc}"

        if not diff_raw:
            print(f"    ✔️ {full_col_name} 💎 полное совпадение во всех источниках")
            continue

        print(f"    🚨 {full_col_name} 📢 обнаружены расхождения")

        # Статистика до/после нормализации
        norm_sets = [{_normalize_text(v) for v in s} for s in raw_sets]
        all_norm = set().union(*norm_sets)
        common_norm = set.intersection(*norm_sets) if norm_sets else set()

        if len(all_norm) < len(all_raw):
            print(f"        📌 После нормализации совпадений стало больше: {len(common_norm)} из {len(all_norm)}")
        else:
            print(f"        📌 Уникальных значений: {len(all_raw)} (совпадает {len(common_raw)})")

        # Собираем записи для сводной таблицы
        for val in diff_raw:
            sources_indices = [i for i, s in enumerate(raw_sets) if val in s]
            sources_labels = [relevant_labels[i] for i in sources_indices]
            norm_val = _normalize_text(val)
            all_diff_records.append({
                "Колонка": full_col_name,
                "Оригинал": val,
                "Норм. значение": norm_val,
                "Источник": " | ".join(sources_labels)
            })

        # Fuzzy-поиск по всем уникальным значениям в этой колонке
        all_unique_vals = all_raw
        typo_groups = _find_typo_groups(all_unique_vals, max_distance=2)
        
        if typo_groups:
            print(f"        🧐 Возможные опечатки:")
            for group in typo_groups[:3]:
                print(f"             📄 {group}")

    # 6. Вывод результатов
    if not all_diff_records:
        print("\n✔️ Все категориальные колонки полностью согласованы!")
        return

    # Сводная таблица
    if report in ("diff", "full"):
        print(f"\nСводная таблица расхождений (выявлено: {len(all_diff_records)} расхождений):")
        diff_df = pd.DataFrame(all_diff_records)
        display_table(diff_df, rows=15, max_header_length=20)

    # Детальный анализ (матрицы)
    if report == "full":
        print(f"\nДетальный анализ по проблемным колонкам:")
        for col in sorted(categorical_common):
            relevant_info = col_to_dfs[col]
            relevant_dfs = [df for _, df in relevant_info]
            relevant_indices = [idx for idx, _ in relevant_info]
            relevant_labels = [dataset_labels[idx] for idx in relevant_indices]

            raw_sets = [set(df[col].dropna().astype(str)) for df in relevant_dfs]
            diff_raw = set().union(*raw_sets) - set.intersection(*raw_sets)
            if not diff_raw:
                continue

            col_name, col_desc = label_for_column(col, separator="•")
            full_col_name = f"{col_name}{col_desc}"
            print(f"\n🎹 Колонка: {full_col_name}")

            # Собираем: norm_val -> {local_index: (оригинал, частота)}
            norm_to_stats = {}
            for local_i, df in enumerate(relevant_dfs):
                value_counts = df[col].dropna().astype(str).value_counts()
                for val, count in value_counts.items():
                    norm_val = _normalize_text(val)
                    if norm_val not in norm_to_stats:
                        norm_to_stats[norm_val] = {}
                    if local_i not in norm_to_stats[norm_val]:
                        norm_to_stats[norm_val][local_i] = (val, count)
                    else:
                        prev_val, prev_count = norm_to_stats[norm_val][local_i]
                        norm_to_stats[norm_val][local_i] = (prev_val, prev_count + count)

            # Формируем строки матрицы
            matrix_rows = []
            for norm_val in sorted(norm_to_stats.keys()):
                row = {"Норм. значение": norm_val}
                for local_i, label in enumerate(relevant_labels):
                    if local_i in norm_to_stats[norm_val]:
                        orig, count = norm_to_stats[norm_val][local_i]
                        row[label] = f"{orig} ({count})"
                    else:
                        row[label] = "-"
                matrix_rows.append(row)

            if matrix_rows:
                matrix_df = pd.DataFrame(matrix_rows)
                display_table(matrix_df, rows=len(matrix_rows), max_header_length=30)


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


#audit_categorical_typos: Выявляет возможные опечатки в категориальных колонках датафрейма.
def audit_categorical_typos(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_distance: int = 2,
    min_frequency: int = 1,
    cmap="Oranges"
) -> None:
    """
    Выявляет возможные опечатки в категориальных колонках датафрейма.

    Описание: Функция анализирует категориальные колонки указанного датафрейма на предмет наличия похожих значений, 
            которые могут быть результатом опечаток. Автоматически исключает числовые колонки и значения с низкой частотой.
            Результаты анализа выводятся в виде отчета и сводной таблицы.

    Параметры:
        df: pd.DataFrame - Датафрейм для проведения аудита категориальных данных.
        columns: Optional[List[str]] - Список имен колонок, которые необходимо проверить. 
            Если не указан (None), анализируются все категориальные колонки датафрейма.
        max_distance: int - Максимальное расстояние Левенштейна между значениями для объединения их в группу похожих значений.
        min_frequency: int - Минимальная частота появления значения в колонке, чтобы оно было включено в анализ.
        cmap: str - Цветовая палитра для визуализации тепловой карты (используется при отображении сводной таблицы).

    Возвращаемое значение:
        None - Функция не возвращает явного значения, а выводит отчет о найденных проблемах и сводную таблицу в stdout.
    """
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    # Определяем колонки для анализа
    if columns is None:
        candidate_cols = []
        for col in df.columns:
            if df[col].dtype.name in ('object', 'category'):
                if not _is_likely_numeric(df[col]):
                    candidate_cols.append(col)
        columns = candidate_cols

    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        print("✔️ Нет категориальных колонок для анализа опечаток")
        return

    # Получаем описание датафрейма
    df_name, df_desc = label_for_dataset(df, separator="•")
    df_label = f"{df_name}{df_desc}" if df_desc else df_name

    print("Анализ опечаток внутри датафрейма\n")
    print(f"🗃️ Датафрейм             : {df_label}")
    print(f"🏷️ Категориальных колонок: {len(columns)}\n")

    print("📋 Чеклист:")
    issues_found = False
    all_typo_records = []

    for col in sorted(columns):
        col_name, col_desc = label_for_column(col, separator="•")
        full_col_name = f"{col_name}{col_desc}"

        value_counts = df[col].dropna().astype(str).value_counts()
        value_counts = value_counts[value_counts >= min_frequency]
        
        has_issues = False
        typo_groups = []

        if len(value_counts) > 1:
            values = value_counts.index.tolist()
            clean_value_map = {}
            for val in values:
                clean_val = re.sub(r'\s+', ' ', str(val).strip().lower())
                clean_value_map[clean_val] = val

            candidates_clean = list(clean_value_map.keys())
            used_clean = set()

            for i, v1_clean in enumerate(candidates_clean):
                if v1_clean in used_clean:
                    continue
                group_originals = [clean_value_map[v1_clean]]
                used_clean.add(v1_clean)
                for v2_clean in candidates_clean[i + 1:]:
                    if v2_clean in used_clean:
                        continue
                    if _levenshtein_distance(v1_clean, v2_clean) <= max_distance:
                        group_originals.append(clean_value_map[v2_clean])
                        used_clean.add(v2_clean)
                if len(group_originals) > 1:
                    has_issues = True
                    group_with_freq = [(orig, value_counts.get(orig, 0)) for orig in group_originals]
                    typo_groups.append(group_with_freq)

        if has_issues:
            issues_found = True
            total_problematic = sum(len(group) for group in typo_groups)
            print(f"    🚨 {full_col_name} 📢 опечаток: {total_problematic}")
            
            # Выводим группы СРАЗУ под колонкой
            for group_with_freq in typo_groups[:3]:  # ограничим 3 группами
                originals = [f"{orig}" for orig, freq in group_with_freq]
                print(f"         📄 {originals}")
            
            # Собираем данные для сводной таблицы
            for group_with_freq in typo_groups:
                for orig, freq in group_with_freq:
                    all_typo_records.append({
                        "Колонка": full_col_name,
                        "Значение": orig,
                        "Частота": freq,
                        "Группа": ", ".join([f"{o}" for o, f in group_with_freq])
                    })
        else:
            print(f"    ✔️ {full_col_name} 💎 качество данных хорошее")

    if not issues_found:
        print("\n✨ Все категориальные колонки прошли проверку на опечатки!")
        return

    # Сводная таблица
    print(f"\n📋 Сводная таблица опечаток (выявлено опечаток: {len(all_typo_records)}):")
    typo_df = pd.DataFrame(all_typo_records)
    display_table(typo_df, 
               rows=15, 
               max_header_length=25, 
               styler_func=lambda s: s.background_gradient(subset=["Частота"], cmap=cmap)
            )
    
    print("\n🛠️ Рекомендации:")
    print("     • Проверьте группы похожих значений на опечатки")
    print("     • Рассмотрите унификацию написания через словарь замен")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# audit_numerical_distribution • Выводит расширенный текстовый отчёт о распределении числового признака
def audit_numerical_distribution(
    df: pd.DataFrame, 
    col: str,
    show_recommendations: bool = False
) -> None:
    """
    Выводит расширенный текстовый отчёт о распределении числового признака.

    Отчёт включает:
        - Основные статистики (среднее, медиана, std, мин/макс, квартили)
        - Коэффициенты асимметрии (skewness) и эксцесса (kurtosis)
        - Количество и процент выбросов (по правилу 1.5хIQR)
        - Экстремальные значения (по правилу 3хIQR)
        - Подозрительные значения (максимум > медианы * 3 и > 1000)
        - Анализ на почти константность
        - Опционально: интерпретация формы распределения и рекомендации

    Параметры
   -------
    df : pd.DataFrame
        Исходный датафрейм.
    col : str
        Название числового столбца для анализа.
    show_recommendations : bool, по умолчанию False
        Если True - выводит раздел с интерпретацией и рекомендациями.
        Если False - выводит только статистику и диагностику.

    Возвращает
   ----
    None
        Выводит отчёт в консоль.
    """

    # Автоматический поиск имени и описания датасета
    dataset_name, dataset_desc = label_for_dataset(df)
    print(f"🗃️ Датасет '{dataset_name}' 📋 {dataset_desc}")

    if col not in df.columns:
        print(f"Колонка '{col}' не найдена")
        return
    
    if not is_numeric_dtype(df[col]):
        print(f"❌ Колонка '{col}' не является числовой")
        return

    data = df[col].dropna()
    n_total = len(df[col])
    n_valid = len(data)
    n_missing = n_total - n_valid

    # Получаем описание колонки
    col_name, col_desc = label_for_column(col, separator='•')

    print(f"🔍 Анализ распределения: {col}{col_desc}")
    print(f"     • Записей: {n_valid:,} (пропусков: {n_missing})")
    print(f"     • Тип: {df[col].dtype}")

    if n_valid == 0:
        print("   ⚠️ Нет валидных данных для анализа")
        return

    # Основные статистики
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skew_val = stats.skew(data)
    kurt_val = stats.kurtosis(data)  # эксцесс (0 = нормальное)

    print(f"📈 Основные статистики:")
    print(f"     • Минимум: {min_val:.3f} | Максимум: {max_val:.3f}")
    print(f"     • Среднее: {mean_val:.3f} | Медиана: {median_val:.3f} 💎 смещение: {mean_val - median_val:+.3f}")
    print(f"     • Стандартное отклонение: {std_val:.3f}")

    # Квартили и IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    print(f"📊 Квартили:")
    print(f"     • Q1 (25%): {q1:.3f}")
    print(f"     • Q2 (50% / медиана): {median_val:.3f}")
    print(f"     • Q3 (75%): {q3:.3f}")
    print(f"     • IQR: {iqr:.3f}")

    # Асимметрия и эксцесс
    skew_desc = (
        "сильная ▶ правосторонняя" if skew_val > 1 else
        "умеренная ▶ правосторонняя" if skew_val > 0.5 else
        "умеренная ◀ левосторонняя" if skew_val < -0.5 else
        "сильная ◀ левосторонняя" if skew_val < -1 else
        "близка к симметричной"
    )
    print(f"⚖️ Асимметрия (skew): {skew_val:.2f} 💎 {skew_desc}")

    kurt_desc = (
        "сильно островерхое" if kurt_val > 1 else
        "островерхое" if kurt_val > 0 else
        "плосковерхое" if kurt_val < -0.5 else
        "близко к нормальному"
    )
    print(f"📉 Эксцесс (kurtosis): {kurt_val:.2f} 💎 {kurt_desc}")

    # Выбросы по правилу 1.5×IQR
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_low = data[data < lower_bound]
    outliers_high = data[data > upper_bound]
    n_outliers = len(outliers_low) + len(outliers_high)
    pct_outliers = n_outliers / n_valid * 100

    if n_outliers > 0:
        print(f"🔶 Выбросы (по правилу 1.5×IQR): {n_outliers} ({pct_outliers:.1f}%)")
        print(f"     • Нижняя граница  : {lower_bound:.3f}")
        print(f"     • Верхняя граница : {upper_bound:.3f}")
        print(f"     • Ниже нижней границы : {len(outliers_low)}")
        print(f"     • Выше верхней границы: {len(outliers_high)}")

        # Проверка на подозрительно большое значение
        if (median_val > 0 and 
            max_val > median_val * 3 and 
            max_val > 1000):
            print(f"     • 🚨 Подозрительно большое значение: {int(max_val):,}")
    else:
        print(f"✔️ Выбросов не обнаружено")

    # Экстремальные значения (по правилу 3×IQR)
    lower_extreme = q1 - 3 * iqr
    upper_extreme = q3 + 3 * iqr
    extremes_low = data[data < lower_extreme]
    extremes_high = data[data > upper_extreme]
    n_extremes = len(extremes_low) + len(extremes_high)
    pct_extremes = n_extremes / n_valid * 100

    if n_extremes > 0:
        print(f"💥 Экстремальные значения (по правилу 3×IQR): {n_extremes} ({pct_extremes:.1f}%)")
        print(f"     • Нижняя граница  : {lower_extreme:.3f}")
        print(f"     • Верхняя граница : {upper_extreme:.3f}")
        print(f"     • Ниже нижней границы : {len(extremes_low)}")
        print(f"     • Выше верхней границы: {len(extremes_high)}")

        # Проверка на подозрительно большое значение
        if (median_val > 0 and 
            max_val > median_val * 3 and 
            max_val > 1000):
            print(f"      🚨 Подозрительно большое значение: {int(max_val):,}")
    else:
        print(f"✔️ Экстремальных значений не обнаружено")

    # Почти константность
    n_unique = data.nunique()
    if n_unique == 1:
        print(f"🔇 Почти константный признак: все значения одинаковы ({min_val})")
    elif n_unique == 2 and len(data) > 10:
        top2_sum = data.value_counts().nlargest(2).sum()
        if top2_sum / len(data) > 0.99:
            top_vals = data.value_counts().nlargest(2).index.tolist()
            print(f"🔇 Почти константный признак: 99%+ значений сосредоточено в двух категориях: {top_vals}")

    # Рекомендации (опционально)
    if show_recommendations:
        print(f"\n🔍 Рекомендации:")

        if abs(skew_val) > 0.5:
            print("     • Распределение скошено 📢 рассмотрите лог-преобразование.")
        else:
            print("     • Распределение близко к симметричному.")

        if n_outliers > 0 or n_extremes > 0:
            print("     • Присутствуют выбросы/экстремальные значения 📢 проверьте их природу (ошибка или особенность?).")
        else:
            print("     • Выбросов и экстремальных значений не обнаружено.")

        if std_val > 0:
            if max_val - min_val > 100:
                print("     • Широкий диапазон значений 📢 рассмотрите стандартизацию для ML.")
            elif max_val <= 1 and min_val >= 0:
                print("     • Данные в [0,1] - масштабирование, вероятно, не требуется.")
            else:
                print("     • Для моделей, чувствительных к масштабу, протестируйте стандартизацию и нормализацию.")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_feature_distribution - Визуализирует распределение числового признака с поддержкой группировки, нормировки и лог-шкалы
def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    hue: Optional[str] = None,
    bins: Union[int, Literal["auto"]] = "auto",
    palette: str = "tab10",
    stat: Literal['auto', 'count', 'density', 'probability'] = 'auto',
    log_scale: bool = False,
    table_metrics: Literal['basic', 'extended'] = 'basic',
    show_legend: bool = True 
) -> None:
    """
    Визуализирует распределение числового признака с поддержкой группировки, нормировки и лог-шкалы.
    
    Генерирует два графика:
        1. Гистограмма с KDE - для анализа формы распределения.
        2. Boxplot + stripplot - для оценки разброса и выбросов.
    
    Параметры:
        df: pd.DataFrame - датафрейм для анализа
        feature: str - имя числового признака
        hue: Optional[str] - категориальный признак для группировки (опционально)
        bins: Union[int, Literal["auto"]] - число бинов или "auto" (по умолчанию)
        palette: str - цветовая палитра seaborn (по умолчанию 'tab10')
        stat: Literal['auto', 'count', 'density', 'probability'] - тип нормировки гистограммы
            - 'auto': 'count' без hue, 'density' с hue (рекомендуется)
        log_scale: bool - применить логарифмическую шкалу по X (только для положительных данных)
        table_metrics: Literal['basic', 'extended'] - уровень детализации таблицы статистик
            - 'basic': количество, доля, среднее, медиана и т.д.
            - 'extended': + асимметрия, эксцесс, IQR/Медиана, доля выбросов (%)
        show_legend: bool - отображать легенду на левом графике (гистограмма + KDE)
    
    Возвращает:
        None - отображает графики и таблицу в ячейку ноутбука
    """
    import warnings

    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    if feature not in df.columns:
        raise ValueError(f"Признак '{feature}' не найден в датафрейме")
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError(f"Признак '{feature}' должен быть числовым")


    #Обработка hue: замена NaN и пробелов на категории
    #if hue is not None and hue in df.columns:
    #    if not pd.api.types.is_numeric_dtype(df[hue]):
    #        original_hue = df[hue].copy()
    #        processed_hue = original_hue.copy()
    #
    #        # Заменяем NaN на [пропуски]
    #        processed_hue = processed_hue.fillna('[пропуски]')
    #
    #        # Заменяем строки, состоящие только из пробелов, на [пробелы]
    #        if pd.api.types.is_string_dtype(original_hue):
    #            whitespace_mask = (original_hue.astype(str).str.strip() == '') & original_hue.notna()
    #            processed_hue.loc[whitespace_mask] = '[пробелы]'
    #
    #        # Обновляем столбец hue в датафрейме
    #        df = df.assign(**{hue: processed_hue}).copy()


    # Подготовка данных
    required_cols = [feature] if hue is None else [feature, hue]
    df_clean = df.dropna(subset=required_cols).copy()
    if len(df_clean) == 0:
        print("⚠️ После удаления пропусков данных нет")
        return

    data = df_clean[feature]
    if len(data) == 0:
        print("⚠️ Признак содержит только пропуски")
        return

    # ЛОГ-ШКАЛА: проверка
    use_log = log_scale
    if use_log:
        if (data <= 0).any():
            warnings.warn(
                f"Признак '{feature}' содержит неположительные значения. Лог-шкала отключена.",
                UserWarning
            )
            use_log = False

    # ОПРЕДЕЛЕНИЕ STAT
    resolved_stat = stat
    if stat == 'auto':
        resolved_stat = 'density' if (hue is not None and hue in df.columns) else 'count'
    else:
        if hue is not None and hue in df.columns and stat == 'count':
            warnings.warn(
                "При сравнении групп разного размера stat='count' может вводить в заблуждение. "
                "Рассмотрите stat='density' или 'probability' для честного сравнения форм распределений.",
                UserWarning
            )

    # Общая статистика (на основе исходных значений)
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Автовыбор бинов (Freedman-Diaconis)
    if bins == "auto":
        series = data
        if series.empty or series.nunique() == 1:
            n_bins = 10
        else:
            q75, q25 = np.percentile(series, [75, 25])
            iqr_fd = q75 - q25
            h = 2 * iqr_fd / (len(series) ** (1/3)) if iqr_fd > 0 else 2 * series.std() / (len(series) ** (1/3))
            n_bins = int(np.ceil((series.max() - series.min()) / h)) if h > 0 else 10
            n_bins = max(5, min(n_bins, 50))
        bins = n_bins

    # Подписи
    feature_name, feature_desc = label_for_column(feature, separator='•')
    feature_label = f"{feature_name}{feature_desc}" if feature_desc else feature_name
    xlabel = f"log({feature_label})" if use_log else feature_label

    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    # 1. ГИСТОГРАММА
    use_hue = hue is not None and hue in df.columns

    if use_hue:
        value_counts = df_clean[hue].value_counts()
        small_cats = value_counts[value_counts < 5].index.tolist()
        large_df = df_clean[~df_clean[hue].isin(small_cats)]
        small_df = df_clean[df_clean[hue].isin(small_cats)]

        hist_kwargs = dict(
            x=feature,
            hue=hue,
            bins=bins,
            palette=palette,
            ax=axes[0],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            stat=resolved_stat,
            common_norm=False,
            log_scale=use_log,
            legend=show_legend
        )

        if not large_df.empty:
            sns.histplot(data=large_df, kde=True, **hist_kwargs)
        if not small_df.empty:
            sns.histplot(data=small_df, kde=False, **hist_kwargs)

        ylabel = {
            'count': "Количество",
            'density': "Плотность",
            'probability': "Вероятность"
        }.get(resolved_stat, resolved_stat.capitalize())

    else:
        sns.histplot(
            data=df_clean,
            x=feature,
            bins=bins,
            color="steelblue",
            ax=axes[0],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            kde=True,
            stat=resolved_stat,
            log_scale=use_log,
            legend=False
        )
        ylabel = {
            'count': "Количество",
            'density': "Плотность",
            'probability': "Вероятность"
        }.get(resolved_stat, resolved_stat.capitalize())

    # Заголовок гистограммы
    norm_note = "\nнормировано по группам" if use_hue and resolved_stat != 'count' else ""
    log_note = "\nлог-шкала" if use_log else ""
    axes[0].set_title(f"Распределение: \n{feature_label}{norm_note}{log_note}", fontsize=10)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # 2. BOXPLOT + STRIP
    boxplot_kwargs = dict(
        data=df_clean,
        y=feature,
        ax=axes[1],
        linewidth=1.5,
        flierprops=dict(marker='o', markerfacecolor='orange', markeredgecolor='black', markersize=8, alpha=0.8),
        boxprops=dict(alpha=0.6 if not use_hue else 0.7, linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color='darkred')
    )

    if use_hue:
        sns.boxplot(x=hue, **boxplot_kwargs, palette=palette, width=0.7)
        sns.stripplot(x=hue, y=feature, data=df_clean, color="#2E5472", alpha=0.5, size=1.5, ax=axes[1], jitter=0.25)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, ha="right")
        hue_name, hue_desc = label_for_column(hue, separator='•')
        hue_label = f"{hue_name}{hue_desc}" if hue_desc else hue_name
        axes[1].set_xlabel(hue_label)
    else:
        sns.boxplot(**boxplot_kwargs, color="lightsteelblue", width=0.5)
        sns.stripplot(y=feature, data=df_clean, color="#2E5472", alpha=0.5, size=1.5, ax=axes[1], jitter=0.25)
        axes[1].set_xlabel("")

    # Применяем лог-шкалу и к boxplot
    if use_log:
        axes[1].set_yscale('log')
        axes[1].set_ylabel(f"log({feature_name})")
    else:
        axes[1].set_ylabel(feature_name)

    axes[1].set_title(f"Выбросы и разброс: \n{feature_label}{log_note}", fontsize=10)

    # Статистика в углу boxplot
    stats_text = (
        f"Среднее: {mean_val:.1f}\n"
        f"Медиана: {median_val:.1f}\n"
        f"Стд: {std_val:.1f}\n"
        f"Мин: {min_val:.1f}\n"
        f"Макс: {max_val:.1f}\n"
        f"Q1: {q1:.1f}\n"
        f"Q3: {q3:.1f}\n"
        f"IQR: {iqr:.1f}\n"
        f"Границы: {lower_bound:.1f} – {upper_bound:.1f}\n"
        f"Выбросы: {len(outliers)}"
    )
    axes[1].text(
        0.98, 0.955,
        stats_text,
        transform=axes[1].transAxes,
        ha='right',
        va='top',
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8)
    )

    # Явно включаем сетку
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # ТАБЛИЦА СТАТИСТИК ПО КАТЕГОРИЯМ
    if use_hue:
        total_count = len(df.dropna(subset=[feature]))
        categories = df_clean[hue].unique()
        colors = sns.color_palette(palette, n_colors=len(categories))
        color_map = dict(zip(categories, colors))
        
        stats_records = []
        single_value_cats = []
        
        for cat in categories:
            cat_data = df_clean[df_clean[hue] == cat][feature]
            n = len(cat_data)
            if n == 0:
                continue
                
            count = n
            pct = count / total_count * 100
            color_hex = matplotlib.colors.to_hex(color_map[cat])
            
            mean_val = cat_data.mean()
            median_val = cat_data.median()
            std_val = cat_data.std() if n > 1 else np.nan
            min_val = cat_data.min()
            max_val = cat_data.max()
            
            if n == 1:
                single_value_cats.append(str(cat))
            
            # Базовые поля
            record = {
                "Категория": str(cat),
                "Количество": count,
                "Доля (%)": pct,
                "Среднее": mean_val,
                "Медиана": median_val,
                "Стд": std_val,
                "Мин": min_val,
                "Макс": max_val,
            }
            
            # Extended метрики
            if table_metrics == 'extended':
                skew_val = skew(cat_data) if n > 2 else np.nan
                kurt_val = kurtosis(cat_data, fisher=False) if n > 3 else np.nan  # Pearson's kurtosis
                
                q1_local = cat_data.quantile(0.25)
                q3_local = cat_data.quantile(0.75)
                iqr_local = q3_local - q1_local
                iqr_over_med = iqr_local / median_val if median_val != 0 else np.nan
                
                lower_local = q1_local - 1.5 * iqr_local
                upper_local = q3_local + 1.5 * iqr_local
                n_outliers_local = ((cat_data < lower_local) | (cat_data > upper_local)).sum()
                outlier_pct_local = (n_outliers_local / n) * 100 if n > 0 else 0.0

                # Направление асимметрии
                if pd.isna(skew_val):
                    skew_dir = "-"
                elif skew_val > 0.5:
                    skew_dir = "▶▶ сильно правосторонняя"
                elif skew_val > 0.1:
                    skew_dir = "▶ правосторонняя"
                elif skew_val < -0.5:
                    skew_dir = "◀◀ сильно левосторонняя"
                elif skew_val < -0.1:
                    skew_dir = "◀ левосторонняя"
                else:
                    skew_dir = "≈ симметричная"
                
                record.update({
                    "Асимметрия": skew_val,
                    "Смещение": skew_dir,
                    "Эксцесс": kurt_val,
                    "IQR / Медиана": iqr_over_med,
                    "Выбросы (%)": outlier_pct_local
                })
            
            stats_records.append(record)
        
        stats_df = pd.DataFrame(stats_records).sort_values("Доля (%)", ascending=False)
        
        if single_value_cats:
            print(f"\n⚠️ Внимание: в категориях {', '.join(single_value_cats)} только одно значение - Стд и форма не определены.")
            print("   Рекомендуется проверить данные на полноту или объединить редкие категории.")
        
        
        
        print(f"\nСтатистика по категориям '{feature_name}'{feature_desc}")

        if not use_hue:
            hue_name, hue_desc = label_for_column(hue, separator='•')
            print(f"для признака '{hue_name}'{hue_desc}")
        
        # Формирование колонок
        base_cols = ["Категория", "Количество", "Доля (%)", "Среднее", "Медиана", "Стд", "Мин", "Макс"]
        extended_cols = ["Асимметрия", "Смещение", "Эксцесс", "IQR / Медиана", "Выбросы (%)"]
        all_cols = base_cols + (extended_cols if table_metrics == 'extended' else [])
        color_map_for_styling = {str(cat): matplotlib.colors.to_hex(color_map[cat]) for cat in categories}

        def styler(s: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
            # Функция для определения цвета текста на основе яркости фона
            def get_text_color(bg_hex):
                # Преобразуем HEX в RGB
                r = int(bg_hex[1:3], 16) / 255.0
                g = int(bg_hex[3:5], 16) / 255.0
                b = int(bg_hex[5:7], 16) / 255.0

                # Вычисляем относительную яркость (W3C formula)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "white" if luminance < 0.5 else "black"

            fmt_dict = {
                "Доля (%)": "{:.1f}%",
                "Среднее": "{:.2f}",
                "Медиана": "{:.2f}",
                "Стд": "{:.2f}",
                "Мин": "{:.2f}",
                "Макс": "{:.2f}"
            }
            if table_metrics == 'extended':
                fmt_dict.update({
                    "Асимметрия": "{:.2f}",
                    "Эксцесс": "{:.2f}",
                    "IQR / Медиана": "{:.2f}",
                    "Выбросы (%)": "{:.1f}%"
                })

            s = s.format(fmt_dict, na_rep="-")
            s = s.set_properties(subset=["Категория"], **{"text-align": "left"})
            s = s.background_gradient(subset=["Доля (%)"], cmap="Oranges")

            if table_metrics == 'extended':
                s = s.background_gradient(subset=["Асимметрия"], cmap="RdYlGn_r", vmin=-2, vmax=2)
                s = s.background_gradient(subset=["Выбросы (%)"], cmap="Oranges")

            # Добавляем цвет фона и автоматический цвет текста
            def apply_category_bg_color(col):
                colors = []
                for val in col:
                    bg_hex = color_map_for_styling.get(val, "#ffffff")  # default white
                    text_color = get_text_color(bg_hex)
                    colors.append(f"background-color: {bg_hex}; color: {text_color};")
                return colors

            s = s.apply(apply_category_bg_color, subset=["Категория"])
            return s

        display_table(
            stats_df[all_cols],
            rows=len(stats_df),
            float_precision=2,
            styler_func=styler
        )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_feature_distribution_advanced • Визуализирует распределение числового признака с помощью комбинированного графика
def plot_feature_distribution_advanced(
    df: pd.DataFrame,
    col: str,
    bins='scott',
    figsize=(15, 4.5),
    xlim=None,
    binwidth=None,
    stat_type='count',
    outlier_iqr_multiplier: float = 1.5,
    ax: Optional[plt.Axes] = None,
    show_stats: bool = True,
    title: Optional[str] = None,
    show_outliers: int = 0
):
    """
    Визуализирует распределение числового признака с помощью многослойного графика и
    (опционально) выводит табличные данные по выявленным выбросам.

    Объединяет гистограмму, KDE, boxplot, stripplot и статистические метки в одном графике.
    Поддерживает гибкое определение выбросов, встраивание в subplot'ы и кастомизацию.

    Параметры
   -------
    df : pd.DataFrame
        Исходный датафрейм.
    col : str
        Название числового столбца для анализа.
    bins : str or int, optional
        Способ определения числа бинов: 'scott', 'fd', 'auto' или int.
    figsize : tuple, optional
        Размер фигуры (ширина, высота). Используется только если ax=None.
    xlim : tuple, optional
        Фиксированные пределы по оси X.
    binwidth : float, optional
        Ширина бина в гистограмме.
    stat_type : str, optional
        Тип статистики: 'count' (по умолчанию) или 'density'.
    outlier_iqr_multiplier : float, optional
        Множитель IQR для определения границ выбросов (по умолчанию 1.5).
        Значения ≥3.0 считаются "экстремальными".
    ax : matplotlib.axes.Axes, optional
        Ось для отрисовки. Если None - создаётся новая фигура.
    show_stats : bool, optional
        Отображать ли текстовый блок со статистикой (по умолчанию True).
    title : str, optional
        Кастомный заголовок. Если None - генерируется автоматически.
    show_outliers : int, optional
        Сколько строк с выбросами показать в виде таблицы (по умолчанию 0 - не показывать).
        При значении > 0 выводится отсортированная таблица с подсветкой колонки `col`.

    Возвращает
   -------
    None
        Отображает график и (опционально) таблицу с выбросами.
    """

    # Валидация параметров
    if stat_type not in ('count', 'density'):
        raise ValueError("stat_type must be 'count' or 'density'")
    

    col_name, col_desc = label_for_column(col, separator='•')

    print(f"Распределение признака {col_name}{col_desc} с KDE, boxplot, scatter и статистиками\n")
    dataset_profile(df, report='head')
    print()

    # Подготовка данных
    dataset_name, dataset_desc = label_for_dataset(df, separator='•')
    data = df[col].dropna()
    if len(data) == 0:
        raise ValueError(f"Колонка '{col_name}' содержит только пропуски")

    # Проверка на числовой тип (исключая bool)
    if not pd.api.types.is_numeric_dtype(data) or data.dtype == bool:
        raise TypeError(
            f"Колонка '{col_name}' имеет тип {df[col].dtype} и не подходит для анализа распределения\n"
            "Используйте эту функцию только для непрерывных или дискретных числовых признаков (int, float)."
        )

    # Статистики
    mean_val = data.mean()
    median_val = data.median()
    min_val = data.min()
    max_val = data.max()
    std_val = data.std()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - outlier_iqr_multiplier * iqr
    upper_bound = q3 + outlier_iqr_multiplier * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Создание или использование оси
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
        is_own_figure = True
    else:
        ax1 = ax
        is_own_figure = False

    #col_name, col_desc = label_for_column(col, separator='•')

    # Заголовок
    if title is None:
        title = f"Датасет  : {dataset_name}{dataset_desc}\nКолонка : {col_name}{col_desc}"
    ax1.set_title(
        title,
        fontsize=12,
        fontweight='bold',
        loc='left',
        pad=20
    )

    # Гистограмма
    sns.histplot(
        data,
        ax=ax1,
        kde=False,
        bins=bins,
        binwidth=binwidth,
        color="#86BCE7",
        edgecolor="#E6F0F5",
        alpha=0.8,
        stat=stat_type
    )

    # Ширина бина и KDE
    patches = ax1.patches
    bin_width = patches[0].get_width() if patches else ((data.max() - data.min()) / bins if isinstance(bins, int) else 1.0)

    sns.kdeplot(data, ax=ax1, color="#295C96", linewidth=2.5, alpha=0.7)
    if stat_type == 'count':
        scale_factor = len(data) * bin_width
        ax1.lines[-1].set_ydata(ax1.lines[-1].get_ydata() * scale_factor)

    # Линии статистик
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Среднее')
    ax1.axvline(median_val, color='#ef7a31', linestyle='-', linewidth=2, label='Медиана')
    ax1.axvline(min_val, color="#1aa38c81", linestyle='--', linewidth=1, label='Минимум')
    ax1.axvline(max_val, color="#9d1aa477", linestyle='--', linewidth=1, label='Максимум')
    ax1.axvspan(q1, q3, alpha=0.1, color='#9467bd', label='IQR')

    # Текстовый блок (опционально)
    if show_stats:
        stats_text = (
            f"Среднее: {mean_val:.1f}\n"
            f"Медиана: {median_val:.1f}\n"
            f"Стд: {std_val:.1f}\n"
            f"Мин: {min_val:.1f} | Макс: {max_val:.1f}\n"
            f"Q1: {q1:.1f} | Q3: {q3:.1f} | IQR: {iqr:.1f}\n"
            f"Границы ({outlier_iqr_multiplier}×IQR): {lower_bound:.1f} – {upper_bound:.1f}\n"
            f"Выбросы: {len(outliers)}"
        )
        ax1.text(
            0.985, 0.95,
            stats_text,
            transform=ax1.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.7)
        )

    # Вторая ось: boxplot + stripplot
    ax2 = ax1.twinx()

    sns.boxplot(
        x=data,
        ax=ax2,
        width=0.1,
        color='orange',
        saturation=0.75,
        linewidth=1.5,
        flierprops=dict(
            marker='o',
            markerfacecolor='#ef7a31',
            markeredgecolor='#7b3910',
            markersize=10,
            alpha=0.8,
            zorder=8
        ),
        medianprops=dict(color='#ef7a31', linewidth=12, alpha=0.4),
        boxprops=dict(alpha=0.3, edgecolor='darkorange'),
        whiskerprops=dict(color='darkorange', linewidth=1.5),
        capprops=dict(color='darkorange', linewidth=1.5)
    )

    sns.stripplot(
        x=data,
        ax=ax2,
        color="#29648f",
        alpha=0.2,
        size=6,
        jitter=0.04,
        edgecolor='white',
        linewidth=0.5,
        zorder=1
    )

    # Маркеры статистик на оси точек
    ax2.scatter(mean_val, 0, color='red', edgecolors='white', s=120, marker='D', zorder=10)
    ax2.scatter(median_val, 0, color='#ef7a31', edgecolors='white', s=120, marker='^', zorder=10)
    ax2.scatter(q1, 0, color='#9467bd', edgecolors='white', s=120, marker='s', zorder=10)
    ax2.scatter(q3, 0, color='#9467bd', edgecolors='white', s=120, marker='s', zorder=10)

    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 0.1)

    # Границы выбросов
    if outlier_iqr_multiplier >= 3.0:
        bound_color = '#d32f2f'
        bound_style = '-.'
    else:
        bound_color = '#ea54f7'
        bound_style = '--'

    ax1.axvline(q1, color='#9467bd', linestyle='-.', linewidth=2)
    ax1.axvline(q3, color='#9467bd', linestyle='-.', linewidth=2)
    ax1.axvline(lower_bound, color=bound_color, linestyle=bound_style, linewidth=1.2, alpha=0.8, label=f'Границы ({outlier_iqr_multiplier}×IQR)')
    ax1.axvline(upper_bound, color=bound_color, linestyle=bound_style, linewidth=1.2, alpha=0.8)

    # Легенда
    ax1.plot([], [], color='red', marker='D', linestyle='', markersize=8, label='Среднее')
    ax1.plot([], [], color='#ef7a31', marker='^', linestyle='', markersize=8, label='Медиана')
    ax1.plot([], [], color='#9467bd', marker='s', linestyle='', markersize=8, label='Q1/Q3')

    if len(outliers) > 0:
        label_text = f"{len(outliers)} выбросов"
        ax2.text(
            0.98, 0.4,
            label_text,
            transform=ax2.transAxes,
            ha='right',
            va='center',
            fontsize=9,
            color='#7b3910',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.3)
        )

    if xlim is not None:
        ax1.set_xlim(xlim)

    ax1.legend(
        loc='upper left',
        fontsize=9,
        bbox_to_anchor=(1.02, 1.025),
        frameon=True,
        facecolor="#ffffff",
        fancybox=True,
        shadow=False,
        borderpad=1
    )

    # Телеметрия и вывод выбросов (опционально)
    n_outliers = len(outliers)
    total_valid = len(data)
    pct_outliers = 100 * n_outliers / total_valid if total_valid > 0 else 0.0

    # Подготовка метки колонки
    #col_name, col_desc = label_for_column(col, separator="•")
    col_label = f"{col_name}{col_desc}" if col_desc else col_name

    # Телеметрия и вывод выбросов (опционально)
    n_outliers = len(outliers)
    total_valid = len(data)
    pct_outliers = 100 * n_outliers / total_valid if total_valid > 0 else 0.0

    if n_outliers > 0:
        if outlier_iqr_multiplier >= 3.0:
            label = f"💥 Экстремальные значения ({outlier_iqr_multiplier}×IQR)"
        else:
            label = f"🔶 Выбросы ({outlier_iqr_multiplier}×IQR)"
        
        print(f"{label} в '{col_name}': {n_outliers} ({pct_outliers:.1f}%)")

    if show_outliers > 0 and n_outliers > 0:
        outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
        outliers_df = df.loc[outlier_indices].copy()
        outliers_df = outliers_df.sort_values(col, ascending=False)
        n_show = min(show_outliers, len(outliers_df))

        # Согласованный термин для таблицы
        term = "экстремальных значений" if outlier_iqr_multiplier >= 3.0 else "выбросов"

        if n_outliers <= show_outliers:
            print(f"\n🚨 Всего {term} по признаку '{col_name}': {n_outliers}")
        else:
            print(f"\n🚨 Топ-{n_show} {term} из {n_outliers} по признаку '{col_name}':")

        def highlight_col(styler):
            return styler.background_gradient(subset=[col], cmap="Oranges")

        display_table(
            outliers_df,
            rows=n_show,
            float_precision=2,
            styler_func=highlight_col
        )

    # Отображение
    if is_own_figure:
        plt.subplots_adjust(left=0.06, right=0.75)

        # Явно включаем сетку
        ax1.grid(True, linestyle='-', alpha=0.5)
        ax2.grid(True, linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_target_relationships: Визуализирует зависимости числовых признаков от целевого признака  
def plot_target_relationships(
    df: pd.DataFrame,
    target: str,
    hue: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    cols_per_row: int = 3,
    palette: str = "tab10",
    report: Literal["summary", "full"] = "summary",
    method: Literal["pearson", "spearman"] = "spearman"
) -> None:
    """
    Визуализирует зависимости числовых признаков от целевого признака.
    
    Описание:
        Создаёт scatter plot для каждого числового признака против таргета.
        Поддерживает цветовую группировку по категориальному признаку (hue).
        В режиме 'full' выводит таблицу легенды с частотами и статистикой по таргету.
        Использует глобальные справочники для автоматической подписи признаков.
        Позволяет выбирать метод расчёта корреляции: Pearson (чувствителен к выбросам)
        или Spearman (устойчив к выбросам и нелинейным монотонным связям).

    Параметры:
        df: pd.DataFrame - датафрейм для анализа
        target: str - имя целевого числового признака
        hue: Optional[str] - категориальный признак для цветовой группировки
        exclude: Optional[List[str]] - колонки для исключения из анализа
        include: Optional[List[str]] - анализировать только указанные колонки
        cols_per_row: int - количество графиков в строке (по умолчанию 3)
        palette: str - цветовая палитра seaborn (по умолчанию 'tab10')
        report: Literal["summary", "full"] - уровень детализации:
            - "summary": только графики,
            - "full": графики + таблица легенды с частотами и статистикой по таргету
        method: Literal["pearson", "spearman"] - метод расчёта корреляции 
            с целевым признаком (по умолчанию "spearman")

    Возвращаемое значение:
        None - отображает графики и (опционально) таблицу легенды
    """
    
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    if method not in ("pearson", "spearman"):
        raise ValueError("Параметр 'method' должен быть 'pearson' или 'spearman'")
        
     # Проверка: существует ли target
    if target not in df.columns:
        available_cols = ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else "")
        error_msg = (
            f"❌ Целевой признак '{target}' не найден в датафрейме\n"
            f"   Доступные колонки: {available_cols}"
        )
        raise ValueError(error_msg)

    # Проверка: является ли target числовым
    target_series = df[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        try:
            pd.to_numeric(target_series, errors='raise')
        except (ValueError, TypeError):
            sample_values = target_series.dropna().head(3).tolist()
            error_msg = (
                f"❌ Целевой признак '{target}' должен быть числовым\n"
                f"   Текущий тип: {target_series.dtype}\n"
                f"   Примеры значений: {sample_values}\n"
            )
            raise ValueError(error_msg)

    # Получаем числовые колонки
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude:
        num_cols = [col for col in num_cols if col not in exclude]
    if include:
        num_cols = [col for col in num_cols if col in include]
    if target in num_cols:
        num_cols.remove(target)
    
    if not num_cols:
        print("✔️ Нет числовых признаков для визуализации")
        return

    # Получаем подписи
    target_name, target_desc = label_for_column(target, separator='•')
    target_label = f"{target_name}{target_desc}"

    # Настройка графиков
    n_plots = len(num_cols)
    n_rows = (n_plots + cols_per_row - 1) // cols_per_row
    
    figsize = (4 * cols_per_row, 5.0 * max(n_rows, 1))
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Построение графиков
    for i, col in enumerate(num_cols):
        ax = axes[i]
        
        if hue and hue in df.columns:
            sns.scatterplot(data=df, x=col, y=target, hue=hue, palette=palette, ax=ax, legend=False)
        else:
            sns.scatterplot(data=df, x=col, y=target, ax=ax, legend=False)
        
        col_name, col_desc = label_for_column(col, separator='•')
        col_label = f"{col_name}{col_desc}" if col_desc else col_name
        
        ax.set_title(f"{col_label}", fontsize=8)
        ax.set_xlabel(col_label, fontsize=8)
        ax.set_ylabel(target_label, fontsize=8)

    # Скрытие пустых subplot'ов
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    # Заголовок
    fig.suptitle(
        f"Зависимости от целевого признака:\n{target_label}",
        fontsize=12, fontweight="bold", ha="left", x=0.02, y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Таблица легенды и статистики (только в режиме 'full' и если задан hue)
    if report == "full" and hue and hue in df.columns:
        print(f"\nЛегенда и статистика по категориальному признаку '{hue}':")
        
        # Статистика по категориям
        value_counts = df[hue].value_counts()
        total = len(df)
        
        legend_records = []
        colors = sns.color_palette(palette, n_colors=len(value_counts))
        
        for i, (cat, count) in enumerate(value_counts.items()):
            pct = count / total * 100
            
            # БЕЗОПАСНОЕ ПРЕОБРАЗОВАНИЕ К ЧИСЛУ
            target_series = df[df[hue] == cat][target]
            target_values = pd.to_numeric(target_series, errors='coerce')
            
            if target_values.isna().all():
                target_mean = target_median = np.nan
            else:
                target_mean = target_values.mean()
                target_median = target_values.median()
            
            legend_records.append({
                "Категория": str(cat),
                "Частота": count,
                "Доля (%)": pct,
                "Среднее по таргету": target_mean,
                "Медиана по таргету": target_median,
                # НЕТ колонки "Цвет"!
            })
        
        legend_df = pd.DataFrame(legend_records)
        
        # Маппинг категория → цвет
        color_map = {}
        for i, (cat, _) in enumerate(value_counts.items()):
            color_map[str(cat)] = matplotlib.colors.to_hex(colors[i])

        def legend_styler(s: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
            def get_text_color(bg_color):
                try:
                    bg_hex = matplotlib.colors.to_hex(bg_color)
                except:
                    bg_hex = "#ffffff"
                r = int(bg_hex[1:3], 16) / 255.0
                g = int(bg_hex[3:5], 16) / 255.0
                b = int(bg_hex[5:7], 16) / 255.0
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "white" if luminance < 0.5 else "black"

            def apply_bg_color(col):
                styles = []
                for val in col:
                    bg = color_map.get(val, "#ffffff")
                    text_color = get_text_color(bg)
                    styles.append(f"background-color: {bg}; color: {text_color};")
                return styles

            return (
                s.set_properties(subset=["Категория"], **{"text-align": "left"})
                .background_gradient(subset=["Доля (%)"], cmap="coolwarm")
                .apply(apply_bg_color, subset=["Категория"])
            )

        # Отображаем таблицу 
        display_table(
            legend_df[["Категория", "Среднее по таргету", "Медиана по таргету", "Частота", "Доля (%)"]],
            rows=len(legend_df),
            float_precision=0,
            styler_func=legend_styler
        )

    # Корреляция с подсветкой и силой связи
    method_name = "Пирсона" if method == "pearson" else "Спирмана"
    print(f"\nКорреляции ({method_name}) с целевым признаком '{target_label}': ")

    correlations = df[num_cols + [target]].corr(method=method)[target].drop(target)

    # НАЗНАЧАЕМ ЦВЕТА ПРИЗНАКАМ
    feature_colors = {}
    palette_colors = sns.color_palette(palette, n_colors=len(num_cols))
    for i, col in enumerate(num_cols):
        col_name, _ = label_for_column(col)
        feature_colors[col_name] = matplotlib.colors.to_hex(palette_colors[i])

    # Подготавливаем данные для таблицы
    corr_records = []
    for col in correlations.index:
        corr_value = correlations[col]
        col_name, col_desc = label_for_column(col)
        
        abs_corr = abs(corr_value)
        if abs_corr < 0.1:
            strength = "очень слабая"
        elif abs_corr < 0.3:
            strength = "слабая"
        elif abs_corr < 0.5:
            strength = "умеренная"
        else:
            strength = "сильная"
        
        corr_records.append({
            "Признак": col_name,
            "Описание": col_desc,
            "Корреляция": corr_value,
            "Сила связи": strength
        })

    corr_df = pd.DataFrame(corr_records).sort_values("Корреляция", key=abs, ascending=False)

    strength_colors = {
        "очень слабая": "#e8f5e8",
        "слабая": "#c8e6c9",
        "умеренная": "#a5d6a7",
        "сильная": "#4caf50"
    }

    # СТИЛИЗАЦИЯ С ЦВЕТАМИ ПРИЗНАКОВ
    def _style_corr_table(styler):
        styler = styler.background_gradient(subset=["Корреляция"], cmap="coolwarm")
        styler = styler.applymap(
            lambda x: f"background-color: {strength_colors.get(x, '')}; color: black",
            subset=["Сила связи"]
        )

        return styler

    display_table(
        corr_df[["Признак", "Описание", "Корреляция", "Сила связи"]],
        rows=len(corr_df),
        float_precision=3,
        styler_func=_style_corr_table
    )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_mixed_correlation • Рассчитывает и визуализирует корреляционную матрицу для смешанных типов данных (числовые и категориальные)
# UPD: add method: Literal['Pearson', 'Spearman'] = 'Pearson'
def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Рассчитывает статистику Cramer's V для корреляции категориальных признаков
    
    Параметры:
        x: pd.Series - первый категориальный признак
        y: pd.Series - второй категориальный признак
    
    Возвращает:
        float - коэффициент корреляции Cramer's V
    """
    contingency_table = pd.crosstab(x, y)
    contingency_table = contingency_table.loc[(contingency_table != 0).any(axis=1)]
    contingency_table = contingency_table.loc[:, (contingency_table != 0).any(axis=0)]
    
    if contingency_table.empty:
        return 0.0
    
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    
    if min_dim == 0:
        return 0.0
    
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return min(cramers_v, 1.0)


def plot_mixed_correlation(
    df: pd.DataFrame, 
    figsize: tuple = None,
    annot: bool = True,
    cmap: str = 'RdBu_r',
    threshold: float = None,
    hide_upper: bool = True,
    show_grid: bool = True,
    show_diagonal: bool = False,
    exclude: list = None,
    include: list = None,
    precision: int = 3,
    filter_no_corr: bool = True,
    auto_font_size: bool = True,
    method: Literal['pearson', 'spearman'] = 'pearson'
) -> pd.DataFrame:
    """
    Строит корреляционную матрицу и таблицу связей для датафрейма со смешанными типами признаков.

    Описание:
        Поддерживает три типа корреляций: Pearson (число-число), Cramer’s V (категория-категория) и point-biserial (число-категория).
        Визуализирует тепловую карту с опциональной маской, порогом значимости и адаптивным форматированием.
        Дополнительно выводит отсортированную таблицу пар признаков с корреляциями выше порога.

    Параметры:
        df: pd.DataFrame - исходный датафрейм
        figsize: tuple - размер графика (ширина, высота); если None - подбирается автоматически
        annot: bool - отображать ли числовые значения на тепловой карте
        cmap: str - цветовая палитра (по умолчанию 'RdBu_r')
        threshold: float - минимальное абсолютное значение корреляции для отображения (None - без фильтрации)
        hide_upper: bool - скрывать верхний треугольник матрицы (по умолчанию True)
        show_grid: bool - отображать сетку между ячейками
        show_diagonal: bool - показывать диагональ (корреляция признака с самим собой)
        exclude: list - список колонок для исключения из анализа
        include: list - если задан, анализируются только указанные колонки
        precision: int - количество знаков после запятой в аннотациях и таблице
        filter_no_corr: bool - исключать признаки, у которых все корреляции ниже порога
        auto_font_size: bool - автоматически подбирать размер шрифта под количество признаков

    Возвращаемое значение:
        pd.DataFrame - полная корреляционная матрица (до фильтрации по порогу)
    """
    if method not in ('pearson', 'spearman'):
        raise ValueError("Параметр 'method' должен быть 'pearson' или 'spearman'")

    # Автоматический поиск имени и описания датасета
    dataset_name, dataset_desc = label_for_dataset(df)
    print(f"🗃️ Датасет '{dataset_name}' 📋 {dataset_desc}")

    # Получаем список колонок на основе exclude/include
    all_cols = df.columns.tolist()
    
    if exclude:
        all_cols = [col for col in all_cols if col not in exclude]
    
    if include:
        all_cols = [col for col in all_cols if col in include]
    
    # Разделяем на числовые и категориальные, проверяя типы
    numeric_cols = []
    categorical_cols = []
    
    for col in all_cols:
        if col in df.select_dtypes(include=[np.number]).columns:
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    all_cols = [col for col in all_cols if col in numeric_cols or col in categorical_cols]
    
    if not all_cols:
        print("Нет доступных колонок для анализа корреляций")
        return pd.DataFrame()
    
    n_cols = len(all_cols)
    corr_matrix = pd.DataFrame(
        np.zeros((n_cols, n_cols)), 
        index=all_cols, 
        columns=all_cols
    )
    
    for col1, col2 in combinations(all_cols, 2):
        idx1, idx2 = all_cols.index(col1), all_cols.index(col2)
        
        series1 = df[col1].dropna()
        series2 = df[col2].dropna()
        
        common_idx = series1.index.intersection(series2.index)
        if len(common_idx) == 0:
            corr_val = 0.0
        elif col1 in numeric_cols and col2 in numeric_cols:
            series1_clean = pd.to_numeric(series1.loc[common_idx], errors='coerce')
            series2_clean = pd.to_numeric(series2.loc[common_idx], errors='coerce')
            mask = ~(series1_clean.isna() | series2_clean.isna())
            if mask.sum() > 1:
                corr_val = pd.Series.corr(series1_clean[mask], series2_clean[mask], method=method)
            else:
                corr_val = 0.0
        elif col1 in categorical_cols and col2 in categorical_cols:
            corr_val = calculate_cramers_v(
                series1.loc[common_idx], 
                series2.loc[common_idx]
            )
        else:
            num_col = col1 if col1 in numeric_cols else col2
            cat_col = col2 if col2 in categorical_cols else col1
            
            series_num = df[num_col].loc[common_idx]
            series_cat = df[cat_col].loc[common_idx]
            
            series_num_clean = pd.to_numeric(series_num, errors='coerce').dropna()
            series_cat_clean = series_cat.loc[series_num_clean.index]
            
            mask = ~(series_num_clean.isna() | series_cat_clean.isna())
            if mask.sum() > 1:
                cat_encoded = pd.Categorical(series_cat_clean[mask]).codes
                corr_val, _ = pointbiserialr(series_num_clean[mask], cat_encoded)
            else:
                corr_val = 0.0
        
        corr_matrix.loc[col1, col2] = corr_val
        corr_matrix.loc[col2, col1] = corr_val
    
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Фильтрация признаков без корреляций (если включено)
    if filter_no_corr and threshold is not None:
        abs_corr_matrix = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr_matrix.values, 0)
        max_corr_per_feature = abs_corr_matrix.max(axis=1)
        active_features = max_corr_per_feature[max_corr_per_feature >= threshold].index.tolist()
        
        if active_features:
            corr_matrix = corr_matrix.loc[active_features, active_features]
            print(f"Отфильтровано {len(all_cols) - len(active_features)} признаков без корреляций")
        else:
            print("Все признаки имеют корреляции ниже порога")
            return pd.DataFrame()
    
    # Автоматический размер фигуры (если не задан)
    if figsize is None:
        size_per_feature = max(0.4, 8 / max(len(corr_matrix), 10))
        figsize = (max(8, len(corr_matrix) * size_per_feature), 
                   max(6, len(corr_matrix) * size_per_feature))
    
    # Создаем маску
    if hide_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
        if threshold is not None:
            threshold_mask = np.abs(corr_matrix) < threshold
            mask = mask | threshold_mask
    else:
        mask = np.abs(corr_matrix) < threshold if threshold is not None else None
        if not show_diagonal:
            mask = mask | np.eye(len(corr_matrix), dtype=bool)
    
    # Создаем формат для отображения
    fmt_str = f'.{precision}f'
    
    # Создаем тепловую карту
    plt.figure(figsize=figsize)
    
    # Автоматический размер шрифта (всегда адаптируется к количеству признаков)
    annot_kws = {}
    if auto_font_size:
        font_size = max(5, min(10, 14 - len(corr_matrix) // 2))
        annot_kws = {'size': font_size}
    
    sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        center=0,
        square=True,
        fmt=fmt_str,
        cbar_kws={'shrink': 0.8},
        mask=mask,
        linewidths=0.5 if show_grid else 0,
        annot_kws=annot_kws
    )
    
    method_name = "Пирсона" if method == "pearson" else "Спирмана"

    plt.suptitle(
        f'Корреляционная матрица ({method_name}, порог: {threshold})\n'
        f' • числовые: {method.capitalize()}\n'
        f' • категориальные: Cramer\'s V\n'
        f' • смешанные: Point-biserial', 
        fontsize=max(10, min(10, 14 - len(corr_matrix) // 3)), 
        x=0.01,
        y=0.98,
        ha='left'
    )

    
    plt.xticks(rotation=45, ha='right', fontsize=max(8, min(6, 12 - len(corr_matrix) // 4)))
    plt.yticks(rotation=0, fontsize=max(8, min(6, 12 - len(corr_matrix) // 4)))
    plt.tight_layout()
    plt.show()
    
    # Выводим таблицу связей
    print(f"Таблица корреляций (порог: {threshold})")
    
    pairs_data = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_val = corr_matrix.iloc[i, j]
            if threshold is None or abs(corr_val) >= threshold:
                feature_1_name, feature_1_desc = label_for_column(corr_matrix.index[i], separator='•')
                feature_2_name, feature_2_desc = label_for_column(corr_matrix.columns[j], separator='•')

                pairs_data.append({
                    'Признак 1': f'{feature_1_name}{feature_1_desc}',
                    'Корреляция': corr_val,
                    'Признак 2': f'{feature_2_name}{feature_2_desc}'
                })
    
    if pairs_data:
        pairs_df = pd.DataFrame(pairs_data)
        pairs_df = pairs_df.sort_values(by='Корреляция', key=abs, ascending=False)
        #pairs_df.insert(0, '#', range(1, len(pairs_df) + 1))
        
        from matplotlib.colors import LinearSegmentedColormap
        colormap = plt.cm.RdBu_r
        
        def color_corr(val):
            normalized_val = (val + 1) / 2
            rgba_color = colormap(normalized_val)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgba_color[0] * 255),
                int(rgba_color[1] * 255), 
                int(rgba_color[2] * 255)
            )
            return f'background-color: {hex_color}; color: white' if normalized_val < 0.2 or normalized_val > 0.8 else f'background-color: {hex_color}; color: black'
        
        fmt_table = f'{{:.{precision}f}}'

        display_table(
            pairs_df,
            rows=len(pairs_df),
            float_precision=precision,
            styler_func=lambda s: s.background_gradient(
                subset=["Корреляция"], 
                cmap="RdBu_r", 
                low=0.3, 
                high=0.7
            )
        )
    else:
        print("Нет корреляций, превышающих заданный порог")
    
    return corr_matrix


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def plot_pairwise_correlations(
    df: pd.DataFrame,
    hue_col: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    palette: str = 'tab10',
    threshold: Optional[float] = None,
    show_report: bool = True,
    report_threshold: float = 0.3,
    precision: int = 3,
    base_point_size: int = 20,
    dpi: int = 150,
    method: Literal['pearson', 'spearman'] = 'spearman' 
) -> None:
    """
    Визуализирует парные зависимости числовых признаков с корреляционным отчётом.
    
    Описание:
        Создаёт матрицу scatter-графиков с коэффициентами корреляции (Пирсона или Спирмана). 
        Поддерживает цветовую группировку по категориальному признаку (hue_col).
        Выводит таблицу корреляций с градиентной подсветкой и интерпретацией силы связи.
        Для hue_col отображает статистику по группам в виде таблицы.

    Параметры:
        df: pd.DataFrame - датафрейм для анализа
        hue_col: Optional[str] - категориальный признак для цветовой группировки
        include: Optional[List[str]] - анализировать только указанные колонки
        exclude: Optional[List[str]] - исключить указанные колонки из анализа
        palette: str - цветовая палитра seaborn (по умолчанию 'tab10')
        threshold: Optional[float] - отбирать только признаки с |r| ≥ threshold
        show_report: bool - показывать корреляционный отчёт (по умолчанию True)
        report_threshold: float - порог для включения пары в отчёт (по умолчанию 0.3)
        precision: int - знаки после запятой для корреляций (по умолчанию 3)
        base_point_size: int - базовый размер точек (по умолчанию 20)
        dpi: int - разрешение графика (по умолчанию 150)
        method: Literal['pearson', 'spearman'] - метод расчёта корреляции (по умолчанию 'spearman')

    Возвращаемое значение:
        None - отображает графики и корреляционный отчёт
    """
    # ЗАЩИТА ОТ ПУСТОГО ДАТАФРЕЙМА
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return
    
    if method not in ("pearson", "spearman"):
        raise ValueError("Параметр 'method' должен быть 'pearson' или 'spearman'")

    # ПОДПИСЬ ДАТАСЕТА
    dataset_name, dataset_desc = label_for_dataset(df)
    print(f"🗃️ Датасет {dataset_name} • {dataset_desc}\n")

    # ОПРЕДЕЛЕНИЕ ЧИСЛОВЫХ КОЛОНОК
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if include is not None:
        numeric_cols = [col for col in include if col in numeric_cols]
    if exclude is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    if not numeric_cols:
        print("⚠️ Нет числовых колонок для анализа.")
        return
    
    if len(numeric_cols) == 1:
        print(f"⚠️ Только один числовой признак: {numeric_cols[0]}. Парный анализ невозможен.")
        return

    # ФИЛЬТРАЦИЯ ПО ПОРОГУ КОРРЕЛЯЦИИ
    if threshold is not None:
        corr_matrix_full = df[numeric_cols].corr()
        relevant_pairs = []
        for i, j in combinations(range(len(numeric_cols)), 2):
            r = corr_matrix_full.iloc[i, j]
            if abs(r) >= threshold:
                relevant_pairs.append((numeric_cols[i], numeric_cols[j]))
        
        if not relevant_pairs:
            print(f"⚠️ Нет пар признаков с корреляцией ≥ {threshold}")
            return
        
        relevant_features = list(set([col for pair in relevant_pairs for col in pair]))
        numeric_cols = relevant_features
        print(f"🔍 Отобрано признаков ({len(numeric_cols)}) с корреляцией ≥ {threshold}")
        for col in numeric_cols:
            col_name, col_desc = label_for_column(col, separator='•')
            col_label = f"{col_name}{col_desc}" if col_desc else col_name
            print(f"   • {col_label}")

    # ПОДГОТОВКА ДАННЫХ
    cols_to_plot = numeric_cols + ([hue_col] if hue_col else [])
    data_to_plot = df[cols_to_plot].dropna(subset=numeric_cols)

    if len(data_to_plot) == 0:
        print("⚠️ Нет данных без пропусков в числовых колонках.")
        return

    # ОБРАБОТКА HUE_COL
    hue_col_final = hue_col
    if hue_col and hue_col in data_to_plot.columns:
        if pd.api.types.is_numeric_dtype(data_to_plot[hue_col]):
            unique_count = data_to_plot[hue_col].nunique()
            if unique_count <= 15:
                data_to_plot = data_to_plot.copy()
                data_to_plot[hue_col] = data_to_plot[hue_col].astype(str)
                print(f"🔄 Колонка '{hue_col}' конвертирована в категориальную ({unique_count} уникальных значений)")
            else:
                print(f"⚠️ Колонка '{hue_col}' содержит {unique_count} уникальных значений и не подходит для цветовой группировки")
                hue_col_final = None
        else:
            print(f"📋 Колонка '{hue_col}' используется как категориальная ({data_to_plot[hue_col].nunique()} уникальных значений)")
    elif hue_col:
        print(f"⚠️ Колонка '{hue_col}' не найдена в данных")
        hue_col_final = None

    # ЗАЩИТА ОТ KDE-ОШИБОК
    # Удаляем категории с нулевой дисперсией в числовых признаках
    if hue_col_final:
        valid_categories = []
        for cat in data_to_plot[hue_col_final].unique():
            cat_data = data_to_plot[data_to_plot[hue_col_final] == cat]
            # Проверяем, есть ли вариативность хотя бы в одном числовом признаке
            has_variance = any(cat_data[col].nunique() > 1 for col in numeric_cols)
            if has_variance:
                valid_categories.append(cat)
            else:
                print(f"⚠️ Категория '{cat}' в '{hue_col_final}' имеет нулевую дисперсию - исключена из визуализации")
        
        if not valid_categories:
            print("⚠️ Все категории в hue_col имеют нулевую дисперсию - отключаем группировку")
            hue_col_final = None
        else:
            data_to_plot = data_to_plot[data_to_plot[hue_col_final].isin(valid_categories)]

    # ПОСТРОЕНИЕ ГРАФИКА
    try:
        n_samples = len(data_to_plot)
        point_size = max(5, min(20, base_point_size * (1000 / n_samples)))

        g = sns.pairplot(
            data_to_plot,
            hue=hue_col_final,
            palette=palette,
            diag_kind='kde',
            corner=True,
            plot_kws={'alpha': 0.7, 's': point_size},
            diag_kws={'shade': True},
            height=2.5,
            aspect=1.0,  
        )

        n_features = len(numeric_cols)
        g.fig.set_size_inches(2 * n_features, 2 * n_features)
        g.fig.set_dpi(dpi)

        # Удаляем легенду
        if hue_col_final is not None:
            for ax in g.axes.flat:
                if ax is not None and ax.legend_ is not None:
                    ax.legend_.remove()
            if hasattr(g, '_legend'):
                g._legend.remove()

        # Добавляем корреляции и рамки
        corr_matrix = df[numeric_cols].corr(method=method)
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax = g.axes[i, j]
                if ax is None:
                    continue

                # Серая рамка
                rect_border = plt.Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    facecolor='none',
                    edgecolor='gray',
                    linewidth=1.5,
                    zorder=4
                )
                ax.add_patch(rect_border)

                # Диагональ - только заголовок
                if i == j:
                    col_name, col_desc = label_for_column(numeric_cols[i], separator='')
                    col_label = f"{col_name}{col_desc}" if col_desc else col_name
                    ax.set_title(col_label, fontsize=5.5, pad=5, loc='left')
                    continue  # ← Не показываем ось Y на диагонали

                # Корреляция на графике
                r = corr_matrix.iloc[i, j]
                abs_r = abs(r)
                if abs_r >= 0.7:
                    text_color = 'darkred'
                    fontweight = 'bold'
                elif abs_r >= 0.5:
                    text_color = 'red'
                    fontweight = 'bold'
                else:
                    text_color = 'gray'
                    fontweight = 'normal'

                ax.text(
                    0.05, 0.95, f'r={r:.{precision}f}',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    color=text_color,
                    fontweight=fontweight,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
                )

        # Легенда
        if hue_col_final is not None:
            from matplotlib.patches import Patch
            unique_cats = data_to_plot[hue_col_final].dropna().unique()
            colors = sns.color_palette(palette, n_colors=len(unique_cats))
            legend_elements = [
                Patch(facecolor=color, edgecolor='black', label=str(cat))
                for color, cat in zip(colors, unique_cats)
            ]
            g.fig.legend(
                legend_elements,
                [str(cat) for cat in unique_cats],
                loc='upper right',
                bbox_to_anchor=(0.99, 0.84),
                ncol=1,
                title=f"Категории по {hue_col_final}",
                frameon=True,
                fontsize=6,
                title_fontsize=7
            )

        # Заголовок
        method_name = "Пирсона" if method == "pearson" else "Спирмана"
        g.fig.suptitle(
            f"Парные зависимости с корреляцией (r, метод: {method_name})\n{dataset_desc}",
            fontsize=10,
            fontweight='bold',
            x=0.98,
            y=0.92,
            ha='right'
        )

        # Настройка отступов - без tight_layout
        g.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

        # Убираем лишние метки на диагонали
        for i in range(len(numeric_cols)):
            ax = g.axes[i, i]
            if ax is not None:
                ax.set_ylabel('')  # Убираем метку Y на диагонали
                ax.set_xlabel('')  # Убираем метку X на диагонали

        # Отображаем график
        plt.show()

    except Exception as e:
        print(f"❌ Ошибка при построении графика: {str(e)}")
        print("💡 Попробуйте уменьшить количество признаков или отключить hue_col")

        # КОРРЕЛЯЦИОННЫЙ ОТЧЁТ
        if show_report:
            pairs_data = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= report_threshold:
                        f1_name, f1_desc = label_for_column(corr_matrix.index[i], separator='•')
                        f2_name, f2_desc = label_for_column(corr_matrix.columns[j], separator='•')
                        pairs_data.append({
                            "Признак 1": f"{f1_name}{f1_desc}" if f1_desc else f1_name,
                            "Корреляция": corr_val,
                            "Признак 2": f"{f2_name}{f2_desc}" if f2_desc else f2_name
                        })
            
            if not pairs_data:
                print(f"🔸 Нет пар с |r| ≥ {report_threshold}")
                return
            
            pairs_df = pd.DataFrame(pairs_data)
            pairs_df = pairs_df.sort_values("Корреляция", key=abs, ascending=False)
            #pairs_df.insert(0, "#", range(1, len(pairs_df) + 1))
            
            print(f"\n📊 Корреляции признаков (порог |r| ≥ {report_threshold}):")
            display_table(
                pairs_df,
                rows=len(pairs_df),
                float_precision=precision,
                styler_func=lambda s: s.background_gradient(
                    subset=["Корреляция"], 
                    cmap="RdBu_r", 
                    low=0.3, 
                    high=0.7
                )
            )
            
            # Статистика по группам
            unique_cats = data_to_plot[hue_col_final].dropna().unique()
            colors = sns.color_palette(palette, n_colors=len(unique_cats))
            color_map = {str(cat): matplotlib.colors.to_hex(colors[i]) for i, cat in enumerate(unique_cats)}
            
            # Статистика по категориям (можно использовать value_counts, но маппинг - по unique_cats!)
            group_counts = data_to_plot[hue_col_final].value_counts()
            total = len(data_to_plot)
            
            group_records = []
            for cat in unique_cats:  # ← итерируемся по правильному порядку!
                count = group_counts.get(cat, 0)
                pct = count / total * 100 if total > 0 else 0
                group_records.append({
                    "Категория": str(cat),
                    "Количество": count,
                    "Доля (%)": pct,
                })
            
            groups_df = pd.DataFrame(group_records)

            def group_styler(s: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
                def get_text_color(bg_color):
                    try:
                        bg_hex = matplotlib.colors.to_hex(bg_color)
                    except:
                        bg_hex = "#ffffff"
                    r = int(bg_hex[1:3], 16) / 255.0
                    g = int(bg_hex[3:5], 16) / 255.0
                    b = int(bg_hex[5:7], 16) / 255.0
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    return "white" if luminance < 0.5 else "black"

                def apply_bg_color(col):
                    styles = []
                    for val in col:
                        bg = color_map.get(val, "#ffffff")
                        text_color = get_text_color(bg)
                        styles.append(f"background-color: {bg}; color: {text_color};")
                    return styles

                return (
                    s.set_properties(subset=["Категория"], **{"text-align": "left"})
                    .background_gradient(subset=["Доля (%)"], cmap="coolwarm")
                    .apply(apply_bg_color, subset=["Категория"])
                )

            display_table(
                groups_df[["Категория", "Количество", "Доля (%)"]],
                rows=len(groups_df),
                float_precision=0,
                styler_func=group_styler
            )
            
            print(f"\n💡 Советы по интерпретации:")
            print(f"   • Если точки одного цвета образуют отдельные облака - у группы свои особенности")
            print(f"   • Если точки разных цветов перемешаны - корреляция не зависит от {hue_label}")

        # ИНТЕРПРЕТАЦИЯ
        strong = pairs_df[pairs_df["Корреляция"].abs() >= 0.7]
        moderate = pairs_df[(pairs_df["Корреляция"].abs() >= 0.5) & (pairs_df["Корреляция"].abs() < 0.7)]
        
        print(f"\n🧠 Интерпретация силы связей:")
        print(f"   • Всего записей: {len(data_to_plot):,}")
        print(f"   • Числовых признаков: {len(numeric_cols)}")
        if len(strong) > 0:
            print(f"   🔥 {len(strong)} очень сильных связей (|r| ≥ 0.7) - возможна мультиколлинеарность")
        if len(moderate) > 0:
            print(f"   ⚡ {len(moderate)} умеренных связей (0.5 ≤ |r| < 0.7) - полезны для модели")
        if len(pairs_df) == len(strong) + len(moderate):
            print(f"   🔸 Остальные связи - слабые (|r| < 0.5)")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_categorical_distribution: Визуализирует распределение категориального признака
def plot_categorical_distribution(
    df: pd.DataFrame,
    feature: str,
    hue: Optional[str] = None,
    report: Literal["summary", "full"] = "summary",
    palette: str = "tab10",
    min_freq_threshold: float = 0.01,  # 1%
) -> None:
    """
    Визуализирует распределение категориального признака с учётом пропусков и пробельных значений.
    
    Описание:
        Генерирует:
        1. Столбчатую диаграмму частот с линией медианы доли и цветовой подсветкой редких категорий.
        2. (Опционально) Boxplot по hue, если hue - числовой признак.
        Особенности:
        - Пропуски (NaN) отображаются как отдельная категория **[пропуски]**.
        - Строки, состоящие только из пробелов, отображаются как **[пробелы]**.
        - Распределение всегда включает 100% строк датафрейма.
        - Медиана и дисбаланс рассчитываются только по «настоящим» категориям.
        Выводит:
        - Чек-лист проблем (дисбаланс, мусор, редкие категории),
        - (В режиме 'full') таблицу частот с цветовой меткой и статистикой.
        Использует глобальные справочники COLUMN_DESCRIPTIONS и DATASET_DESCRIPTIONS.

    Параметры:
        df: pd.DataFrame - датафрейм для анализа
        feature: str - имя категориального признака
        hue: Optional[str] - числовой или категориальный признак для группировки
        report: Literal["summary", "full"] - уровень детализации
        palette: str - палитра seaborn (по умолчанию 'tab10')
        min_freq_threshold: float - порог для выделения редких категорий (по умолчанию 0.01 = 1%)

    Возвращаемое значение:
        None - вывод через display_table и графики
    """
    if df.empty:
        print("⚠️ Датафрейм пуст")
        return

    if feature not in df.columns:
        raise ValueError(f"Признак '{feature}' не найден в датафрейме")

    # Подпись признака
    feature_name, feature_desc = label_for_column(feature, separator='•')
    feature_label = f"{feature_name}{feature_desc}" if feature_desc else feature_name

    # Создаём копию для обработки
    df_processed = df[[feature]].copy()

    # Заменяем NaN на [пропуски]
    missing_mask = df_processed[feature].isna()
    n_missing = missing_mask.sum()
    df_processed.loc[missing_mask, feature] = "[пропуски]"

    # Заменяем строки из одних пробелов на [одни пробелы]
    whitespace_mask = df_processed[feature].astype(str).str.match(r'^\s*$') & ~missing_mask
    n_whitespace = whitespace_mask.sum()
    df_processed.loc[whitespace_mask, feature] = "[одни пробелы]"

    n_total = len(df)
    missing_pct = n_missing / n_total * 100 if n_total > 0 else 0
    whitespace_pct = n_whitespace / n_total * 100 if n_total > 0 else 0

    # Частоты (включая служебные категории)
    value_counts = df_processed[feature].value_counts(dropna=False)
    freq_df = pd.DataFrame({
        'Категория': value_counts.index,
        'Частота': value_counts.values,
        'Доля (%)': (value_counts.values / n_total * 100)
    }).reset_index(drop=True)

    # Медиана доли - только по «настоящим» категориям
    clean_freqs = freq_df[~freq_df['Категория'].isin(["[пропуски]", "[одни пробелы]"])]
    median_pct = clean_freqs['Доля (%)'].median() if not clean_freqs.empty else 0.0

    # Определение проблем
    issues = []
    rare_cats = clean_freqs[clean_freqs['Доля (%)'] < min_freq_threshold * 100]['Категория'].tolist()
    if rare_cats:
        issues.append(f"редкие категории (<{min_freq_threshold:.0%}): {len(rare_cats)} шт")

    # Проверка на мусор (исключая служебные категории)
    str_series = df[feature].dropna().astype(str)
    non_whitespace = ~str_series.str.match(r'^\s*$')
    junk_mask = non_whitespace & str_series.str.lower().isin(['null', 'n/a', 'nan', 'none', ''])
    junk_values = str_series[junk_mask]
    if not junk_values.empty:
        issues.append(f"мусорные значения: {junk_values.nunique()} типов")

    # Дисбаланс - только по «настоящим» категориям
    if not clean_freqs.empty:
        max_pct = clean_freqs['Доля (%)'].max()
        if max_pct > 95:
            top_cat = clean_freqs.loc[clean_freqs['Доля (%)'].idxmax(), 'Категория']
            issues.append(f"сильный дисбаланс: '{top_cat}' - {max_pct:.1f}%")

    # ВЫВОД ОТЧЁТА
    dataset_name, dataset_desc = label_for_dataset(df, separator="•")
    print(f"\n🗃️ Датасет: {dataset_name}{dataset_desc}")
    print(f"🏷️ Признак: {feature_label}")

    if n_missing > 0:
        print(f"⚠️ Пропусков (NaN): {n_missing} ({missing_pct:.1f}%)")
    else:
        print("✔️ Пропусков (NaN) нет")

    if n_whitespace > 0:
        print(f"⚠️ Строк из одних пробелов: {n_whitespace} ({whitespace_pct:.1f}%)")

    if issues:
        print("🚨 Проблемы:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print("💎 Качество данных хорошее")

    # ВИЗУАЛИЗАЦИЯ
    n_plots = 2 if hue and hue in df.columns else 1

    # Адаптивная высота
    if n_plots == 1:
        height = min(5, max(3, len(freq_df) * 0.4))
        figsize = (16, height)
    else:
        height = max(5, len(freq_df) * 0.4)
        figsize = (16, height)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
    ax1 = axes[0, 0]

    # Цвета: служебные - серые, остальные - из палитры
    colors = []
    palette_colors = sns.color_palette(palette, n_colors=len(freq_df))
    for i, row in freq_df.iterrows():
        if row['Категория'] in ["[пропуски]", "[пробелы]"]:
            colors.append('lightgray')
        else:
            colors.append(palette_colors[i % len(palette_colors)])

    # Столбчатая диаграмма
    bars = ax1.bar(
        range(len(freq_df)),
        freq_df['Частота'],
        color=colors,
        edgecolor='white',
        linewidth=0.8
    )
    ax1.set_xticks(range(len(freq_df)))
    ax1.set_xticklabels(freq_df['Категория'], rotation=0, ha='right', fontsize=9)
    ax1.set_ylabel('Частота', fontsize=10)
    ax1.set_title(f"Распределение: {feature_label}", fontsize=11)

    # Линия медианы доли (в шкале частот)
    median_freq = median_pct / 100 * n_total
    ax1.axhline(median_freq, color='darkred', linestyle='--', linewidth=1.2, label=f'Медиана доли ({median_pct:.1f}%)')
    ax1.legend(fontsize=9)

    # Второй график: boxplot по hue (если hue числовой)
    if n_plots == 2:
        ax2 = axes[0, 1]
        if pd.api.types.is_numeric_dtype(df[hue]):
            sns.boxplot(
                data=df.dropna(subset=[feature, hue]),
                x=feature,
                y=hue,
                ax=ax2,
                palette=palette,
                flierprops=dict(
                    marker='o',
                    markerfacecolor="#DE1885",
                    markeredgecolor="#560A34",
                    markersize=8,
                    alpha=0.5
                ),
                medianprops=dict(color='darkred', linewidth=2),
                boxprops=dict(alpha=0.5, linewidth=1.5)
            )
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='right')
            ax2.set_title(f"{feature_label} vs {hue}", fontsize=11)
            hue_name, hue_desc = label_for_column(hue, separator='•')
            ax2.set_ylabel(f"{hue_name}{hue_desc}")
        else:
            crosstab = pd.crosstab(df[feature], df[hue])
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title(f"Совместное распределение: {feature_label} и {hue}", fontsize=11)

    plt.tight_layout()
    plt.show()

    # Отчёт
    if report == "full" and not freq_df.empty:
        # Создаём маппинг категория → цвет (без HTML!)
        color_map = {}
        for i, cat in enumerate(freq_df['Категория']):
            if cat in ["[пропуски]", "[одни пробелы]"]:
                color_map[cat] = "lightgray"
            else:
                color_map[cat] = matplotlib.colors.to_hex(palette_colors[i % len(palette_colors)])

        def styler(s: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
            def get_text_color(bg_color):
                try:
                    bg_hex = matplotlib.colors.to_hex(bg_color)
                except:
                    bg_hex = "#ffffff"
                r = int(bg_hex[1:3], 16) / 255.0
                g = int(bg_hex[3:5], 16) / 255.0
                b = int(bg_hex[5:7], 16) / 255.0
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "white" if luminance < 0.5 else "black"

            def apply_bg_color(col):
                styles = []
                for val in col:
                    bg = color_map.get(val, "#ffffff")
                    text_color = get_text_color(bg)
                    styles.append(f"background-color: {bg}; color: {text_color};")
                return styles

            return (
                s.format({'Доля (%)': '{:.2f}%'})
                .set_properties(subset=['Категория'], **{'text-align': 'left'})
                .background_gradient(subset=['Доля (%)'], cmap='coolwarm')
                .apply(apply_bg_color, subset=['Категория'])
            )

        print(f"\n📋 Таблица частот ({len(freq_df)} категорий):")
        display_table(
            freq_df[['Категория', 'Частота', 'Доля (%)']],
            rows=len(freq_df),
            float_precision=2,
            styler_func=styler
        )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_phik_correlation: Строит корреляционную матрицу на основе Phik - меры ассоциации для смешанных типов данных
def plot_phik_correlation(
    df: pd.DataFrame,
    interval_cols: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    threshold: float = 0.3,
    figsize: Optional[tuple] = None,
    cmap: str = 'Blues',
    annot: bool = True,
    precision: int = 3,
    show_report: bool = True,
    report_threshold: Optional[float] = None,
    show_triangle: str = 'lower',  # 'lower', 'upper', 'full'
    hide_empty_labels: bool = True,
    grid_color: str = "#c5d1e0",
    show_border: bool = True,
    border_color: str = "#407B8D",
    dpi: int = 150
) -> pd.DataFrame:
    """
    Строит корреляционную матрицу на основе Phik - меры ассоциации для смешанных типов данных.
    
    Описание:
        Использует библиотеку `phik` для расчёта ассоциаций между признаками любого типа:
        - числовые ↔ числовые → корреляция Пирсона (внутри phik),
        - категориальные ↔ категориальные → Cramér’s V,
        - числовые ↔ категориальные → корреляция с рангами.
        Поддерживает фильтрацию, треугольную/полную визуализацию, автоматическое определение
        числовых признаков и гибкую настройку отображения.
    
    Особенности:
        • Работает со смешанными типами данных без предварительного кодирования,
        • Устойчив к пропускам и низкой кардинальности,
        • Выводит интерпретируемый отчёт с силой связей и рекомендациями,
        • Адаптивный размер фигуры и шрифтов под количество признаков,
        • Явное управление DPI - игнорирует глобальные plt.rcParams.

    Параметры:
        df: pd.DataFrame - исходный датафрейм
        interval_cols: Optional[List[str]] - список числовых колонок (если не задан - определяется автоматически)
        include/exclude: Optional[List[str]] - фильтрация колонок
        threshold: float - порог значимости Phik для визуализации (по умолчанию 0.3)
        figsize: Optional[tuple] - размер фигуры (если None - подбирается автоматически)
        cmap: str - цветовая палитра (по умолчанию 'Blues')
        annot: bool - отображать ли числовые значения в ячейках (по умолчанию True)
        precision: int - знаки после запятой в аннотациях (по умолчанию 3)
        show_report: bool - показывать ли сводную таблицу ассоциаций (по умолчанию True)
        report_threshold: Optional[float] - порог для отчёта (если None - используется threshold)
        show_triangle: str - 'lower', 'upper' или 'full' (по умолчанию 'lower')
        hide_empty_labels: bool - скрывать признаки без значимых связей (по умолчанию True)
        grid_color: str - цвет сетки между ячейками (по умолчанию "#c5d1e0")
        show_border: bool - рисовать ли внешнюю рамку (по умолчанию True)
        border_color: str - цвет внешней рамки (по умолчанию "#407B8D")
        dpi: int - разрешение графика (по умолчанию 150)

    Возвращаемое значение:
        pd.DataFrame - полная Phik-матрица

    Примеры:
        >>> # Базовый вызов
        >>> plot_phik_correlation(df)
        
        >>> # С фильтрацией и высоким разрешением
        >>> plot_phik_correlation(
        ...     df,
        ...     exclude=['id'],
        ...     threshold=0.25,
        ...     dpi=300,
        ...     show_report=True
        ... )

    Замечания:
        - Требуется установка библиотеки `phik`: `pip install phik`
        - Phik ∈ [0, 1]: 0 - независимость, 1 - полная зависимость
        - Интерпретация силы связи:
            • < 0.1 - очень слабая
            • 0.1–0.3 - слабая
            • 0.3–0.5 - умеренная
            • ≥ 0.5 - сильная
    """
    try:
        import phik
    except ImportError:
        raise ImportError("❗ Библиотека 'phik' не установлена. Выполните: !pip install phik -q")

    # 1. Подготовка списка колонок
    all_cols = df.columns.tolist()
    if exclude:
        all_cols = [col for col in all_cols if col not in exclude]
    if include:
        all_cols = [col for col in all_cols if col in include]
    if not all_cols:
        print("⚠️ Нет колонок для анализа после фильтрации.")
        return pd.DataFrame()

    df_subset = df[all_cols].copy()

    # 2. Автоматическое определение числовых колонок
    if interval_cols is None:
        interval_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🔍 Авто-определение числовых признаков: {len(interval_cols)} шт.")
    else:
        missing = set(interval_cols) - set(df_subset.columns)
        if missing:
            raise ValueError(f"❌ Колонки не найдены в датафрейме: {missing}")

    # 3. Расчёт Phik-матрицы
    print(f"🧮 Расчёт Phik-корреляций (поддержка смешанных типов).\n")
    dataset_profile(df, report='summary')
    phik_matrix_full = df_subset.phik_matrix(interval_cols=interval_cols)

    # 4. Определяем, какие признаки показывать
    if hide_empty_labels:
        # Находим признаки, у которых есть хотя бы одна связь >= threshold
        mask_by_threshold = phik_matrix_full >= threshold
        # Исключаем диагональ (Phik=1)
        np.fill_diagonal(mask_by_threshold.values, False)
        has_significant = mask_by_threshold.any(axis=1)
        cols_to_show = phik_matrix_full.columns[has_significant]
        if len(cols_to_show) == 0:
            print(f"🔸 Нет пар с Phik ≥ {threshold}")
            if show_report:
                print(f"📋 Таблица ассоциаций (Phik ≥ {report_threshold or threshold}):")
                print("Нет значимых пар.")
            return phik_matrix_full
    else:
        cols_to_show = phik_matrix_full.columns

    # 5. Создаём подматрицу только для отображаемых признаков
    phik_vis = phik_matrix_full.loc[cols_to_show, cols_to_show]

    # 6. Маска для треугольника (только если нужно)
    if show_triangle == 'lower':
        triangle_mask = np.triu(np.ones_like(phik_vis, dtype=bool), k=1)
    elif show_triangle == 'upper':
        triangle_mask = np.tril(np.ones_like(phik_vis, dtype=bool), k=-1)
    else:
        triangle_mask = np.zeros_like(phik_vis, dtype=bool)

    # 🔥 Скрываем диагональ
    np.fill_diagonal(triangle_mask, True)

    # 7. Адаптивный размер фигуры
    n_vis = len(phik_vis)
    if figsize is None:
        base_size_per_feature = 1.2
        min_figsize = 5.0
        max_figsize = 16.0
        fig_size = max(min_figsize, min(max_figsize, n_vis * base_size_per_feature))
        figsize = (fig_size, fig_size)

    # 8. Визуализация
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    base_fontsize = max(10, 14 - n_vis // 2)

    sns.heatmap(
        phik_vis,
        mask=triangle_mask,
        annot=annot,
        cmap=cmap,
        square=True,
        fmt=f'.{precision}f',
        cbar_kws={'shrink': 0.8, 'label': 'Phi-K'},
        linewidths=0.5,
        linecolor=grid_color,
        annot_kws={'size': max(8, base_fontsize - 2)},
        ax=ax
    )

    # Внешняя рамка
    if show_border:
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(0.8)

    # Заголовок
    dataset_name, dataset_desc = label_for_dataset(df, separator='•')
    ax.set_title(
        f'Phik-матрица ассоциаций\n'
        f' • датасет: {dataset_name}{dataset_desc}\n'
        f' • порог: {threshold}\n'
        f' • признаков: {n_vis}',
        fontsize=max(9, base_fontsize - 2),
        loc='left',
        pad=12
    )

    # Подписи осей
    ax.set_xticklabels(cols_to_show, rotation=45, ha='right', fontsize=max(8, base_fontsize - 2))
    ax.set_yticklabels(cols_to_show, rotation=0, fontsize=max(8, base_fontsize - 2))

    plt.tight_layout()
    plt.show()

    # 9. Отчёт
    if not show_report:
        return phik_matrix_full

    if report_threshold is None:
        report_threshold = threshold

    pairs_data = []
    cols = phik_matrix_full.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            phik_val = phik_matrix_full.iloc[i, j]
            if phik_val >= report_threshold:
                col1, col1_desc = label_for_column(cols[i], separator='•')
                col2, col2_desc = label_for_column(cols[j], separator='•')
                pairs_data.append({
                    'Признак 1': f'{col1}{col1_desc}' if col1_desc else col1,
                    'Phik': phik_val,
                    'Признак 2': f'{col2}{col2_desc}' if col2_desc else col2
                })

    if not pairs_data:
        print(f"🔸 Нет пар с Phik ≥ {report_threshold}")
        return phik_matrix_full

    pairs_df = pd.DataFrame(pairs_data).sort_values('Phik', ascending=False)

    def _phik_strength(val: float) -> str:
        if val < 0.1:
            return "очень слабая"
        elif val < 0.3:
            return "слабая"
        elif val < 0.5:
            return "умеренная"
        else:
            return "сильная"

    pairs_df['Сила связи'] = pairs_df['Phik'].apply(_phik_strength)

    print(f"\n📋 Таблица ассоциаций (Phik ≥ {report_threshold}):")
    display_table(
        pairs_df[['Признак 1', 'Phik', 'Признак 2', 'Сила связи']],
        rows=len(pairs_df),
        float_precision=precision,
        styler_func=lambda s: (
            s.background_gradient(subset=['Phik'], cmap='Blues', low=0.2, high=0.8)
            .applymap(
                lambda x: "background-color: #e8f5e8; color: #69a85d" if x == "очень слабая" else
                          "background-color: #c8e6c9; color: #458239" if x == "слабая" else
                          "background-color: #a5d6a7; color: #2e5e25" if x == "умеренная" else
                          "background-color: #4caf50; color: #f9ff80",
                subset=['Сила связи']
            )
        )
    )

    return # phik_matrix_full


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# plot_train_test_distribution - Сравнивает распределение одного признака между обучающей (train) и тестовой (test) выборками
def plot_train_test_distribution(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    feature_label: str = "",
    palette: str = "tab10",
    table_metrics: Literal['basic', 'extended'] = 'extended'
) -> None:
    """
        Сравнивает распределение одного признака между обучающей (train) и тестовой (test) выборками.
        
        Функция автоматически определяет тип признака и строит соответствующую визуализацию:
        
        - Для **числовых признаков**:  
        использует `plot_feature_distribution` с группировкой по 'dataset' (train/test),  
        строит гистограмму с KDE, boxplot и расширенную таблицу статистик (если table_metrics='extended').
        
        - Для **категориальных признаков**:  
        строит горизонтальный barplot с группировкой по категориям,  
        добавляет таблицу с абсолютными и относительными частотами,  
        а также абсолютной разницей в долях (в процентных пунктах).
        
        Цель: выявить **сдвиг распределения (data drift)** между train и test,  
        который может негативно повлиять на обобщающую способность модели.
        
        Параметры
       -------
        train : pd.DataFrame
            Обучающая выборка.
        test : pd.DataFrame
            Тестовая выборка.
        feature : str
            Название столбца для анализа. Должен присутствовать в обоих датафреймах.
        feature_label : str, optional
            Человекочитаемое название признака для заголовков и осей.  
            Если не указано, используется значение `feature`.
        palette : str, optional
            Название цветовой палитры seaborn (например, 'tab10', 'Set2', 'husl').  
            Используется для раскраски групп 'train' и 'test'. По умолчанию 'Set2'.
        table_metrics : {'basic', 'extended'}, optional
            Уровень детализации таблицы статистик (только для числовых признаков):
            - 'basic': базовые метрики (среднее, медиана, стд и т.д.)
            - 'extended': + асимметрия, эксцесс, IQR/медиана, доля выбросов (%)
            По умолчанию 'extended'.
        
        Возвращает
       -------
        None
            Отображает график и таблицу в ячейку Jupyter Notebook.
        
        Примеры
       -----
        >>> # Числовой признак
        >>> plot_train_test_distribution(train, test, 'pages_per_visit', 'Страниц за визит')
        
        >>> # Категориальный признак
        >>> plot_train_test_distribution(train, test, 'top_category', 'Категория товара', palette='tab10')
        
        Замечания
       ------
        - Для корректной работы требуется, чтобы функция `plot_feature_distribution` 
        была определена в том же окружении (для числовых признаков).
        - Категории, присутствующие только в одном из датасетов, будут отображены с нулевым 
        количеством в другом - это помогает выявить новые/пропавшие значения.
        - Абсолютная разница (`Δ доля (pp)`) указывается в **процентных пунктах (pp)**, 
        а не в относительных процентах.
    """
    # Проверка наличия признака
    if feature not in train.columns or feature not in test.columns:
        raise ValueError(f"Признак '{feature}' отсутствует в train или test")
    
    # Подготавливаем данные
    train_labeled = train[[feature]].copy()
    train_labeled['dataset'] = 'train'
    test_labeled = test[[feature]].copy()
    test_labeled['dataset'] = 'test'
    combined = pd.concat([train_labeled, test_labeled], ignore_index=True)

    if not feature_label:
        feature_label = feature

    # Определяем тип признака
    is_categorical = not pd.api.types.is_numeric_dtype(combined[feature])

    col_name, col_desc = label_for_column(feature_label, separator="•")
    full_col_name = f"'{col_name}'{col_desc}"

    if is_categorical:
        # Категориальный признак
        total_train = len(combined[combined['dataset'] == 'train'])
        total_test = len(combined[combined['dataset'] == 'test'])
        if total_train == 0 or total_test == 0:
            print("⚠️ Один из датасетов пуст")
            return

        # Заменяем пропуски на явную метку ДО группировки
        combined = combined.copy()
        combined[feature] = combined[feature].fillna("[пропуски ")
        # Также заменяем пустые строки
        combined[feature] = combined[feature].replace("", "[пробелы]")

        # Статистика по категориям
        stats = (
            combined.groupby(['dataset', feature])
            .size()
            .reset_index(name='count')
            .pivot(index=feature, columns='dataset', values='count')
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        stats.columns.name = None

        stats['train_pct'] = stats['train'] / total_train * 100
        stats['test_pct'] = stats['test'] / total_test * 100
        stats['Δ доля (pp)'] = (stats['train_pct'] - stats['test_pct']).abs()
        stats = stats.sort_values('Δ доля (pp)', ascending=False)

        # Получаем цвета из палитры
        unique_datasets = ['train', 'test']
        colors = sns.color_palette(palette, n_colors=len(unique_datasets))
        palette_dict = dict(zip(unique_datasets, colors))

        # Подготовка данных для графика
        stats_long = stats.melt(
            id_vars=feature,
            value_vars=unique_datasets,
            var_name='dataset',
            value_name='count'
        )

        # Расчёт Cramér’s V для заголовка
        observed = pd.crosstab(combined['dataset'], combined[feature])
        try:
            chi2, _, _, expected = chi2_contingency(observed)
            n = observed.sum().sum()
            min_dim = min(observed.shape) - 1
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                cramers_v = min(cramers_v, 1.0)
            else:
                cramers_v = 0.0
        except:
            cramers_v = 0.0

        # Адаптивная высота графика
        n_categories = len(stats)
        height = min(12.0, max(3.5, n_categories * 0.4))
        figsize = (16, height)

        # Горизонтальный barplot с constrained_layout
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sns.barplot(
            data=stats_long,
            y=feature,
            x='count',
            hue='dataset',
            palette=palette_dict,
            ax=ax,
            edgecolor="white",
            linewidth=0.8,
            dodge=True
        )
        ax.set_title(
            f"Распределение: {full_col_name} (train vs test)\n"
            f"Cramér’s V = {cramers_v:.3f}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("Количество", fontsize=10)
        ax.set_ylabel(feature_label, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        legend = ax.legend(title="Dataset", loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True)
        plt.setp(legend.get_title(), fontsize=10)
        plt.setp(legend.get_texts(), fontsize=9)
        plt.show()

        print(f"\nСравнение долей по {full_col_name}:")

        # Интерпретация с учётом пропусков
        def _interpret_diff(diff: float, category: str):
            if "[ пропуск ]" in category:
                if diff > 0:
                    return "🚨 Пропуски только в одной выборке", "critical", "Проверить источник данных"
                else:
                    return "🟢 Пропуски согласованы", "low", ""
            elif diff < 1.0:
                return "🟢 Норма", "low", ""
            elif diff < 3.0:
                return "🟠 Внимание", "medium", "Проверить баланс"
            elif diff < 5.0:
                return "🔴 Значительно", "high", "Рассмотреть стратификацию"
            else:
                return "💥 Критично", "critical", "Объединить категории"

        stats[['Статус', 'Уровень', 'Рекомендация']] = stats.apply(
            lambda row: pd.Series(_interpret_diff(row['Δ доля (pp)'], row[feature])), axis=1
        )

        # Сортировка по риску
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        stats['risk_sort'] = stats['Уровень'].map(risk_order)
        stats = stats.sort_values(['risk_sort', 'Δ доля (pp)'], ascending=[True, False]).drop(columns='risk_sort')

        # Итоговая таблица
        display_table(
            stats[[
                feature, 'train', 'test', 'train_pct', 'test_pct', 'Δ доля (pp)',
                'Статус', 'Рекомендация'
            ]],
            rows=len(stats),
            float_precision=2,
            styler_func=lambda s: (
                s.format({
                    'train_pct': '{:.1f}%',
                    'test_pct': '{:.1f}%',
                    'Δ доля (pp)': '{:.1f} pp'
                }, na_rep="-")                
                .background_gradient(subset=['train'], cmap='vlag')
                .background_gradient(subset=['test'], cmap='vlag')
                .background_gradient(subset=['Δ доля (pp)'], cmap='Reds')
            )
        )

    else:
        # Числовой признак
        try:
            plot_feature_distribution(
                df=combined,
                feature=feature,
                hue='dataset',
                stat='density',
                table_metrics=table_metrics,
                palette=palette
            )
        except NameError:
            print("⚠️ Функция plot_feature_distribution не найдена. Убедитесь, что она определена.")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def plot_discrete_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    figsize: Tuple[int, int] = (12, 3),
    palette: Tuple[str, str] = ('#295C96', '#ffa230'),
    table_metrics: Optional[Literal['basic', 'extended']] = None  # Изменено: теперь по умолчанию None
) -> None:
    """
    Сравнение распределения ДИСКРЕТНОГО признака между train и test через countplot и (опционально) таблицу.

    Построение двух графиков `sns.countplot` для сравнения частот уникальных значений
    дискретного признака в обучающей и тестовой выборках. Также может выводить
    сравнительную таблицу с абсолютными и относительными частотами, разницей в долях
    и статусом расхождения. Использует глобальные справочники DATASET_DESCRIPTIONS
    и COLUMN_DESCRIPTIONS для автоматического формирования подписей.

    Параметры:
    ----------
    train : pd.DataFrame
        Обучающая выборка.
    test : pd.DataFrame
        Тестовая выборка.
    feature : str
        Название дискретного признака (например, 'employment_years').
    figsize : tuple, optional
        Размер фигуры (по умолчанию (12, 3)).
    palette : tuple, optional
        Цвета для train и test (по умолчанию ('#295C96', '#ffa230')).
    table_metrics : {'basic', 'extended', None}, optional
        Уровень детализации таблицы:
        - 'basic': частоты, доли, отклонение (pp), статус.
        - 'extended': как 'basic' + Cramér’s V.
        - None (по умолчанию): таблица не выводится.
    """
    # Подписи
    train_name, train_desc = label_for_dataset(train, separator="•")
    test_name, test_desc = label_for_dataset(test, separator="•")
    feature_name, feature_desc = label_for_column(feature, separator="•")
    full_feature_label = f"{feature_name}{feature_desc}" if feature_desc else feature_name

    # График
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Сравнение распределения дискретного признака: {full_feature_label}",
        fontsize=12,
        fontweight=200,
        y=1.02
    )

    sns.countplot(data=train, x=feature, ax=axes[0], color=palette[0], alpha=0.8)
    axes[0].set_title(f"{train_name}", fontsize=9)
    axes[0].set_ylabel('Количество', fontsize=9)
    axes[0].set_xlabel(full_feature_label, fontsize=9)
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    sns.countplot(data=test, x=feature, ax=axes[1], color=palette[1], alpha=0.8)
    axes[1].set_title(f"{test_name}", fontsize=9)
    axes[1].set_ylabel('Количество', fontsize=9)
    axes[1].set_xlabel(full_feature_label, fontsize=9)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.02, f"Признак: {full_feature_label}", ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    plt.show()

    # Таблица (опционально)
    if table_metrics is None:
        return

    print(f"\nСравнение распределения долей дискретного признака по '{feature_name}'{feature_desc}:")

    # Подготовка данных
    train_labeled = train[[feature]].copy()
    train_labeled['dataset'] = 'train'
    test_labeled = test[[feature]].copy()
    test_labeled['dataset'] = 'test'
    combined = pd.concat([train_labeled, test_labeled], ignore_index=True)

    counts = pd.crosstab(combined[feature], combined['dataset'], dropna=False)
    total_train = counts['train'].sum()
    total_test = counts['test'].sum()

    result = pd.DataFrame({
        'train': counts['train'],
        'test': counts['test'],
        'train_pct': (counts['train'] / total_train * 100).round(1),
        'test_pct': (counts['test'] / total_test * 100).round(1)
    }).reset_index()
    result['Δ доля (pp)'] = (result['train_pct'] - result['test_pct']).abs().round(1)

    def _get_status(diff: float) -> Tuple[str, str]:
        if diff < 1.0:
            return "🟢 Норма", ""
        elif diff < 3.0:
            return "🟠 Внимание", "Проверить баланс"
        else:
            return "🔴 Значительно", "Рассмотреть стратификацию"

    result[['Статус', 'Рекомендация']] = result['Δ доля (pp)'].apply(
        lambda x: pd.Series(_get_status(x))
    )
    result = result.sort_values('Δ доля (pp)', ascending=False)

    display_table(
        result,
        rows=len(result),
        float_precision=1,
        styler_func=lambda s: s.format({
            'train_pct': '{:.1f}%',
            'test_pct': '{:.1f}%',
            'Δ доля (pp)': '{:.1f} pp'
        }).background_gradient(subset=['Δ доля (pp)'], cmap='Reds')
    )

    # Cramér’s V (только в 'extended')
    if table_metrics == 'extended':
        observed = counts.copy()
        try:
            chi2, p_val, dof, expected = chi2_contingency(observed)
            n = observed.sum().sum()
            min_dim = min(observed.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
            cramers_v = min(cramers_v, 1.0)
            print(f"\n📊 Cramér’s V = {cramers_v:.3f} {'(слабая связь)' if cramers_v < 0.1 else '(умеренная связь)' if cramers_v < 0.3 else '(сильная связь)'}")
        except Exception as e:
            print(f"\n⚠️ Не удалось рассчитать Cramér’s V: {e}")
   


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••



# plot_compare_train_test_ecdf - Сравнивает распределения признака между train и test
def plot_compare_train_test_ecdf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    feature_label: Optional[str] = None,
    palette: str = "tab10",
    show_stats: bool = True,
    figsize: tuple = (18, 5)
) -> None:
    """
    Сравнивает распределения признака между train и test с помощью трёх графиков.
    Автоматически определяет тип признака (числовой / категориальный).
    
    Описание:
        Для числовых признаков:
        - Гистограмма + KDE
        - Boxplot
        - ECDF + KS-тест
        
        Для категориальных признаков:
        - Barplot долей
        - Таблица сопряжённости
        - Cramér's V + Chi² тест
        
        Использует глобальные справочники COLUMN_DESCRIPTIONS и DATASET_DESCRIPTIONS.

    Параметры:
        train/test: pd.DataFrame - выборки
        feature: str - имя признака
        feature_label: Optional[str] - человекочитаемое название
        palette: str - палитра seaborn
        show_stats: bool - показывать статистики
        figsize: tuple - размер фигуры

    Возвращаемое значение:
        None - отображает график и статистики
    """
    if feature not in train.columns or feature not in test.columns:
        raise ValueError(f"Признак '{feature}' отсутствует в train или test")

    # Подпись признака
    col_name, col_desc = label_for_column(feature, separator='•')
    if not feature_label:
        feature_label = f"{col_name}{col_desc}" if col_desc else col_name

    # Цвета - всегда доступны
    colors = sns.color_palette(palette, n_colors=2)
    color_train, color_test = colors[0], colors[1]

    # Определяем тип признака
    is_numeric = pd.api.types.is_numeric_dtype(train[feature]) and pd.api.types.is_numeric_dtype(test[feature])
    n_unique_train = train[feature].nunique()
    n_unique_test = test[feature].nunique()

    # Если числовой и достаточно уникальных значений
    if is_numeric and n_unique_train > 20 and n_unique_test > 20:
        # ЧИСЛОВОЙ РЕЖИМ
        colors = sns.color_palette(palette, n_colors=2)
        color_train, color_test = colors[0], colors[1]

        train_data = train[feature].dropna()
        test_data = test[feature].dropna()

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"⚠️ Недостаточно данных для анализа {feature_label}")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Гистограмма + KDE
        axes[0].hist(train_data, bins=20, alpha=0.7, color=color_train, label='Train', density=True)
        axes[0].hist(test_data, bins=20, alpha=0.7, color=color_test, label='Test', density=True)
        sns.kdeplot(data=train_data, color=color_train, ax=axes[0], linewidth=2)
        sns.kdeplot(data=test_data, color=color_test, ax=axes[0], linewidth=2)
        axes[0].set_title('Распределение: Train vs Test')
        axes[0].set_xlabel(feature_label)
        axes[0].set_ylabel('Плотность')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Boxplot
        data_to_plot = [train_data, test_data]
        bp = axes[1].boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True,
                            boxprops=dict(facecolor=color_train), medianprops=dict(color='white'))
        bp['boxes'][1].set_facecolor(color_test)
        axes[1].set_title('Boxplot: Train vs Test')
        axes[1].set_ylabel(feature_label)
        axes[1].grid(True, alpha=0.3)

        # 3. ECDF
        try:
            from statsmodels.distributions.empirical_distribution import ECDF
            ecdf_train = ECDF(train_data)
            ecdf_test = ECDF(test_data)
            axes[2].plot(ecdf_train.x, ecdf_train.y, color=color_train, label='Train', linewidth=2)
            axes[2].plot(ecdf_test.x, ecdf_test.y, color=color_test, label='Test', linewidth=2)
            axes[2].set_title('ECDF: Train vs Test')
            axes[2].set_xlabel(feature_label)
            axes[2].set_ylabel('Накопленная вероятность')
            axes[2].legend()
            axes[2].grid(True, linestyle='--', alpha=0.6)
        except ImportError:
            axes[2].text(0.5, 0.5, 'statsmodels не установлена', ha='center', va='center',
                        transform=axes[2].transAxes)
            axes[2].set_title('ECDF: ошибка')

        plt.tight_layout()
        plt.show()

        if show_stats:
            ks_stat, p_value = stats.ks_2samp(train_data, test_data)
            print(f"\nСтатистики для '{feature_label}':")
            print(f"   Kolmogorov-Smirnov test:")
            print(f"     • Статистика: {ks_stat:.4f}")
            print(f"     • p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("   🔺 Распределения статистически различаются (H₁)")
            else:
                print("   ✔️ Распределения не отличаются (H₀)")

    else:
        # Категориальный режим
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Подготовка данных
        combined = pd.concat([
            train[[feature]].assign(dataset='train'),
            test[[feature]].assign(dataset='test')
        ], ignore_index=True).dropna()

        # 1. Barplot долей
        counts = pd.crosstab(combined[feature], combined['dataset'])
        counts.plot(kind='bar', ax=axes[0], color=[colors[0], colors[1]], alpha=0.8, edgecolor='white')
        axes[0].set_title('Частоты: Train vs Test')
        axes[0].set_xlabel(feature_label)
        axes[0].set_ylabel('Количество')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, axis='y', alpha=0.3)

        # 2. Boxplot - заменяем на таблицу сопряжённости
        table = pd.crosstab(combined['dataset'], combined[feature])
        ax2_text = "Таблица сопряжённости:\n\n" + table.to_string()
        axes[1].text(0.5, 0.5, ax2_text, ha='center', va='center', transform=axes[1].transAxes,
                    fontfamily='monospace', fontsize=9)
        axes[1].set_title('Таблица сопряжённости')
        axes[1].axis('off')

        # 3. ECDF - заменяем на Cramér's V
        observed = pd.crosstab(combined['dataset'], combined[feature])
        chi2, p_val, dof, expected = chi2_contingency(observed)
        n = observed.sum().sum()
        min_dim = min(observed.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

        stat_text = f"Cramer's V = {cramers_v:.4f}\nχ² p-value = {p_val:.4f}"
        axes[2].text(0.5, 0.5, stat_text, ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat"))
        axes[2].set_title('Статистическая связь')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        if show_stats:
            print(f"\n📊 Статистики для '{feature_label}':")
            print(f"   Cramer's V: {cramers_v:.4f}")
            print(f"   χ² p-value: {p_val:.4f}")
            if p_val < 0.05 and cramers_v > 0.1:
                print("   🔺 Значимая разница в распределении категорий (H₁)")
            else:
                print("   ✔️ Распределения категорий согласованы (H₀)")




#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def plot_discrete_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    figsize: Tuple[int, int] = (12, 3),
    palette: Tuple[str, str] = ('#295C96', '#ffa230'),
    table_metrics: Optional[Literal['basic', 'extended']] = None  # Изменено: теперь по умолчанию None
) -> None:
    """
    Сравнение распределения ДИСКРЕТНОГО признака между train и test через countplot и (опционально) таблицу.

    Построение двух графиков `sns.countplot` для сравнения частот уникальных значений
    дискретного признака в обучающей и тестовой выборках. Также может выводить
    сравнительную таблицу с абсолютными и относительными частотами, разницей в долях
    и статусом расхождения. Использует глобальные справочники DATASET_DESCRIPTIONS
    и COLUMN_DESCRIPTIONS для автоматического формирования подписей.

    Параметры:
    ----------
    train : pd.DataFrame
        Обучающая выборка.
    test : pd.DataFrame
        Тестовая выборка.
    feature : str
        Название дискретного признака (например, 'employment_years').
    figsize : tuple, optional
        Размер фигуры (по умолчанию (12, 3)).
    palette : tuple, optional
        Цвета для train и test (по умолчанию ('#295C96', '#ffa230')).
    table_metrics : {'basic', 'extended', None}, optional
        Уровень детализации таблицы:
        - 'basic': частоты, доли, отклонение (pp), статус.
        - 'extended': как 'basic' + Cramér’s V.
        - None (по умолчанию): таблица не выводится.
    """
    # Подписи
    train_name, train_desc = label_for_dataset(train, separator="•")
    test_name, test_desc = label_for_dataset(test, separator="•")
    feature_name, feature_desc = label_for_column(feature, separator="•")
    full_feature_label = f"{feature_name}{feature_desc}" if feature_desc else feature_name

    # График
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Сравнение распределения дискретного признака: {full_feature_label}",
        fontsize=12,
        fontweight=200,
        y=1.02
    )

    sns.countplot(data=train, x=feature, ax=axes[0], color=palette[0], alpha=0.8)
    axes[0].set_title(f"{train_name}", fontsize=9)
    axes[0].set_ylabel('Количество', fontsize=9)
    axes[0].set_xlabel(full_feature_label, fontsize=9)
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    sns.countplot(data=test, x=feature, ax=axes[1], color=palette[1], alpha=0.8)
    axes[1].set_title(f"{test_name}", fontsize=9)
    axes[1].set_ylabel('Количество', fontsize=9)
    axes[1].set_xlabel(full_feature_label, fontsize=9)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.02, f"Признак: {full_feature_label}", ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    plt.show()

    # Таблица (опционально)
    if table_metrics is None:
        return

    print(f"\nСравнение распределения долей дискретного признака по '{feature_name}'{feature_desc}:")

    # Подготовка данных
    train_labeled = train[[feature]].copy()
    train_labeled['dataset'] = 'train'
    test_labeled = test[[feature]].copy()
    test_labeled['dataset'] = 'test'
    combined = pd.concat([train_labeled, test_labeled], ignore_index=True)

    counts = pd.crosstab(combined[feature], combined['dataset'], dropna=False)
    total_train = counts['train'].sum()
    total_test = counts['test'].sum()

    result = pd.DataFrame({
        'train': counts['train'],
        'test': counts['test'],
        'train_pct': (counts['train'] / total_train * 100).round(1),
        'test_pct': (counts['test'] / total_test * 100).round(1)
    }).reset_index()
    result['Δ доля (pp)'] = (result['train_pct'] - result['test_pct']).abs().round(1)

    def _get_status(diff: float) -> Tuple[str, str]:
        if diff < 1.0:
            return "🟢 Норма", ""
        elif diff < 3.0:
            return "🟠 Внимание", "Проверить баланс"
        else:
            return "🔴 Значительно", "Рассмотреть стратификацию"

    result[['Статус', 'Рекомендация']] = result['Δ доля (pp)'].apply(
        lambda x: pd.Series(_get_status(x))
    )
    result = result.sort_values('Δ доля (pp)', ascending=False)

    display_table(
        result,
        rows=len(result),
        float_precision=1,
        styler_func=lambda s: s.format({
            'train_pct': '{:.1f}%',
            'test_pct': '{:.1f}%',
            'Δ доля (pp)': '{:.1f} pp'
        }).background_gradient(subset=['Δ доля (pp)'], cmap='Reds')
    )

    # Cramér’s V (только в 'extended')
    if table_metrics == 'extended':
        observed = counts.copy()
        try:
            chi2, p_val, dof, expected = chi2_contingency(observed)
            n = observed.sum().sum()
            min_dim = min(observed.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
            cramers_v = min(cramers_v, 1.0)
            print(f"\n📊 Cramér’s V = {cramers_v:.3f} {'(слабая связь)' if cramers_v < 0.1 else '(умеренная связь)' if cramers_v < 0.3 else '(сильная связь)'}")
        except Exception as e:
            print(f"\n⚠️ Не удалось рассчитать Cramér’s V: {e}")
            



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••



def plot_shap_summary(
    model_pipeline,
    X_test,
    feature_names_mapping=None,
    max_display=10,
    figsize=(12, 8),
    title_fontsize=12,
    axis_fontsize=8,
    label_mode='combined',
    preprocessor_step_name='preprocessor'
):
    """
    Строит SHAP Summary Plot для модели, обученной в сложном пайплайне.
    
    Параметры:
    - model_pipeline: обученный Pipeline
    - X_test: тестовые данные (DataFrame) до применения пайплайна
    - feature_names_mapping: dict для кастомных названий (опционально)
    - max_display: максимальное число признаков на графике
    - figsize: размер фигуры
    - title_fontsize: размер шрифта заголовка
    - axis_fontsize: размер шрифта меток осей
    - label_mode: способ отображения названий признаков
        'name' - только имя колонки (dept)
        'description' - только описание (отдел)
        'combined' - имя и описание (dept • отдел) [по умолчанию]
    - preprocessor_step_name: имя шага в model_pipeline, содержащего препроцессор (по умолчанию 'preprocessor')
    """
    # Извлекаем модель и препроцессор
    if 'model' not in model_pipeline.named_steps:
        raise ValueError("Pipeline должен содержать шаг с именем 'model'")
    model = model_pipeline.named_steps['model']

    if preprocessor_step_name not in model_pipeline.named_steps:
        raise ValueError(f"Pipeline должен содержать шаг '{preprocessor_step_name}'")
    preprocessor = model_pipeline.named_steps[preprocessor_step_name]

    # Преобразуем данные
    X_test_processed = preprocessor.transform(X_test)

    # Получаем имена признаков
    try:
        # Современный способ (scikit-learn >= 1.0)
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            raise AttributeError()
    except (AttributeError, NotImplementedError):
        # Fallback: соберём вручную (для совместимости)
        feature_names = []
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if trans == 'drop' or trans is None:
                    continue
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        feature_names.extend(trans.get_feature_names_out(cols))
                    except:
                        feature_names.extend([f"{name}_{i}" for i in range(len(cols))])
                else:
                    feature_names.extend(cols)
        else:
            feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

    # Генерация отображаемых имён (ваша логика)
    display_names = []
    for name in feature_names:
        if feature_names_mapping and name in feature_names_mapping:
            display_name = feature_names_mapping[name]
        else:
            try:
                col_name, col_desc = label_for_column(name, separator="•")
                if label_mode == 'name':
                    display_name = col_name
                elif label_mode == 'description':
                    display_name = col_desc.strip()
                elif label_mode == 'combined':
                    display_name = f"{col_name}{col_desc}"
                else:
                    display_name = name
            except:
                display_name = name
        display_names.append(display_name)

    # SHAP и визуализация
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_test_processed,
        feature_names=display_names,
        max_display=max_display,
        show=False
    )

    ax = plt.gca()
    ax.set_title("SHAP Summary Plot: Вклад признаков", fontsize=title_fontsize, pad=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(axis_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()








#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••





def plot_numerical_profile(
    df: pd.DataFrame,
    feature: str,
    compare_df: Optional[pd.DataFrame] = None,
    compare_label: str = "test",
    palette: str = "tab10",
    figsize: Tuple[int, int] = (16, 4),
    show_report: bool = True,
    show_recommendations: bool = True
) -> None:
    """
    Строит адаптивный профиль числового признака с диагностикой распределения и рекомендациями.
    
    Описание:
        Автоматически определяет тип признака:
        - бинарный (2 значения) → barplot + boxplot + ECDF
        - дискретный (3–20 целых) → barplot + boxplot + ECDF
        - непрерывный → hist+KDE + boxplot + Q-Q + ECDF
        
        Поддерживает сравнение с другой выборкой (например, test).
        Выводит статистику и практические рекомендации.
    
    Параметры:
        df: pd.DataFrame - основная выборка (например, train)
        feature: str - имя числового признака
        compare_df: Optional[pd.DataFrame] - выборка для сравнения (например, test)
        compare_label: str - метка для сравнения (по умолчанию "test")
        palette: str - палитра seaborn (по умолчанию "tab10")
        figsize: Tuple[int, int] - размер фигуры
        show_report: bool - показывать сводную таблицу статистик
        show_recommendations: bool - показывать рекомендации

    Возвращает:
        None - вывод через matplotlib и display_table

    Примеры:
        1. plot_numerical_profile(train_df, 'stress_level', compare_df=test_df, compare_label="test")
        2. plot_numerical_profile(train_df, 'stress_level')

    Зависимости:
        from typing import Optional, Tuple
    """
    from scipy import stats as scipy_stats
    from statsmodels.distributions.empirical_distribution import ECDF

    # Проверки
    if feature not in df.columns:
        raise ValueError(f"Признак '{feature}' не найден в датафрейме")
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError(f"Признак '{feature}' не является числовым")

    # Данные
    data = df[feature].dropna()
    n_total = len(df[feature])
    n_valid = len(data)
    n_missing = n_total - n_valid

    if n_valid == 0:
        print(f"⚠️ Признак '{feature}' содержит только пропуски")
        return

    # Подписи
    col_name, col_desc = label_for_column(feature, separator='•')
    full_label = f"{col_name}{col_desc}" if col_desc else col_name
    dataset_name, dataset_desc = label_for_dataset(df, separator='•')

    print(f"🔍 Профиль признака: {full_label}")
    print(f"🗃️ Датасет: {dataset_name}{dataset_desc}")
    print(f"     • Валидных значений: {n_valid:,} (пропусков: {n_missing})")

    # Статистики
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    skew_val = scipy_stats.skew(data)
    kurt_val = scipy_stats.kurtosis(data)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)

    # Тип признака
    n_unique = data.nunique()
    is_integer = pd.api.types.is_integer_dtype(data)
    if n_unique <= 1:
        feature_type = "🔇 почти константный"
    elif n_unique == 2:
        feature_type = "💊 бинарный"
    elif n_unique <= 20 and is_integer:
        feature_type = "🔢 дискретный"
    else:
        feature_type = "🔢 непрерывный"

    print(f"     • Тип: {feature_type} ({n_unique} уникальных значений)")

    # Цвета
    colors = sns.color_palette(palette, n_colors=2)
    color_train = colors[0]
    color_test = colors[1]

    # ВИЗУАЛИЗАЦИЯ
    if feature_type in ["💊 бинарный", "🔢 дискретный"]:
        # Для дискретных: 3 графика
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Barplot
        value_counts_train = data.value_counts().sort_index()
        bars_train = axes[0].bar(
            value_counts_train.index, value_counts_train.values,
            color=color_train, alpha=0.8, label="train", width=0.4
        )
        if compare_df is not None and feature in compare_df.columns:
            test_data = compare_df[feature].dropna()
            value_counts_test = test_data.value_counts().sort_index()
            # Выравниваем индексы
            all_keys = sorted(set(value_counts_train.index) | set(value_counts_test.index))
            train_vals = [value_counts_train.get(k, 0) for k in all_keys]
            test_vals = [value_counts_test.get(k, 0) for k in all_keys]
            bars_test = axes[0].bar(
                [k + 0.4 for k in all_keys], test_vals,
                color=color_test, alpha=0.8, label=compare_label, width=0.4, hatch='//'
            )
        axes[0].set_title("Распределение значений", fontsize=11, fontweight='bold')
        axes[0].set_xlabel(full_label)
        axes[0].set_ylabel("Частота")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.4)

        # 2. Boxplot
        box_data = [data]
        box_labels = ["train"]
        box_colors = [color_train]
        if compare_df is not None and feature in compare_df.columns:
            box_data.append(compare_df[feature].dropna())
            box_labels.append(compare_label)
            box_colors.append(color_test)
        bplot = axes[1].boxplot(
            box_data, labels=box_labels, patch_artist=True,
            medianprops=dict(color='white', linewidth=1.5)
        )
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Boxplot", fontsize=11, fontweight='bold')
        axes[1].set_ylabel(full_label)
        axes[1].grid(True, linestyle='--', alpha=0.4)

        # 3. ECDF
        ecdf_train = ECDF(data)
        axes[2].step(ecdf_train.x, ecdf_train.y, where='post', color=color_train, linewidth=2.5, label="train")
        if compare_df is not None and feature in compare_df.columns:
            ecdf_test = ECDF(compare_df[feature].dropna())
            axes[2].step(ecdf_test.x, ecdf_test.y, where='post', color=color_test, linewidth=2.5, linestyle='--', label=compare_label)
        axes[2].set_title("ECDF", fontsize=11, fontweight='bold')
        axes[2].set_xlabel(full_label)
        axes[2].set_ylabel("Накопленная вероятность")
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.4)

    else:  # Непрерывный
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # 1. Hist + KDE
        bins = min(50, max(10, int(np.sqrt(n_valid))))
        axes[0].hist(data, bins=bins, density=True, alpha=0.6, color=color_train, edgecolor='white', linewidth=0.5, label="train")
        if n_unique > 10:
            sns.kdeplot(data, ax=axes[0], color=color_train, linewidth=2.5)
        if compare_df is not None and feature in compare_df.columns:
            comp_data = compare_df[feature].dropna()
            axes[0].hist(comp_data, bins=bins, density=True, alpha=0.4, color=color_test, edgecolor='white', linewidth=0.5, label=compare_label)
            if len(comp_data) > 10:
                sns.kdeplot(comp_data, ax=axes[0], color=color_test, linewidth=2.5, linestyle="--")
        axes[0].set_title("Гистограмма + KDE", fontsize=11, fontweight='bold')
        axes[0].set_xlabel(full_label)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.4)

        # 2. Boxplot
        box_data = [data]
        box_labels = ["train"]
        box_colors = [color_train]
        if compare_df is not None and feature in compare_df.columns:
            box_data.append(compare_df[feature].dropna())
            box_labels.append(compare_label)
            box_colors.append(color_test)
        bplot = axes[1].boxplot(
            box_data, labels=box_labels, patch_artist=True,
            medianprops=dict(color='white', linewidth=1.5)
        )
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Boxplot", fontsize=11, fontweight='bold')
        axes[1].set_ylabel(full_label)
        axes[1].grid(True, linestyle='--', alpha=0.4)

        # 3. Q-Q plot
        scipy_stats.probplot(data, dist="norm", plot=axes[2])
        axes[2].get_lines()[0].set_markerfacecolor(color_train)
        axes[2].get_lines()[0].set_markersize(4)
        axes[2].get_lines()[1].set_color("red")
        axes[2].set_title("Q-Q plot (нормальность)", fontsize=11, fontweight='bold')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        # 4. ECDF
        ecdf_train = ECDF(data)
        axes[3].step(ecdf_train.x, ecdf_train.y, where='post', color=color_train, linewidth=2.5, label="train")
        if compare_df is not None and feature in compare_df.columns:
            ecdf_test = ECDF(compare_df[feature].dropna())
            axes[3].step(ecdf_test.x, ecdf_test.y, where='post', color=color_test, linewidth=2.5, linestyle='--', label=compare_label)
        axes[3].set_title("ECDF", fontsize=11, fontweight='bold')
        axes[3].set_xlabel(full_label)
        axes[3].set_ylabel("Накопленная вероятность")
        axes[3].legend()
        axes[3].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

    # СВОДКА И РЕКОМЕНДАЦИИ
    if show_report:
        stats_df = pd.DataFrame({
            "Метрика": ["Среднее", "Медиана", "Стд", "Мин", "Макс", "Q1", "Q3", "IQR", "Асимметрия", "Эксцесс", "Выбросы"],
            "Значение": [mean_val, median_val, std_val, min_val, max_val, q1, q3, iqr, skew_val, kurt_val, n_outliers]
        })
        print(f"\n📋 Сводная статистика ({n_valid} значений):")
        display_table(
            stats_df,
            rows=len(stats_df),
            float_precision=3,
            styler_func=lambda s: s.format({"Значение": "{:.3f}"})
        )

    if show_recommendations:
        print(f"\n🔍 Рекомендации:")
        if feature_type == "💊 бинарный":
            print("   ✔️ Признак бинарный - используйте как есть или закодируйте в 0/1")
        elif feature_type == "🔢 дискретный":
            print("   ℹ️ Признак дискретный - можно использовать как числовой или преобразовать в категориальный")
        else:
            # Непрерывный
            if abs(skew_val) > 1.0:
                print("   📢 Сильная асимметрия - рассмотрите log/sqrt/Box-Cox трансформацию для линейных моделей")
            elif abs(skew_val) > 0.5:
                print("   💡 Умеренная асимметрия - можно попробовать трансформацию")
            else:
                print("   ✔️ Распределение близко к симметричному")

        # Выбросы
        if n_outliers > 0:
            pct_out = n_outliers / n_valid * 100
            if pct_out > 5:
                print(f"   ⚠️ Много выбросов ({pct_out:.1f}%) - проверьте их природу")
            else:
                print(f"   ✔️ Выбросы в пределах нормы ({pct_out:.1f}%)")

        # Масштабирование
        if min_val >= 0 and max_val <= 1:
            print("   ✔️ Данные уже в [0, 1] - масштабирование не требуется")
        elif std_val > 100:
            print("   ⚠️ Широкий диапазон - рассмотрите стандартизацию")
        else:
            print("   💡 Для градиентного бустинга масштабирование не критично")

        # Сравнение
        if compare_df is not None and feature in compare_df.columns:
            from scipy.stats import ks_2samp
            comp_data = compare_df[feature].dropna()
            if len(comp_data) > 0:
                ks_stat, p_val = ks_2samp(data, comp_data)
                print(f"\n📊 Сравнение с {compare_label}:")
                print(f"   • Kolmogorov-Smirnov: статистика={ks_stat:.4f}, p={p_val:.4f}")
                if p_val < 0.05:
                    print("   🔺 Распределения статистически различаются (дрифт!)")
                else:
                    print("   ✔️ Распределения согласованы")



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
def plot_missing_summary(
    df: pd.DataFrame,
    threshold: float = 0.0,
    top_n: Optional[int] = None,
    figsize: Tuple[float, float] = (14, None),
    palette: str = "Reds_r",
    show_values: bool = True,
    value_threshold: float = 1.5,
    grid_color: str = "#e0e0e0",
    spine_color: str = "#ddd",
    title: Optional[str] = None,
    xlabel: str = "Процент пропусков (%)",
    ylabel: str = "",
    xticks_step: int = 10,
    dpi: Optional[int] = None,
    use_descriptions: bool = True
) -> None:
    """
    Визуализирует процент пропусков по колонкам в виде горизонтальной столбчатой диаграммы.
    
    Описание:
        Функция строит эстетичную и информативную barplot-диаграмму,
        показывающую долю пропусков в каждой колонке датафрейма.
        Поддерживает фильтрацию, сортировку, кастомизацию цветов и сетки.
        Использует COLUMN_DESCRIPTIONS для подписей колонок (если доступно).
    
    Параметры:
        df : pd.DataFrame
            Исходный датафрейм.
        threshold : float, по умолчанию 0.0
            Минимальный процент пропусков для отображения колонки.
        top_n : Optional[int], по умолчанию None
            Сколько топ-колонок по пропускам отобразить.
        figsize : Tuple[float, float], по умолчанию (14, None)
            Размер фигуры. Высота автоматически подстраивается под число колонок,
            если указана как None.
        palette : str, по умолчанию "Reds_r"
            Цветовая палитра seaborn для столбцов.
        show_values : bool, по умолчанию True
            Отображать ли числовые значения (%) на концах столбцов.
        value_threshold : float, по умолчанию 1.5
            Минимальный % пропуска, при котором значение отображается на графике.
        grid_color : str, по умолчанию "#e0e0e0"
            Цвет сетки.
        spine_color : str, по умолчанию "#ddd"
            Цвет осевых линий (spines).
        title : Optional[str], по умолчанию None
            Заголовок графика. Если None - генерируется автоматически.
        xlabel : str, по умолчанию "Процент пропусков (%)"
            Подпись оси X.
        ylabel : str, по умолчанию ""
            Подпись оси Y.
        xticks_step : int, по умолчанию 10
            Шаг делений по оси X (например, каждые 10%).
        dpi : Optional[int], по умолчанию None
            Разрешение графика.
        use_descriptions : bool, по умолчанию True
            Использовать ли COLUMN_DESCRIPTIONS для подписей колонок.
    
    Возвращает:
        None - отображает график через plt.show().
    """
    # Подготовка данных
    na_percent = df.isna().mean() * 100
    na_percent = na_percent[na_percent > threshold].sort_values(ascending=False)
    
    if top_n is not None:
        na_percent = na_percent.head(top_n)
    
    if na_percent.empty:
        print("✔️ Пропусков не обнаружено")
        return
    
    # Автоматическая высота
    height = max(4, len(na_percent) * 0.45) if figsize[1] is None else figsize[1]
    figsize_use = (figsize[0], height)
    
    # DPI
    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    # Стиль
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize_use)
    
    # Цвета
    colors = sns.color_palette(palette, len(na_percent))
    
    # Подписи с учётом описаний
    if use_descriptions:
        y_labels = [
            label_for_column(col, separator="•", format="string")
            for col in na_percent.index
        ]
    else:
        y_labels = list(na_percent.index)
    
    # Горизонтальные столбцы
    bars = ax.barh(y_labels, na_percent.values, color=colors, edgecolor='white', linewidth=1.2)
    
    # Подписи значений
    if show_values:
        for i, pct in enumerate(na_percent.values):
            if pct >= value_threshold:
                ax.text(pct + 0.6, i, f"{pct:.1f}%", va='center', fontsize=9, color='#222')
    
    # Заголовок
    if title is None:
        title = f"Процент пропусков по колонкам ({len(na_percent)} из {df.shape[1]} с пропусками)"
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # Оси
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, xticks_step))
    
    # Спайны
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_color(spine_color)
    
    # Сетка
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7, color=grid_color, alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
def plot_binary_heatmap(
    df: pd.DataFrame,
    legend_labels: Tuple[str, str] = ("Данные есть", "Пропуск"),
    figsize: Tuple[float, float] = (16, 3),
    cmap_present: str = "#31434E",
    cmap_missing: str = "#f2d70e",
    legend_loc: str = "upper right",
    title: Optional[str] = None,
    xlabel: str = "Строки наблюдения",
    ylabel: str = "Колонки с пропусками",
    rotate_yticks: bool = True,
    max_y_labels: int = 20,
    dpi: Optional[int] = None,
    use_descriptions: bool = True
) -> None:
    """
    Визуализирует бинарную матрицу в виде тепловой карты с кастомной легендой.
    
    Описание:
        Функция отображает DataFrame, содержащий бинарные значения (True/False, 0/1, NaN/not-NaN),
        в виде heatmap, где каждый пиксель соответствует ячейке:
        - `cmap_negative` — значение считается "отрицательным" (например, данные есть, ошибка отсутствует),
        - `cmap_positive` — значение считается "положительным" (например, пропуск, аномалия, событие).
        
        Поддерживает использование описаний колонок через COLUMN_DESCRIPTIONS.
    
    Параметры:
        df : pd.DataFrame
            Бинарный датафрейм. Значения интерпретируются как:
            - "Положительные": True, 1, np.nan (если остальные — not-NaN)
            - "Отрицательные": False, 0, not-NaN
        figsize : Tuple[float, float], по умолчанию (16, 3)
            Размер фигуры (ширина, высота).
        cmap_negative : str, по умолчанию "#31434E"
            Цвет для "отрицательных" значений (фон, норма, данные есть).
        cmap_positive : str, по умолчанию "#f2d70e"
            Цвет для "положительных" значений (аномалии, пропуски, ошибки).
        legend_labels : Tuple[str, str], по умолчанию ("Отсутствует", "Присутствует")
            Подписи для легенды: (отрицательный класс, положительный класс).
        legend_loc : str, по умолчанию "upper right"
            Расположение легенды.
        title : Optional[str], по умолчанию None
            Заголовок графика. Если None — генерируется автоматически.
        xlabel : str, по умолчанию "Номер строки (наблюдения)"
            Подпись оси X.
        ylabel : str, по умолчанию "Колонки"
            Подпись оси Y.
        rotate_yticks : bool, по умолчанию True
            Поворачивать ли подписи колонок.
        max_y_labels : int, по умолчанию 20
            Максимальное число колонок, при котором подписи не поворачиваются.
        dpi : Optional[int], по умолчанию None
            Разрешение графика.
        use_descriptions : bool, по умолчанию True
            Использовать ли COLUMN_DESCRIPTIONS для подписей колонок.
    
    Примеры использования:
        # 1. Карта пропусков
        missing_df = df.isna()
        plot_binary_heatmap(missing_df, 
                           cmap_negative="#31434E", 
                           cmap_positive="#f2d70e",
                           legend_labels=("Данные есть", "Пропуск"),
                           title="Пропуски в данных")

        # 2. Карта аномалий давления
        error_df = pd.DataFrame({
            'invalid_bp': df['systolic'] <= df['diastolic']
        })
        plot_binary_heatmap(error_df,
                           cmap_negative="#31434E",
                           cmap_positive="#ff4444",
                           legend_labels=("Корректно", "Ошибка: systolic ≤ diastolic"),
                           title="Аномалии артериального давления")
    
    Возвращает:
        None — отображает график через plt.show().
    """
    # Выбор колонок с пропусками
    cols_with_na = df.columns[df.isna().any()]
    if len(cols_with_na) == 0:
        print("✔️ Пропусков не обнаружено")
        return
    
    df_na = df[cols_with_na].isna()
    
    # DPI
    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    # Стиль
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap без colorbar
    sns.heatmap(
        df_na.T,
        cmap=[cmap_present, cmap_missing],
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        ax=ax
    )
    
    # Легенда
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=cmap_present, label=legend_labels[0]),
        mpatches.Patch(color=cmap_missing, label=legend_labels[1])
    ]
    ax.legend(handles=legend_elements, loc=legend_loc, title="Статус данных", frameon=True)
    
    # Заголовок
    if title is None:
        title = f"Пропуски в {df_na.shape[1]} из {df.shape[1]} колонок"
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Подписи колонок с учётом описаний
    if use_descriptions:
        y_labels = [
            label_for_column(col, separator="•", format="string")
            for col in cols_with_na
        ]
        ax.set_yticklabels(y_labels)
    
    # Поворот подписей
    if len(cols_with_na) <= max_y_labels:
        plt.yticks(rotation=0)
    elif rotate_yticks:
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# compare_train_test_overview - EDA: сравнение train/test с обнаружением data drift
def compare_train_test_overview(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    numeric_threshold: float = 0.35,
    categorical_threshold: float = 0.015,
    max_categories: int = 100,
    show_plot: Union[bool, Literal["all", "problematic"]] = False,
    palette: str = "tab10"
) -> pd.DataFrame:
    """
    Профессиональный EDA: сравнение train/test с обнаружением data drift.
    Выявляет практически важные расхождения и отображает статистическую значимость отдельно.

    Улучшения:
        • Пороги работают напрямую: если Cohen's d > numeric_threshold → флагуется
        • Стат. значимость (p-value) отображается в отдельной колонке - не влияет на флаг
        • Рекомендации адаптивны: «Сдвиг подтверждён» vs «Сдвиг возможен»
        • Поддержка `target` - анализ сдвига таргета и его исключение из сравнения
        • Авто-обнаружение ID-колонок (только для информации)
        • Итоговая сводка: «Найдено проблем: X из Y»

    Метрики и пороги:
        - Числовые признаки: **Cohen's d** (нормированная разница средних)
            • Порог по умолчанию: 0.35 (0.2 = малый эффект, 0.5 = средний)
        - Категориальные признаки: **Cramer's V** (сила ассоциации между распределениями)
            • Порог по умолчанию: 0.1 (0.1 = слабая связь, 0.3 = умеренная, 0.5 = сильная)
        - Для таргета используется пониженный порог (0.1) для раннего предупреждения

    Параметры:
        train: pd.DataFrame - обучающая выборка
        test: pd.DataFrame - тестовая выборка
        target: Optional[str] - целевая переменная (автоисключение из сравнения)
        include/exclude: Optional[List[str]] - фильтрация признаков
        numeric_threshold: float - порог Cohen's d для числовых признаков (по умолчанию 0.35)
        categorical_threshold: float - порог Cramer's V для категориальных признаков (по умолчанию 0.1)
        max_categories: int - макс. число категорий для анализа (по умолчанию 100)
        show_plot: bool - рисовать графики проблемных признаков (по умолчанию False)
        palette: str - палитра для графиков (по умолчанию "tab10")

    Возвращает: 
        pd.DataFrame - отчёт с колонками:
            - Признак
            - Описание
            - Тип (числовой / категориальный)
            - Статус (✔️ Ok / 🚨 Расхождение / ⚠️ Почти константный)
            - Расхождение (Cohen's d или Cramer's V)
            - Метрика
            - p-value (статистическая значимость)
            - Рекомендация

    Примечания:
        - Cohen's d = |μ₁ - μ₂| / σ_pooled - масштаб-инвариантная мера эффекта
        - Cramer's V = √(χ² / (n × min(r-1, c-1))) - нормированная мера ассоциации

    Статистические тесты:
        - Для числовых признаков:
            • Welch's t-test (если ≥20 уникальных значений в обеих выборках),
            • Mann-Whitney U test (в остальных случаях, непараметрический).
        - Для категориальных признаков:
            • Chi-square test of independence на таблице сопряжённости.
        - p-value отображается только как дополнительная информация;
          флаг «🚨 Расхождение» ставится по размеру эффекта (Cohen's d / Cramer's V),
          что предотвращает ложные срабатывания на больших выборках.

    Примеры:
        >>> # Базовый вызов без таргета
        report = compare_train_test_overview(X_train, X_test)

        >>> # С таргетом и графиками проблемных признаков
        report = compare_train_test_overview(
            train=df_train,
            test=df_test,
            target='profit',
            exclude=['id'],
            numeric_threshold=0.35,
            categorical_threshold=0.1,
            show_plot=True
        )

        >>> # Сниженный порог для чувствительного анализа
        report = compare_train_test_overview(
            train=df_train,
            test=df_test,
            numeric_threshold=0.1,      # малый эффект
            categorical_threshold=0.05  # слабая связь
        )    
    """    

    # 1. Общие колонки
    common_cols = sorted(set(train.columns) & set(test.columns))
    common_cols_original = common_cols.copy()
    if target and target in common_cols:
        target_name, target_desc = label_for_column(target, separator="•")
        print(f"🎯 Целевая переменная '{target_name}'{target_desc}")
        print(f"    👁️‍🗨️ Проверка сдвига распределения...")
        # Анализ таргета
        target_has_drift = False
        tr = train[target].dropna()
        te = test[target].dropna()
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            n_tr, n_te = len(tr), len(te)
            if n_tr > 1 and n_te > 1:
                pooled_std = np.sqrt(((n_tr - 1) * tr.std()**2 + (n_te - 1) * te.std()**2) / (n_tr + n_te - 2))
                cohens_d = abs(tr.mean() - te.mean()) / pooled_std if pooled_std > 0 else 0.0
                try:
                    _, p_val = ttest_ind(tr, te, equal_var=False)
                except:
                    p_val = np.nan
                if cohens_d > 0.1:
                    print(f"    🚨 Сдвиг таргета:  d={cohens_d:.3f}, p={p_val:.3f}\n")
                    target_has_drift = True
                else:
                    print(f"    ✔️ Таргет стабилен: Cohen's d={cohens_d:.3f}\n")
        else:
            # Объединяем данные для анализа
            train_labeled = train[[target]].copy()
            train_labeled['dataset'] = 'train'
            test_labeled = test[[target]].copy()
            test_labeled['dataset'] = 'test'
            combined_target = pd.concat([train_labeled, test_labeled], ignore_index=True)

            # Таблица сопряжённости
            observed = pd.crosstab(combined_target['dataset'], combined_target[target])

            # Chi-square и Cramer's V
            try:
                chi2, p_val, dof, expected = chi2_contingency(observed)
                n = observed.sum().sum()
                min_dim = min(observed.shape) - 1
                if min_dim > 0 and n > 0:
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    cramers_v = min(cramers_v, 1.0)
                else:
                    cramers_v = 0.0
                    p_val = np.nan
            except:
                cramers_v = 0.0
                p_val = np.nan

            if cramers_v > 0.1:
                print(f"    🚨 Сдвиг таргета: Cramer's V={cramers_v:.3f}, p={p_val:.3f}\n")
                target_has_drift = True
            else:
                print(f"    ✔️ Таргет стабилен: Cramer's V={cramers_v:.3f}\n")
        common_cols = [col for col in common_cols if col != target]

    if include is not None:
        common_cols = [col for col in common_cols if col in include]
    if exclude is not None:
        common_cols = [col for col in common_cols if col not in exclude]

    if not common_cols:
        print("⚠️ Нет общих колонок для анализа")
        return pd.DataFrame()

    # 1.5. Авто-обнаружение ID-подобных колонок
    id_candidates = []
    for col in common_cols:
        is_unique_train = (train[col].nunique() == len(train)) if len(train) > 1 else False
        is_unique_test = (test[col].nunique() == len(test)) if len(test) > 1 else False
        if is_unique_train and is_unique_test:
            if re.search(r'id$', col, re.IGNORECASE):
                id_candidates.append(col)
            elif train[col].dtype == 'object':
                non_null = pd.concat([train[col], test[col]], ignore_index=True).dropna().astype(str)
                if len(non_null) > 0 and non_null.str.match(r'^[a-zA-Z0-9._-]+$').all():
                    id_candidates.append(col)

    train_name, train_desc = label_for_dataset(train, separator='•')
    test_name, test_desc = label_for_dataset(test, separator='•')

    if id_candidates:
        print(f"📢 Обнаружены потенциальные 🆔 колонки: {id_candidates}")
        print(f"    📌 Рекомендуется исключить их через exclude=.\n")

    print(f"🕵 Анализ согласованности:")
    print(f"    🗃️ {train_name}{train_desc}")
    print(f"    🗃️ {test_name}{test_desc}")
    print(f"по {len(common_cols)} признакам")

    # 2. Разделение на типы
    numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(test[col])]
    categorical_cols = [col for col in common_cols if col not in numeric_cols]
    all_analyzed_cols = numeric_cols + categorical_cols

    results = []

    # 3. Числовые признаки - Cohen's d + p-value
    for col in numeric_cols:
        tr = train[col].dropna()
        te = test[col].dropna()
        if len(tr) == 0 or len(te) == 0:
            continue

        if tr.nunique() <= 1 and te.nunique() <= 1:
            results.append({
                "Признак": col,
                "Тип": "1️⃣ числовой",
                "Статус": "⚠️ Почти константный",
                "Расхождение": 0.0,
                "Метрика": "почти константный",
                "p-value": np.nan,
                "Рекомендация": "Проверить источник данных"
            })
            continue

        n_tr, n_te = len(tr), len(te)
        mean_tr, mean_te = tr.mean(), te.mean()
        std_tr, std_te = tr.std(), te.std()

        pooled_std = np.sqrt(((n_tr - 1) * std_tr**2 + (n_te - 1) * std_te**2) / (n_tr + n_te - 2))
        cohens_d = abs(mean_tr - mean_te) / pooled_std if pooled_std > 0 else 0.0

        # Статистическая значимость (только для информации)
        p_val = np.nan
        if n_tr > 1 and n_te > 1:
            try:
                if min(tr.nunique(), te.nunique()) > 20:
                    _, p_val = ttest_ind(tr, te, equal_var=False)
                else:
                    _, p_val = mannwhitneyu(tr, te, alternative='two-sided')
            except:
                p_val = np.nan

        # Решение по практической важности
        if cohens_d > numeric_threshold:
            status = "🚨 Расхождение"
            if not np.isnan(p_val) and p_val < 0.05:
                rec = "Сдвиг подтверждён статистически"
            else:
                rec = "Сдвиг возможен - проверьте объём данных и дисперсию"
        else:
            status = "✔️ Ok"
            rec = ""

        col_name, col_desc = label_for_column(col, separator="")

        results.append({
            "Признак": col_name,
            "Описание": col_desc,
            "Тип": "1️⃣ числовой",
            "Статус": status,
            "Расхождение": cohens_d,
            "Метрика": "Cohen's d",
            "p-value": p_val,
            "Рекомендация": rec
        })

    # 4. Категориальные признаки - Cramer's V + chi-square p-value
    min_freq = 0.005
    for col in categorical_cols:
        if max(train[col].nunique(), test[col].nunique()) > max_categories:
            continue

        # Объединяем данные для анализа
        train_labeled = train[[col]].copy()
        train_labeled['dataset'] = 'train'
        test_labeled = test[[col]].copy()
        test_labeled['dataset'] = 'test'
        combined = pd.concat([train_labeled, test_labeled], ignore_index=True)

        # Фильтрация редких категорий
        value_counts = combined[col].value_counts(normalize=True)
        keep_cats = value_counts[value_counts >= min_freq].index
        if len(keep_cats) == 0:
            continue
        combined = combined[combined[col].isin(keep_cats)]

        # Таблица сопряжённости
        observed = pd.crosstab(combined['dataset'], combined[col])

        # Chi-square test и Cramer's V
        try:
            chi2, p_val, dof, expected = chi2_contingency(observed)
            n = observed.sum().sum()
            min_dim = min(observed.shape) - 1
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                cramers_v = min(cramers_v, 1.0)  # защита от численных ошибок
            else:
                cramers_v = 0.0
                p_val = np.nan
        except:
            cramers_v = 0.0
            p_val = np.nan

        # Решение по практической важности
        if cramers_v > categorical_threshold:
            status = "🚨 Расхождение"
            if not np.isnan(p_val) and p_val < 0.05:
                rec = "Сдвиг подтверждён статистически"
            else:
                rec = "Сдвиг возможен - проверьте объём данных"
        else:
            status = "✔️ Ok"
            rec = ""

        col_name, col_desc = label_for_column(col, separator="")

        results.append({
            "Признак": col_name,
            "Описание": col_desc,
            "Тип": "🏷️ категориальный",
            "Статус": status,
            "Расхождение": cramers_v,
            "Метрика": "Cramer's V",
            "p-value": p_val,
            "Рекомендация": rec
        })

    # 5. Вывод отчёта
    if not results:
        print("✔️ Нет признаков для анализа")
        return pd.DataFrame()

    report_df = pd.DataFrame(results)

    # Сводка
    n_issues = len(report_df[report_df["Статус"].str.contains("🚨")])
    print(f"\n🔍 Найдено проблем: {n_issues} из {len(report_df)} признаков")
    if n_issues == 0:
        print("💎 Данные train/test согласованы - можно переходить к моделированию!")

    # Сортировка: сначала по типу, потом по имени
    report_df["Признак_сортировка"] = report_df["Признак"].apply(
        lambda x: x.split("•")[0].strip() if "•" in x else x
    )
    report_df["Тип_сортировка"] = report_df["Тип"].map({"1️⃣ числовой": 0, "🏷️ категориальный": 1})
    report_df = report_df.sort_values(
        ["Тип_сортировка", "Признак_сортировка"]
    ).reset_index(drop=True)
    report_df = report_df.drop(columns=["Тип_сортировка", "Признак_сортировка"])

    # Нумерация
    cols_order = ['Признак', 'Описание', 'Тип', 'Статус', 'Расхождение', 'Метрика', 'p-value', 'Рекомендация']
    report_df = report_df[cols_order]

    def _color_status(val):
        if "🚨" in val:
            return "background-color: #ffebee; color: #c62828"
        elif "⚠️" in val:
            return "background-color: #fff3e0; color: #ef6c00"
        return ""

    display_table(
        report_df,
        rows=len(report_df),
        float_precision=3,
        styler_func=lambda s: s.applymap(_color_status, subset=["Статус"])
    )    

    # 6. Графики - поддержка трёх режимов
    plot_mode = show_plot
    if isinstance(plot_mode, bool):
        plot_mode = "problematic" if plot_mode else None

    cols_to_plot = []

    if plot_mode == "problematic":
        # Только признаки с 🚨 + target (если проблемный)
        cols_to_plot = report_df[report_df["Статус"].str.contains("🚨")]["Признак"].tolist()
        if target and target_has_drift:
            cols_to_plot = [target] + cols_to_plot

    elif plot_mode == "all":
        # Все проанализированные признаки + target (если анализировался)
        cols_to_plot = all_analyzed_cols.copy()
        if target and target in common_cols_original:
            # target исключён из анализа, но если он был - добавим при условии
            if target_has_drift:
                cols_to_plot = [target] + cols_to_plot
            else:
                # Даже если не дрифтанул - можно показать, т.к. режим 'all'
                cols_to_plot = [target] + cols_to_plot

    # Ограничение на 15 графиков остаётся (по соображениям производительности)
    if cols_to_plot:
        print(f"\n📊 Визуализация ({len(cols_to_plot)} признаков, показаны первые 5):")
        for col in cols_to_plot[:15]:
            try:
                col_name, col_desc = label_for_column(col, separator="•")
                print(f"\nСтатистика 📄 {col_name}{col_desc}")
                plot_train_test_distribution(train, test, col, palette=palette, table_metrics='extended')
            except Exception as e:
                print(f"⚠️ Ошибка при построении графика для {col}: {e}")

    return report_df



#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••



# validate_datasets_consistency: Проверяет согласованность идентификаторов между основным и зависимыми датасетами
def validate_datasets_consistency(
    master_df: pd.DataFrame,
    id_column: str,
    dependent_dfs: list[pd.DataFrame]
) -> None:
    """
    Проверяет согласованность идентификаторов между основным и зависимыми датасетами.
    
    Описание:
        Сравнивает множество ID из мастер-датасета с ID в каждом зависимом датасете.
        Использует `label_for_dataset` для автоматического определения имён таблиц.
        Для каждой таблицы выводит:
        - отсутствующие ID (есть в мастер, но нет в зависимой),
        - лишние ID (есть в зависимой, но нет в мастер).
        Помогает избежать пропусков при join и "мусорных" записей.

    Параметры:
        master_df : pd.DataFrame
            Основной датасет - источник истины по списку клиентов/объектов.
        id_column : col
            Название столбца с идентификаторами (должен быть во всех датасетах).
        dependent_dfs : list[pd.DataFrame]
            Список зависимых датафреймов (без явных имён - имена определяются автоматически).

    Возвращаемое значение:
        None - результат выводится напрямую в ячейку ноутбука.
        
    Пример использования:
        validate_datasets_consistency(
            master_df=df_market_file,
            id_column='id',
            dependent_dfs=[df_market_money, df_market_time, df_money]
        )
    """
    # Проверка наличия id_column в мастер-датасете
    if id_column not in master_df.columns:
        print(f"❌ ОШИБКА: в мастер-датасете отсутствует столбец '{id_column}'")
        return

    # Получаем имя и описание мастер-датасета
    master_name, master_desc = label_for_dataset(master_df, separator="•")
    master_label = f"{master_name}{master_desc}" if master_desc else master_name

    master_ids = set(master_df[id_column].unique())
    n_master = len(master_ids)


    col_name, col_desc = label_for_column(id_column, separator="()")
    full_col_name = f"'{col_name}'{col_desc}"

    dataset_profile(master_df, report='summary')
    print(f"\n🕵 Проверка согласованности датасетов по столбцу {full_col_name}")

    all_good = True

    for df in dependent_dfs:
        # Получаем имя и описание зависимого датасета
        df_name, df_desc = label_for_dataset(df, separator="•")
        df_label = f"{df_name}{df_desc}" if df_desc else df_name

        if id_column not in df.columns:
            print(f"     ❌ ОШИБКА: в датасете '{df_label}' отсутствует столбец '{id_column}'")
            all_good = False
            continue

        table_ids = set(df[id_column].unique())
        missing = master_ids - table_ids      # есть в мастер, нет в зависимой
        extra = table_ids - master_ids        # есть в зависимой, нет в мастер

        if not missing and not extra:
            print(f"     ✔️ {df_label} 💎 полное совпадение '{col_name}'")
        else:
            print(f"     ⚠️  {df_label}:")
            if missing:
                print(f"    ❌ Отсутствуют в таблице: {sorted(missing)}")
            if extra:
                print(f"    🗑️ Лишние ID: {sorted(extra)}")
            all_good = False

    if all_good:
        print(f"\n💎 Все датасеты согласованы по столбцу {full_col_name}")
    else:
        print(f"\n💡 Совет: для корректного join оставьте в зависимых таблицах только ID из мастер-датасета.")


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


# check_train_test_id_leakage - Проверяет наличие утечки данных через пересечение ID между train и test
def check_train_test_id_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str = "id"
) -> bool:
    """
    Проверяет наличие утечки данных через пересечение ID между train и test.
    
    Описание:
        Сравнивает множества ID в обучающей и тестовой выборках.
        Если пересечение не пусто - это data leakage, что делает оценку модели некорректной.
        Использует `_label_for_dataset` для автоматической подписи источников.
    
    Параметры:
        train: pd.DataFrame - обучающая выборка
        test: pd.DataFrame - тестовая выборка
        id_column: str - название колонки с идентификаторами (по умолчанию 'id')
    
    Возвращает:
        bool - True, если утечка обнаружена (пересечение ≠ ∅), иначе False
    
    Пример:
        >>> has_leak = check_train_test_id_leakage(df_train, df_test, id_column='user_id')
        >>> if has_leak:
        ...     print("⚠️ Утечка данных! Нужно переразделить выборки.")
    """
    if id_column not in train.columns:
        raise ValueError(f"Колонка '{id_column}' отсутствует в train")
    if id_column not in test.columns:
        raise ValueError(f"Колонка '{id_column}' отсутствует в test")
    
    train_ids = set(train[id_column].dropna().unique())
    test_ids = set(test[id_column].dropna().unique())
    overlap = train_ids & test_ids
    
    train_name, train_desc = label_for_dataset(train, separator="•")
    test_name, test_desc = label_for_dataset(test, separator="•")
    
    if overlap:
        print(f"🚨 УТЕЧКА ДАННЫХ: обнаружено {len(overlap)} пересекающихся ID между {train_name} и {test_name}")
        if len(overlap) <= 10:
            print(f"   📌 ID: {sorted(overlap)}")
        else:
            sample_overlap = sorted(list(overlap))[:5]
            print(f"   📌 Примеры ID: {sample_overlap} ... (всего {len(overlap)})")
        return True
    else:
        print(f"🕵 Проверка наличия утечки данных через пересечение [ {id_column} ] в датафреймах :")        
        print(f"    🗃️ {train_name}{train_desc}")
        print(f"    🗃️ {test_name}{test_desc}")
        print(f"✔️ Нет пересечения [ {id_column} ]")
        print(f"💎 Данные разделены корректно")
        return False



# •••••••••• КОНЕЦ ФУНКЦИЙ 3Filoff ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••



# Экспорт только публичных функций (без подчёркивания)
__all__ = [
    "set_global_styles",
    "set_output_mode",
    "label_for_column",
    "label_for_dataset",
    "display_table",
    "standardize_column_names",
    "preview",
    "dataset_convert_datetime",
    "load_dataset",
    "dataset_profile",
    "dataset_quick_audit",
    "dataset_overview",
    "handle_duplicates",
    "audit_numerical",
    "report_numerical_consistency",
    "audit_categorical",
    "audit_categorical_frequencies",
    "audit_categorical_cross",
    "audit_categorical_typos",
    "audit_numerical_distribution",
    "plot_feature_distribution",
    "plot_feature_distribution_advanced",
    "plot_target_relationships",
    "plot_mixed_correlation",
    "plot_pairwise_correlations",
    "plot_categorical_distribution",
    "plot_phik_correlation",
    "plot_train_test_distribution",
    "plot_compare_train_test_ecdf",
    "plot_shap_summary",
    "plot_discrete_train_test",
    "plot_numerical_profile",
    "plot_missing_summary",
    "plot_binary_heatmap",
    "compare_train_test_overview",
    "validate_datasets_consistency",
    "check_train_test_id_leakage"
]