# src/utils/preprocessing.py

"""
Утилиты для препроцессинга данных.
"""

import pandas as pd
import re


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
        'Heart Rate'  - 'heart_rate'
        'CK-MB'       - 'ck_mb'
        'Systolic BP' - 'systolic_bp'
    
    Параметры:
        df: pd.DataFrame
            Датафрейм, колонки которого будут переименованы на месте.
        verbose: bool, по умолчанию False
            Выводить ли информацию о выполненных преобразованиях.
        handle_camel_case: bool, по умолчанию True
            Преобразовывать ли CamelCase - snake_case.
    
    Возвращает:
        None. Изменения применяются непосредственно к переданному датафрейму.
    
    Исключения:
        ValueError: если после преобразования возникают дублирующиеся имена колонок.
    
    Примеры:
        >>> standardize_column_names(df)
        >>> standardize_column_names(df, verbose=True)
    
    Зависимости:
        import re
    """
    def _to_snake_case(name: str, handle_camel: bool = True) -> str:
        # Шаг 1: заменяем все недопустимые символы на подчёркивания
        s1 = re.sub(r'[^a-zA-Z0-9]+', '_', name)
        # Шаг 2: (опционально) преобразуем CamelCase - snake_case
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
