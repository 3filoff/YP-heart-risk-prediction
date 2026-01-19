import pandas as pd
from typing import Set, Optional
from src.utils.preprocessing import standardize_column_names


class DataPreprocessor:
    """
    Подготавливает входные данные для модели предсказания риска сердечного приступа.
    
    Обеспечивает:
    - унификацию имён колонок (snake_case)
    - проверку обязательных признаков
    - обработку пола и пропусков
    - исключение диагностических маркеров
    - добавление флага отсутствия анамнеза
    - добавление флага некорректного кровяного давления
    - оптимальную типизацию колонок (память + производительность)
    """

    REQUIRED_COLUMNS: Set[str] = {
        'age', 'cholesterol', 'heart_rate', 'diabetes', 'family_history',
        'smoking', 'obesity', 'alcohol_consumption', 'exercise_hours_per_week',
        'diet', 'previous_heart_problems', 'medication_use', 'stress_level',
        'sedentary_hours_per_day', 'income', 'bmi', 'triglycerides',
        'physical_activity_days_per_week', 'sleep_hours_per_day',
        'blood_sugar', 'ck_mb', 'troponin', 'gender',
        'systolic_blood_pressure', 'diastolic_blood_pressure'
    }

    DROP_COLUMNS: Set[str] = {'id'}

    ANAMNESIS_COLUMNS: Set[str] = {
        'diabetes', 'family_history', 'smoking', 'obesity',
        'alcohol_consumption', 'previous_heart_problems',
        'medication_use', 'stress_level', 'physical_activity_days_per_week'
    }

    LEAKY_COLUMNS: Set[str] = {'ck_mb', 'troponin'}

    # Нормализованные непрерывные признаки (ожидаются в [0, 1])
    NORMALIZED_FEATURES: Set[str] = {
        'age', 'cholesterol', 'heart_rate', 'alcohol_consumption',
        'exercise_hours_per_week', 'sedentary_hours_per_day', 'income',
        'bmi', 'triglycerides', 'sleep_hours_per_day', 'blood_sugar',
        'systolic_blood_pressure', 'diastolic_blood_pressure'
    }

    def __init__(
        self,
        drop_leaky_features: bool = True,
        add_missing_anamnesis_flag: bool = True,
        target_column: str = 'heart_attack_risk_binary'
    ):
        self.drop_leaky_features = drop_leaky_features
        self.add_missing_anamnesis_flag = add_missing_anamnesis_flag
        self.target_column = target_column
        self._is_fitted = False

    def _unify_gender(self, series: pd.Series) -> pd.Series:
        def _parse_gender(val):
            if pd.isna(val):
                return -1
            val_str = str(val).strip()
            if val_str in {'0', '0.0', 'Female', 'female', 'F', 'f', '0.00'}:
                return 0
            elif val_str in {'1', '1.0', 'Male', 'male', 'M', 'm', '1.00'}:
                return 1
            else:
                return -1
        return series.apply(_parse_gender).astype('int8')

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет производные признаки.
        """
        df = df.copy()
        if 'systolic_blood_pressure' in df.columns and 'diastolic_blood_pressure' in df.columns:
            df['invalid_blood_pressure'] = (
                df['systolic_blood_pressure'] <= df['diastolic_blood_pressure']
            ).astype('int8')
        return df

    def _set_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит колонки к оптимальным типам данных."""
        df = df.copy()

        # Бинарные анамнестические признаки — float32 (могут быть NaN)
        binary_cols = list(self.ANAMNESIS_COLUMNS)
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        # Дискретные признаки
        if 'diet' in df.columns:
            df['diet'] = df['diet'].astype('int8')
        for col in ['stress_level', 'physical_activity_days_per_week']:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        # gender и флаги — int8
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype('int8')
        if 'missing_anamnesis' in df.columns:
            df['missing_anamnesis'] = df['missing_anamnesis'].astype('int8')
        if 'invalid_blood_pressure' in df.columns:
            df['invalid_blood_pressure'] = df['invalid_blood_pressure'].astype('int8')

        # Непрерывные признаки — float32
        continuous_cols = [
            'age', 'cholesterol', 'heart_rate', 'exercise_hours_per_week',
            'sedentary_hours_per_day', 'income', 'bmi', 'triglycerides',
            'sleep_hours_per_day', 'blood_sugar',
            'systolic_blood_pressure', 'diastolic_blood_pressure'
        ]
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        return df

    def _validate_normalized_bounds(self, df: pd.DataFrame) -> None:
        """Проверяет, что нормализованные признаки в [0, 1]."""
        tolerance = 1e-6
        violations = []
        for col in self.NORMALIZED_FEATURES:
            if col not in df.columns:
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < -tolerance or col_max > 1 + tolerance:
                violations.append((col, col_min, col_max))
        if violations:
            msg = "Обнаружены значения вне допустимого диапазона [0, 1]:\n"
            for col, min_val, max_val in violations:
                msg += f"  • {col}: [{min_val:.6f}, {max_val:.6f}]\n"
            msg += "\nВозможные причины:\n" \
                   "  - Ошибка в исходных данных\n" \
                   "  - Проблема при нормализации\n" \
                   "  - Артефакт обработки"
            raise ValueError(msg)

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Ожидался pd.DataFrame")
        if df.empty:
            raise ValueError("Входной датафрейм пуст")
        if not all(isinstance(col, str) for col in df.columns):
            bad_cols = [col for col in df.columns if not isinstance(col, str)]
            raise TypeError(f"Названия колонок должны быть строками. Найдены: {bad_cols}")

        df = df.copy()

        # 1. Стандартизация имён
        standardize_column_names(df, handle_camel_case=True)

        # 2. Валидация обязательных колонок
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_required}")

        # 3. Обработка gender
        if 'gender' in df.columns:
            df['gender'] = self._unify_gender(df['gender'])

        # 4. Флаг отсутствия анамнеза
        if self.add_missing_anamnesis_flag:
            df['missing_anamnesis'] = (
                df[list(self.ANAMNESIS_COLUMNS)].isnull().all(axis=1).astype('int8')
            )

        # 5. Производные признаки (включая invalid_blood_pressure)
        df = self._add_features(df)

        # 6. Удаление ненужных колонок
        cols_to_drop = set(self.DROP_COLUMNS)
        if self.drop_leaky_features:
            cols_to_drop.update(self.LEAKY_COLUMNS)
        cols_to_drop = cols_to_drop & set(df.columns)
        if cols_to_drop:
            df = df.drop(columns=list(cols_to_drop))

        # 7. Установка оптимальных типов
        df = self._set_dtypes(df)

        # 8. Валидация диапазонов
        self._validate_normalized_bounds(df)

        return df

    def fit_transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        return self.fit(df).transform(df, is_train=is_train)