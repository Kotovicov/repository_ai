from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    find_constant_columns,
    find_suspicious_id_duplicates,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)
    
    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_constant_columns_flag():
    # DataFrame с одинаковыми значениями в колонке
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [5, 5, 5, 5],  # Все значения одинаковые
        "normal_col": [10, 20, 30, 40],
        "mixed_col": [None, "A", "A", "B"]  # Не одинаковые
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг установлен в True
    assert flags["has_constant_columns"] == True
    
    # Проверяем функцию find_constant_columns отдельно
    constant_cols = find_constant_columns(df)
    assert "constant_col" in constant_cols
    assert "id" not in constant_cols  
    assert "normal_col" not in constant_cols  
    assert "mixed_col" not in constant_cols 


def test_suspicious_id_duplicates_flag():
    # DataFrame с дубликатами в ID-колонках
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2],  # Есть дубликаты
        "id_column": [100, 101, 102, 103, 104],  # Нет дубликатов
        "customer_id": [10, 10, 11, 12, 13],  # Есть дубликаты
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "normal_column": [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_suspicious_id_duplicates"] == True
    
    has_duplicates = find_suspicious_id_duplicates(df)
    assert has_duplicates == True 

def test_no_suspicious_id_duplicates():
    # DataFrame без дубликатов в ID-колонках
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],  
        "id_column": [100, 101, 102, 103, 104],  
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "normal_column": [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_suspicious_id_duplicates"] == False
    
    has_duplicates = find_suspicious_id_duplicates(df)
    assert has_duplicates == False


def test_constant_columns_with_nulls():
    # DataFrame с константной колонкой, содержащей None
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_with_nulls": [5, 5, 5, None],  # Все не-NaN значения одинаковые
        "non_constant_with_nulls": [5, 5, 10, None],  # Не константная
    })
    
    # Проверяем функцию find_constant_columns
    constant_cols = find_constant_columns(df)
    
    # constant_with_nulls должна быть константной (все не-NaN значения = 5)
    assert "constant_with_nulls" in constant_cols
    # non_constant_with_nulls не должна быть константной (есть 5 и 10)
    assert "non_constant_with_nulls" not in constant_cols
    
    # Проверяем через compute_quality_flags
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_constant_columns"] == True


def test_id_patterns():
    """Тест для различных паттернов ID-колонок"""
    # Проверяем разные варианты написания ID
    test_cases = [
        ("user_id", True),
        ("userId", True),
        ("USER_ID", True),
        ("id", True),
        ("ID", True),
        ("Id", True),
        ("customer_id", True),
        ("productId", True),
        ("_id", True),    # Специальный случай из паттернов
        ("Id_", True),    # Специальный случай из паттернов
    ]
    
    for col_name, should_be_id in test_cases:
        df = pd.DataFrame({col_name: [1, 2, 3, 1]})  # Создаем дубликат
        has_duplicates = find_suspicious_id_duplicates(df)
        
        if should_be_id:
            # Если это ID-колонка и есть дубликаты, функция должна вернуть True
            assert has_duplicates == True
        else:
            # Если это не ID-колонка, функция должна вернуть False
            # (хотя технически в колонке есть дубликаты, но она не распознается как ID)
            assert has_duplicates == False


def test_quality_score_with_issues():
    # DataFrame с несколькими проблемами
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1],  # Дубликаты в ID
        "constant_col": [10, 10, 10, 10],  # Константная колонка
        "age": [20, None, 30, None],  # Пропуски (50%)
        "name": ["A", "B", "C", "D"]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем все флаги
    assert flags["has_constant_columns"] == True
    assert flags["has_suspicious_id_duplicates"] == True
    assert flags["too_many_missing"] == False  # 50% не превышает порог 50%
    assert flags["max_missing_share"] == 0.5  # age имеет 50% пропусков
    
    # Проверяем, что качество снижено из-за проблем
    # Базовая оценка: 1.0
    # - 0.5 за пропуски (max_missing_share)
    # - 0.1 за константные колонки
    # - 0.2 за дубликаты ID
    # Итого: 1.0 - 0.5 - 0.1 - 0.2 = 0.2
    # Но также может быть вычтено 0.2 за мало строк (n_rows < 100)
    # Итого: 0.0
    assert flags["quality_score"] == 0.0