"""Оркестратор ETL. Запуск: python pipeline.py из папки etl."""

import sys
from pathlib import Path

# Добавляем папку выше etl в sys.path, чтобы пакет etl был виден
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.extract  import load_raw
from etl.clean    import clean_sales
from etl.features import build_features
from etl.marts    import build_marts, export_all, validate, write_summary


def main() -> None:
    print("1. Extract")
    raw = load_raw()

    print("2. Clean")
    clean = clean_sales(raw)

    print("3. Features")
    features = build_features(clean)

    print("4. Marts")
    marts = build_marts(clean, features)

    print("5. Validate")
    validate(clean, marts)

    print("6. Export")
    export_all(marts)

    print("7. Summary")
    write_summary(clean, marts)

    print("\nETL завершён.")


if __name__ == "__main__":
    main()