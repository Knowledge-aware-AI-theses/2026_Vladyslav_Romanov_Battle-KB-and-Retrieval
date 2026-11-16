import json
from pathlib import Path


JSON_DIR = Path("data/json_kb_v2")


def is_non_empty(value):
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return True


def get_field_values(data, field_path):
    parts = field_path.split(".")
    values = [data]

    for part in parts:
        next_values = []

        is_array = part.endswith("[]")
        key = part[:-2] if is_array else part

        for value in values:
            if not isinstance(value, dict):
                continue

            if key not in value:
                continue

            field_value = value[key]

            if is_array:
                if isinstance(field_value, list):
                    next_values.extend(field_value)
            else:
                next_values.append(field_value)

        values = next_values

    return values


def has_field_value(data, field_path):
    values = get_field_values(data, field_path)

    return any(is_non_empty(value) for value in values)


def json_coverage(field_paths):
    if isinstance(field_paths, str):
        field_paths = [field_paths]

    json_files = list(JSON_DIR.glob("Q*.json"))

    total_battles = len(json_files)
    battles_with_field = 0

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            battle = json.load(f)

        if any(has_field_value(battle, path) for path in field_paths):
            battles_with_field += 1

    coverage = (
        battles_with_field / total_battles * 100
        if total_battles > 0
        else 0
    )

    return {
        "fields": field_paths,
        "total_battles": total_battles,
        "battles_with_field": battles_with_field,
        "coverage_percent": round(coverage, 2)
    }


if __name__ == "__main__":
    result = json_coverage("narrative_patterns.similarity_tags")
    print(result)