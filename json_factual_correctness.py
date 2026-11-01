import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from llm import generate_answer, load_llm


JSON_KB_DIR = Path("data/json_kb_v2")
RAW_TEXT_DIR = Path("data/extracted_wikipedia_raw")

RESULTS_DIR = Path("results/fact_validation")
FACTS_FILE = RESULTS_DIR / "facts.json"
VALIDATION_FILE = RESULTS_DIR / "validated_facts.json"

FACTS_PER_BATTLE = 4
RANDOM_SEED = 42
VALIDATION_MAX_NEW_TOKENS = 800

BATTLES = {
    "Q151005": "Battle of Leipzig",
    "Q38789": "Battle of Stalingrad",
    "Q48314": "Battle of Waterloo",
    "Q200056": "Battle of Zama",
    "Q2038907": "Battle of Wuhan",
    "Q33132": "Battle of Gettysburg",
    "Q184320": "Battle of Borodino",
    "Q200855": "Battle of Vienna",
    "Q83224": "Battle of Hastings",
    "Q179591": "Battle of Cannae",
    "Q31900": "Battle of Marathon",
    "Q188495": "Battle of Agincourt",
    "Q134114": "Battle of Austerlitz",
    "Q132568": "Battle of the Somme",
    "Q180182": "Battle of Iwo Jima",
    "Q131969": "Battle of Thermopylae",
    "Q192660": "Battle of Okinawa",
    "Q154182": "Battle of Berlin",
    "Q153858": "Battle of Tannenberg",
    "Q173077": "Battle of Tours",
}

PREFERRED_CATEGORIES = (
    "metadata",
    "participants",
    "outcome",
    "military_facets",
)

VALID_VERDICTS = {
    "supported",
    "contradicted",
    "not_found",
}


def load_json(file_path: str | Path) -> Any:
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _add_candidate(
    candidates: list[dict[str, Any]],
    *,
    category: str,
    field_path: str,
    value: Any,
    fact: str,
) -> None:
    """Append a usable fact candidate linked to one exact JSON field."""
    if value is None or value == "" or value == []:
        return

    candidates.append({
        "category": category,
        "field_path": field_path,
        "value": value,
        "fact": fact,
    })


def build_fact_candidates(
    battle: dict[str, Any],
    qid: str,
) -> list[dict[str, Any]]:
    """Create natural-language facts from selected JSON leaf fields."""
    name = battle.get("identification", {}).get("name")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Missing identification.name in JSON record {qid}."
        )

    name = name.strip()
    candidates: list[dict[str, Any]] = []

    # Metadata fields
    metadata = battle.get("metadata", {}) or {}
    location = metadata.get("location", {}) or {}

    location_name = location.get("name")
    _add_candidate(
        candidates,
        category="metadata",
        field_path="metadata.location.name",
        value=location_name,
        fact=f"{name} took place at {location_name}.",
    )

    region = location.get("region")
    _add_candidate(
        candidates,
        category="metadata",
        field_path="metadata.location.region",
        value=region,
        fact=f"{name} took place in the region of {region}.",
    )

    country = location.get("country")
    _add_candidate(
        candidates,
        category="metadata",
        field_path="metadata.location.country",
        value=country,
        fact=f"{name} took place in {country}.",
    )

    conflict = metadata.get("conflict")
    _add_candidate(
        candidates,
        category="metadata",
        field_path="metadata.conflict",
        value=conflict,
        fact=f"{name} was part of {conflict}.",
    )

    start_date = (
        metadata.get("time_period", {}) or {}
    ).get("start_date", {}) or {}

    if start_date.get("date_label"):
        date_path = "metadata.time_period.start_date.date_label"
        date_value = start_date["date_label"]
        date_fact = f"{name} began on {date_value}."
    elif start_date.get("iso_date"):
        date_path = "metadata.time_period.start_date.iso_date"
        date_value = start_date["iso_date"]
        date_fact = f"{name} began on {date_value}."
    else:
        date_path = "metadata.time_period.start_date.year"
        date_value = start_date.get("year")
        date_fact = f"{name} began in {date_value}."

    _add_candidate(
        candidates,
        category="metadata",
        field_path=date_path,
        value=date_value,
        fact=date_fact,
    )

    # Participant fields
    for participant_index, participant in enumerate(
        battle.get("participants", []) or []
    ):
        if not isinstance(participant, dict):
            continue

        side = participant.get("side")
        _add_candidate(
            candidates,
            category="participants",
            field_path=f"participants[{participant_index}].side",
            value=side,
            fact=f"{side} participated in {name}.",
        )

        role = participant.get("role")
        _add_candidate(
            candidates,
            category="participants",
            field_path=f"participants[{participant_index}].role",
            value=role,
            fact=f"{side or 'This participant'} was the {role} in {name}.",
        )

        for commander_index, commander in enumerate(
            participant.get("commanders", []) or []
        ):
            _add_candidate(
                candidates,
                category="participants",
                field_path=(
                    f"participants[{participant_index}]"
                    f".commanders[{commander_index}]"
                ),
                value=commander,
                fact=f"{commander} was a commander at {name}.",
            )

        strength = participant.get("strength", {}) or {}
        strength_value = strength.get("total")
        strength_path = (
            f"participants[{participant_index}].strength.total"
        )

        if strength_value is None:
            strength_value = strength.get("label")
            strength_path = (
                f"participants[{participant_index}].strength.label"
            )

        _add_candidate(
            candidates,
            category="participants",
            field_path=strength_path,
            value=strength_value,
            fact=(
                f"The reported strength of {side or 'a participant'} "
                f"at {name} was {strength_value}."
            ),
        )

    # Outcome fields
    outcome = battle.get("outcome", {}) or {}

    winner = outcome.get("winner")
    _add_candidate(
        candidates,
        category="outcome",
        field_path="outcome.winner",
        value=winner,
        fact=f"{winner} won {name}.",
    )

    result = outcome.get("result")
    _add_candidate(
        candidates,
        category="outcome",
        field_path="outcome.result",
        value=result,
        fact=f"The recorded result of {name} was {result}.",
    )

    result_type = outcome.get("result_type")
    _add_candidate(
        candidates,
        category="outcome",
        field_path="outcome.result_type",
        value=result_type,
        fact=f"The result type of {name} was {result_type}.",
    )

    # Military-facet fields
    military_facets = battle.get("military_facets", {}) or {}

    field_templates = (
        (
            "battle_type",
            "{value} was a battle type associated with {name}.",
        ),
        ("domain", "{name} was fought in the {value} domain."),
        ("terrain", "{value} terrain was relevant to {name}."),
        ("tactics", "{value} was used as a tactic during {name}."),
        (
            "weapons_or_units",
            "{value} was used or deployed during {name}.",
        ),
        (
            "special_features",
            "{value} was a special feature of {name}.",
        ),
    )

    for field_name, template in field_templates:
        for value_index, value in enumerate(
            military_facets.get(field_name, []) or []
        ):
            _add_candidate(
                candidates,
                category="military_facets",
                field_path=(
                    f"military_facets.{field_name}[{value_index}]"
                ),
                value=value,
                fact=template.format(value=value, name=name),
            )

    # Ensure that one JSON path cannot be sampled twice.
    unique_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique_candidates.setdefault(
            candidate["field_path"],
            candidate,
        )

    return list(unique_candidates.values())


def select_fact_candidates(
    candidates: list[dict[str, Any]],
    facts_per_battle: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Select distinct fields, preferring one from each schema category.

    With four requested facts, this normally selects one metadata,
    participant, outcome, and military-facet field per battle.
    """
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()

    for category in PREFERRED_CATEGORIES:
        category_candidates = [
            candidate
            for candidate in candidates
            if candidate["category"] == category
            and candidate["field_path"] not in selected_paths
        ]

        if category_candidates and len(selected) < facts_per_battle:
            candidate = rng.choice(category_candidates)
            selected.append(candidate)
            selected_paths.add(candidate["field_path"])

    remaining_candidates = [
        candidate
        for candidate in candidates
        if candidate["field_path"] not in selected_paths
    ]
    rng.shuffle(remaining_candidates)

    for candidate in remaining_candidates:
        if len(selected) >= facts_per_battle:
            break

        selected.append(candidate)
        selected_paths.add(candidate["field_path"])

    if len(selected) != facts_per_battle:
        raise ValueError(
            f"Could select only {len(selected)} distinct JSON fields; "
            f"{facts_per_battle} are required."
        )

    return selected


def generate_fact_sample(
    battles: dict[str, str] = BATTLES,
    facts_per_battle: int = FACTS_PER_BATTLE,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Generate exactly four JSON-field facts for each selected battle."""
    if facts_per_battle <= 0:
        raise ValueError("facts_per_battle must be positive.")

    rng = random.Random(seed)
    facts: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for qid, expected_name in battles.items():
        json_path = JSON_KB_DIR / f"{qid}.json"

        if not json_path.exists():
            missing_files.append(str(json_path))
            continue

        battle = load_json(json_path)
        actual_name = battle.get("identification", {}).get("name")

        if not isinstance(actual_name, str) or not actual_name.strip():
            raise ValueError(
                f"Missing identification.name in {json_path}."
            )

        if actual_name.casefold() != expected_name.casefold():
            print(
                f"Warning: {qid} is named {actual_name!r} in JSON, "
                f"but the configured label is {expected_name!r}."
            )

        candidates = build_fact_candidates(battle, qid)
        selected = select_fact_candidates(
            candidates,
            facts_per_battle,
            rng,
        )

        for fact_index, candidate in enumerate(selected, start=1):
            facts.append({
                "id": f"{qid}_fact_{fact_index}",
                "battle_qid": qid,
                "battle_name": actual_name,
                **candidate,
            })

    if missing_files:
        raise FileNotFoundError(
            "The following required JSON records are missing:\n"
            + "\n".join(missing_files)
        )

    expected_total = len(battles) * facts_per_battle
    if len(facts) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} facts, generated {len(facts)}."
        )

    save_json(facts, FACTS_FILE)

    print(f"Generated exactly {len(facts)} facts.")
    print(
        f"Battles: {len(battles)}; "
        f"facts per battle: {facts_per_battle}."
    )
    print(f"Saved to: {FACTS_FILE}")

    return facts


def build_validation_system_message() -> str:
    return (
        "You evaluate whether a factual claim extracted from a structured "
        "battle record is supported by the provided Wikipedia article.\n\n"
        "Use only the provided article text. Do not use external knowledge.\n"
        "Evaluate semantic support: the article may express the same fact "
        "using different wording, aliases, or reasonable paraphrases.\n"
        "Do not treat absence of evidence as a contradiction.\n\n"
        "Choose exactly one verdict:\n"
        "- supported: the article clearly supports the complete claim\n"
        "- contradicted: the article clearly states incompatible information\n"
        "- not_found: the article does not provide enough evidence either way\n\n"
        "For supported or contradicted claims, provide a short verbatim article "
        "excerpt when possible. For not_found, use an empty evidence string.\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "verdict": "supported|contradicted|not_found",\n'
        '  "evidence": "short excerpt or empty string",\n'
        '  "explanation": "one short explanation"\n'
        "}\n"
        "Keep the explanation concise and stop after the closing JSON brace."
    )


def parse_json_response(answer: str) -> dict[str, Any]:
    """Parse a judge response with optional Markdown or surrounding text."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The validation model returned an empty answer.")

    cleaned = answer.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError(
                "The validation response does not contain a JSON object."
            )

        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The validation response must be a JSON object.")

    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict {verdict!r}; expected one of "
            f"{sorted(VALID_VERDICTS)}."
        )

    evidence = parsed.get("evidence", "")
    explanation = parsed.get("explanation", "")

    return {
        "verdict": verdict,
        "evidence": evidence if isinstance(evidence, str) else str(evidence),
        "explanation": (
            explanation
            if isinstance(explanation, str)
            else str(explanation)
        ),
    }


def validate_fact(
    llm,
    fact: dict[str, Any],
    max_attempts: int = 2,
) -> dict[str, Any]:
    raw_text_path = RAW_TEXT_DIR / f"{fact['battle_qid']}.txt"

    if not raw_text_path.exists():
        raise FileNotFoundError(
            f"Raw Wikipedia text not found: {raw_text_path}"
        )

    raw_text = raw_text_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Raw Wikipedia text is empty: {raw_text_path}")

    prompt = (
        f"Battle QID: {fact['battle_qid']}\n"
        f"Battle name: {fact['battle_name']}\n"
        f"JSON field: {fact['field_path']}\n"
        f"JSON value: {json.dumps(fact['value'], ensure_ascii=False)}\n"
        f"Claim: {fact['fact']}\n\n"
        f"Wikipedia article text:\n{raw_text}"
    )

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            answer = generate_answer(
                llm=llm,
                prompt=prompt,
                system_message=build_validation_system_message(),
                max_new_tokens=VALIDATION_MAX_NEW_TOKENS,
            )

            validation = parse_json_response(answer)
            validation["judge_response"] = answer
            return validation

        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                print(
                    f"  Attempt {attempt} failed for {fact['id']}: "
                    f"{error}. Retrying..."
                )

    raise RuntimeError(
        f"Validation failed for {fact['id']} after {max_attempts} "
        f"attempts: {last_error}"
    )


def build_validation_report(
    facts: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    verdict_counts = Counter(
        result.get("verdict")
        for result in results
        if result.get("verdict") in VALID_VERDICTS
    )

    evaluated_count = sum(verdict_counts.values())
    supported_count = verdict_counts["supported"]
    error_count = sum(
        result.get("verdict") == "evaluation_error"
        for result in results
    )

    source_support_rate = (
        supported_count / evaluated_count * 100
        if evaluated_count > 0
        else None
    )

    category_summary: dict[str, dict[str, Any]] = {}
    for category in PREFERRED_CATEGORIES:
        category_results = [
            result
            for result in results
            if result.get("category") == category
            and result.get("verdict") in VALID_VERDICTS
        ]
        category_supported = sum(
            result["verdict"] == "supported"
            for result in category_results
        )
        category_total = len(category_results)

        category_summary[category] = {
            "supported": category_supported,
            "evaluated": category_total,
            "source_support_rate": (
                category_supported / category_total * 100
                if category_total > 0
                else None
            ),
        }

    return {
        "experiment": "json_factual_correctness",
        "reference_source": "English Wikipedia raw article text",
        "selection": {
            "num_battles": len(BATTLES),
            "facts_per_battle": FACTS_PER_BATTLE,
            "expected_fact_count": len(BATTLES) * FACTS_PER_BATTLE,
            "generated_fact_count": len(facts),
            "random_seed": RANDOM_SEED,
            "battles": BATTLES,
        },
        "Final_results": {
            "total_facts": len(facts),
            "processed_facts": len(results),
            "evaluated_facts": evaluated_count,
            "evaluation_error_count": error_count,
            "supported": supported_count,
            "contradicted": verdict_counts["contradicted"],
            "not_found": verdict_counts["not_found"],
            "source_support_rate": source_support_rate,
            "category_results": category_summary,
        },
        "facts": results,
    }


def validate_facts(
    facts_file: str | Path = FACTS_FILE,
    validation_file: str | Path = VALIDATION_FILE,
    resume: bool = True,
) -> dict[str, Any]:
    facts = load_json(facts_file)

    if not isinstance(facts, list):
        raise ValueError("The facts file must contain a JSON list.")

    expected_total = len(BATTLES) * FACTS_PER_BATTLE
    if len(facts) != expected_total:
        raise ValueError(
            f"Expected {expected_total} facts, found {len(facts)} in "
            f"{facts_file}. Run generate_fact_sample() again."
        )

    validation_file = Path(validation_file)
    results_by_id: dict[str, dict[str, Any]] = {}

    if resume and validation_file.exists():
        previous = load_json(validation_file)
        previous_results = (
            previous.get("facts", [])
            if isinstance(previous, dict)
            else previous
        )

        if isinstance(previous_results, list):
            results_by_id = {
                result["id"]: result
                for result in previous_results
                if isinstance(result, dict)
                and isinstance(result.get("id"), str)
                and result.get("verdict") in VALID_VERDICTS
            }

    llm = load_llm()

    for index, fact in enumerate(facts, start=1):
        fact_id = fact["id"]

        if fact_id in results_by_id:
            print(f"[{index}/{len(facts)}] Skipping {fact_id}: already done.")
            continue

        print(
            f"[{index}/{len(facts)}] Validating {fact_id} "
            f"({fact['battle_name']}, {fact['field_path']})..."
        )

        try:
            validation = validate_fact(llm, fact)
            result = {**fact, **validation}
        except Exception as error:
            result = {
                **fact,
                "verdict": "evaluation_error",
                "evidence": "",
                "explanation": "",
                "evaluation_error": str(error),
            }
            print(f"  Evaluation error: {error}")

        results_by_id[fact_id] = result

        ordered_results = [
            results_by_id[item["id"]]
            for item in facts
            if item["id"] in results_by_id
        ]
        report = build_validation_report(facts, ordered_results)
        save_json(report, validation_file)

    final_results = [
        results_by_id[fact["id"]]
        for fact in facts
        if fact["id"] in results_by_id
    ]
    final_report = build_validation_report(facts, final_results)
    save_json(final_report, validation_file)

    summary = final_report["Final_results"]

    print("\n" + "=" * 80)
    print("JSON FACTUAL-CORRECTNESS VALIDATION FINISHED")
    print("=" * 80)
    print(f"Total facts: {summary['total_facts']}")
    print(f"Evaluated facts: {summary['evaluated_facts']}")
    print(f"Supported: {summary['supported']}")
    print(f"Contradicted: {summary['contradicted']}")
    print(f"Not found: {summary['not_found']}")
    print(f"Evaluation errors: {summary['evaluation_error_count']}")

    if summary["source_support_rate"] is not None:
        print(
            "Source support rate: "
            f"{summary['source_support_rate']:.2f}%"
        )

    print(f"Report saved to: {validation_file}")

    return final_report


def run_factual_correctness_experiment() -> dict[str, Any]:
    """Generate all 80 facts and validate them against Wikipedia text."""
    generate_fact_sample()
    return validate_facts(resume=False)


if __name__ == "__main__":
    run_factual_correctness_experiment()
