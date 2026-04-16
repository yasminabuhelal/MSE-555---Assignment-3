"""
q1.py  –  Prompt Engineering: Extracting Per-Session Progress Scores

Pipeline overview
-----------------
    labeled_notes.json   ──► score ──► compute_metrics() ──► print results  (Q1a: validate prompt)
    unlabeled_notes.json ──► score ──► save                                  (Q1b: score at scale)

The LLM's job
-------------
For each client, the model receives the full sequence of session notes and
must return one progress score per consecutive note pair:

    notes 1→2 : score
    notes 2→3 : score
    ...
    notes 11→12 : score

Scores are integers 1–4, returned as a JSON list, e.g. [3, 2, 1, 2, ...].

What is already done for you
------------------------------
- Parsing and validating the LLM's JSON response
- Retrying once automatically if the response is malformed
- Looping over every client in a dataset
- Aligning true vs. predicted scores into a flat list of (true, predicted) pairs
- Building and printing the confusion matrix
- Saving all outputs to JSON

Your tasks  (search for # TODO to find each one)
--------------------------------------------------
1. build_prompt()      Write the prompt that instructs the LLM.
2. call_llm()          Wire up your chosen LLM API (OpenAI, Gemini, Anthropic, etc.).
3. compute_metrics()   Define and compute the performance metric(s) you will use
                       to evaluate and compare prompt versions.

Expected inputs:
    data/labeled_notes.json     – hand-scored by Patel; use this to test your prompt
    data/unlabeled_notes.json   – apply your validated prompt here

Expected outputs:
    output/evaluated_labeled_results.json   – scored test set with true labels (Q1a)
    output/scored_notes.json                – scored unlabeled clients (Q1b, feeds Q2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BaseQ1Config:
    client_id_key: str = "client_id"
    notes_key: str = "notes"
    note_number_key: str = "note_number"
    note_text_key: str = "note_text"
    true_vector_key: str = "scored_progress"
    pred_vector_key: str = "estimated_trajectory_vector"

    valid_scores: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class Q1ALabeledConfig(BaseQ1Config):
    test_path: str = "data/test_notes.json"
    evaluated_output_path: str = "outputs/evaluated_labeled_results.json"


@dataclass
class Q1BUnlabeledConfig(BaseQ1Config):
    unlabeled_path: str = "data/unlabeled_notes.json"
    output_path: str = "outputs/scored_notes.json"


# ============================================================================
# DATA LOADING / SAVING
# ============================================================================

def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent folders for an output path and return it as a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a top-level JSON list from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}.")
    return data


def save_json(data: Any, path: str) -> None:
    """Save JSON to disk and create parent folders if needed."""
    output_path = ensure_parent_dir(path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_vector_pair(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> tuple[str, List[int], List[int]]:
    """Pull the client id, true vector, and estimated vector from one scored record."""
    client_id = str(record[config.client_id_key])
    true_vector = record.get(config.true_vector_key, [])
    estimated_vector = record.get(config.pred_vector_key, [])
    return client_id, true_vector, estimated_vector


def build_step_comparisons(
    client_id: str,
    true_vector: List[int],
    estimated_vector: List[int],
) -> List[Dict[str, Any]]:
    """Build one row per compared step between the true and estimated vectors."""
    rows = []
    for step_idx, (true_score, estimated_score) in enumerate(
        zip(true_vector, estimated_vector),
        start=1,
    ):
        rows.append(
            {
                "client_id": client_id,
                "step_number": step_idx,
                "true_score": true_score,
                "estimated_score": estimated_score,
            }
        )
    return rows


def build_client_comparison(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Create the per-client comparison payload used by evaluation code."""
    client_id, true_vector, estimated_vector = get_vector_pair(record, config)
    step_rows = build_step_comparisons(client_id, true_vector, estimated_vector)
    return {
        "client_id": client_id,
        "true_vector": true_vector,
        "estimated_vector": estimated_vector,
        "n_true_scores": len(true_vector),
        "n_estimated_scores": len(estimated_vector),
        "n_compared_scores": len(step_rows),
        "step_comparisons": step_rows,
    }


def build_evaluation_comparisons(
    scored_test_data: List[Dict[str, Any]],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Build client-level and step-level comparison tables for evaluation."""
    client_level_comparisons = []
    step_level_comparisons = []

    for record in scored_test_data:
        client_summary = build_client_comparison(record, config)
        client_level_comparisons.append(client_summary)
        step_level_comparisons.extend(client_summary["step_comparisons"])

    return {
        "n_clients": len(scored_test_data),
        "client_level_comparisons": client_level_comparisons,
        "step_level_comparisons": step_level_comparisons,
    }


def build_confusion_matrix(
    step_rows: List[Dict[str, Any]],
    valid_scores: List[int] | tuple[int, ...],
) -> Dict[str, Any]:
    """Build a confusion matrix with row totals, column totals, and a printable table."""
    matrix = {
        true_score: {estimated_score: 0 for estimated_score in valid_scores}
        for true_score in valid_scores
    }

    for row in step_rows:
        true_score = row["true_score"]
        estimated_score = row["estimated_score"]
        if true_score in matrix and estimated_score in matrix[true_score]:
            matrix[true_score][estimated_score] += 1

    row_totals = {
        true_score: sum(
            matrix[true_score][estimated_score] for estimated_score in valid_scores
        )
        for true_score in valid_scores
    }
    column_totals = {
        estimated_score: sum(
            matrix[true_score][estimated_score] for true_score in valid_scores
        )
        for estimated_score in valid_scores
    }
    grand_total = sum(row_totals.values())

    headers = ["true\\pred", *[str(score) for score in valid_scores], "Total"]
    row_label_width = max(
        len(headers[0]),
        len("Total"),
        max(len(str(score)) for score in valid_scores),
    )
    cell_width = max(
        5,
        max(
            len(str(value))
            for value in [
                *[
                    matrix[true_score][estimated_score]
                    for true_score in valid_scores
                    for estimated_score in valid_scores
                ],
                *row_totals.values(),
                *column_totals.values(),
                grand_total,
            ]
        ),
    )

    header_line = " | ".join(
        [headers[0].rjust(row_label_width)]
        + [header.rjust(cell_width) for header in headers[1:]]
    )
    separator_line = "-+-".join(
        ["-" * row_label_width] + ["-" * cell_width for _ in headers[1:]]
    )

    table_lines = [header_line, separator_line]
    for true_score in valid_scores:
        row_values = [
            str(matrix[true_score][estimated_score])
            for estimated_score in valid_scores
        ]
        row_line = " | ".join(
            [str(true_score).rjust(row_label_width)]
            + [value.rjust(cell_width) for value in row_values]
            + [str(row_totals[true_score]).rjust(cell_width)]
        )
        table_lines.append(row_line)

    total_line = " | ".join(
        ["Total".rjust(row_label_width)]
        + [
            str(column_totals[estimated_score]).rjust(cell_width)
            for estimated_score in valid_scores
        ]
        + [str(grand_total).rjust(cell_width)]
    )
    table_lines.append(separator_line)
    table_lines.append(total_line)

    return {
        "labels": list(valid_scores),
        "counts": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "grand_total": grand_total,
        "table": "\n".join(table_lines),
    }


# ============================================================================
# TODO 1 of 3 — PROMPT
# ============================================================================
_SYSTEM_PROMPT = """You are an experienced speech-language pathologist tasked with scoring clinical progress between consecutive therapy sessions for a child with speech and language delays.

SCORING CRITERIA:
- 0: Performance is essentially unchanged — goals, accuracy, and support needs are at the same level as the prior session.
- 1: A noticeable but modest gain — the child is slightly more accurate, needs a bit less prompting, or shows some carryover, but remains within the same overall level.
- 2: A substantial improvement — the child is performing markedly more independently, with much greater consistency, or has started applying skills across new contexts.
- 3: A major developmental shift — the child has moved to a higher level of the goal hierarchy or achieved genuine spontaneous use where significant support was previously required. Reserve for clear breakthroughs only.

DISTINGUISHING ADJACENT SCORES:
- Between 0 and 1: Is there any concrete documented change, or is the clinician describing the same performance? If the latter, score 0.
- Between 1 and 2: Has the child crossed a meaningful threshold, or just improved slightly within where they already were? Threshold crossing warrants a 2.
- Between 2 and 3: Is this a stronger version of existing progress, or a genuine qualitative shift to a new level of functioning? Only the latter earns a 3.

Read the notes sequentially and for each consecutive pair evaluate the child's goals, accuracy, prompting level, and independence. Assign one score per pair.

Return a JSON list of integers only (0–3), one per transition, with no commentary, extra wording, or formatting.
Example for 12 sessions: [0, 1, 2, 0, 1, 3, 1, 0, 1, 2, 0]"""

def build_prompt(notes_json_str: str) -> str:
    """
    Write the prompt that instructs the LLM to score a client's note sequence.

    The pipeline calls this once per client and passes the result directly to
    call_llm().  The notes arrive pre-serialised as a JSON string.

    What the LLM must do
    --------------------
    Read the notes in order and, for every consecutive pair (note N → note N+1),
    assign a progress score from 0 to 3.
    
    For a client with 12 notes, the model must return exactly 11 scores.

    Required output format
    ----------------------
    A JSON list and nothing else — no explanation, no markdown, no extra keys:
        [3, 2, 3, 1, 2, 3, 3, 2, 2, 2, 3]

    Parameters
    ----------
    notes_json_str : str
        The client's full note sequence serialised as a JSON string.
        Each note is a dict with keys "note_number" and "note_text".

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    notes = json.loads(notes_json_str)
    sections = [
        f"Session {note['note_number']}:\n{note['note_text']}"
        for note in notes
    ]
    return "SESSION NOTES:\n\n" + "\n\n".join(sections)


# ============================================================================
# TODO 2 of 3 — LLM CALL
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Send a prompt to your chosen LLM and return the raw response text.

    Parameters
    ----------
    prompt : str
        The string returned by build_prompt().

    Returns
    -------
    str
        The model's raw text response (the pipeline will parse it).

    Instructions
    ------------
    Pick ONE of the three provider examples below, uncomment it, and add
    your API key.  Delete the other two and the raise at the bottom.

    Tips
    ----
    - Set temperature=0.0 so results are deterministic and reproducible.
    - Do not post-process the response here — return it raw.  Parsing and
      validation happen in parse_vector_from_response().
    """
    # ── Option A: OpenAI ────────────────────────────────────────────────────
    # import os
    # from openai import OpenAI
    
    # client = OpenAI(api_key="API_KEY")
    # resp = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.0,
    # )
    # return (resp.choices[0].message.content or "").strip()

    # ── Option B: Anthropic (Claude) ────────────────────────────────────────
    import os
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

    # ── Option C: Google Gemini ──────────────────────────────────────────────
    # import os
    # from google import genai
    #
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # resp = client.models.generate_content(
    #     model="gemini-2.5-pro",
    #     contents=prompt,
    # )
    # return resp.text.strip()


# ============================================================================
# CLIENT-LEVEL SCORING
# ============================================================================

def parse_vector_from_response(
    response_text: str,
    expected_length: int,
    valid_scores: List[int] | tuple[int, ...] = (0, 1, 2, 3),
) -> List[int]:
    """
    Parse the model's response into one full trajectory vector.

    This function checks that:
    - the response is a JSON list
    - every item is an allowed score
    - the list length matches the number of note-to-note transitions

    Example valid response:
    [3, 2, 1]
    """
    try:
        data = json.loads(response_text)
        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        cleaned = []
        for value in data:
            score = int(value)
            if score not in valid_scores:
                raise ValueError(f"Invalid score {score}")
            cleaned.append(score)

        if len(cleaned) != expected_length:
            raise ValueError(
                f"Expected vector length {expected_length}, got {len(cleaned)}"
            )
        return cleaned
    except Exception:
        return []


def get_validated_vector_from_llm(
    prompt: str,
    expected_length: int,
    config: BaseQ1Config,
    client_id: str,
) -> List[int]:
    """
    Call the LLM, validate the returned vector, and retry once if needed.

    If the first response is empty or malformed, this function runs the same
    prompt one more time. If the second response is still invalid, it raises an
    error so the whole program stops instead of continuing with bad outputs.
    """
    if expected_length == 0:
        return []

    for attempt in (1, 2):
        raw_response = call_llm(prompt)
        estimated_vector = parse_vector_from_response(
            raw_response,
            expected_length=expected_length,
            valid_scores=config.valid_scores,
        )
        if estimated_vector:
            return estimated_vector

        if attempt == 1:
            print(
                f"Invalid LLM response for client {client_id}. "
                "Retrying once with the same prompt..."
            )

    raise RuntimeError(
        f"LLM returned an invalid trajectory vector twice for client {client_id}. "
        "Stopping program."
    )


def score_client_record(
    client_record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """
    Score one client's full note sequence.

    What this function does:
    - pulls all notes for one client
    - turns those notes into a JSON string for the prompt
    - calls the LLM once for the whole sequence
    - parses the returned vector of progress scores
    - returns one output record with the estimated vector

    If the input record already has a true scored vector, it is copied into the
    output too so the evaluation step can compare true vs estimated values.
    """
    all_notes = client_record[config.notes_key]
    client_id = str(client_record[config.client_id_key])
    notes_json_str = json.dumps(all_notes, ensure_ascii=False, indent=2)
    expected_length = max(len(all_notes) - 1, 0)

    prompt = build_prompt(notes_json_str)
    estimated_vector = get_validated_vector_from_llm(
        prompt=prompt,
        expected_length=expected_length,
        config=config,
        client_id=client_id,
    )

    scored_record = {
        config.client_id_key: client_record[config.client_id_key],
        config.notes_key: client_record[config.notes_key],
        config.pred_vector_key: estimated_vector,
    }
    if config.true_vector_key in client_record:
        scored_record[config.true_vector_key] = client_record[config.true_vector_key]
    return scored_record


def score_dataset(
    data: List[Dict[str, Any]],
    config: BaseQ1Config,
    progress_desc: str,
) -> List[Dict[str, Any]]:
    """Score every client record in a dataset and return the scored records."""
    scored = []

    for client_record in tqdm(data, desc=progress_desc):
        scored_record = score_client_record(client_record, config)
        scored.append(scored_record)

    return scored


# ============================================================================
# EVALUATION SECTION
# ============================================================================

# ============================================================================
# TODO 3 of 3 — PERFORMANCE METRICS
# ============================================================================

def compute_metrics(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute one or more performance metrics from the step-level comparisons.

    The assignment asks you to choose and justify an evaluation approach that
    is appropriate for this task.  Implement your chosen metric(s) here.

    Parameters
    ----------
    step_rows : List[Dict[str, Any]]
        One dict per scored note transition across all clients.
        Each dict has at minimum:
            "true_score"      – Patel's hand-assigned score (int, 1–4)
            "estimated_score" – your LLM's predicted score  (int, 1–4)
        Example:
            [
              {"client_id": "C_0011", "step_number": 1,
               "true_score": 3, "estimated_score": 2},
              {"client_id": "C_0011", "step_number": 2,
               "true_score": 2, "estimated_score": 2},
              ...
            ]

    Returns
    -------
    Dict[str, Any]
        A dict mapping metric name → value.  Whatever you return here will
        be printed by print_evaluation().  Example shape:
            {"metricA": 0.61, "metricB": 0.88}

    """
    true_scores = [row["true_score"] for row in step_rows]
    pred_scores = [row["estimated_score"] for row in step_rows]

    n = len(true_scores)
    if n == 0:
        return {"accuracy": 0.0, "mae": 0.0}

    pairs = list(zip(true_scores, pred_scores))

    # Fraction of exact matches
    accuracy = sum(t == p for t, p in pairs) / n

    # Average score gap per transition
    mae = sum(abs(t - p) for t, p in pairs) / n

    return {
        "accuracy": round(accuracy, 4),
        "mae": round(mae, 4),
    }


def evaluate_predictions(
    config: Q1ALabeledConfig,
) -> Dict[str, Any]:
    """
    Compare each client's true scored_vector with the predicted
    estimated_trajectory_vector, then compute metrics and the confusion matrix.
    """
    scored_test_data = load_json(config.evaluated_output_path)
    comparisons = build_evaluation_comparisons(scored_test_data, config)
    step_rows = comparisons["step_level_comparisons"]

    metrics = compute_metrics(step_rows)
    confusion_matrix = build_confusion_matrix(step_rows, config.valid_scores)

    return {
        **metrics,
        "confusion_matrix": confusion_matrix,
    }


def print_evaluation(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if key == "confusion_matrix" and isinstance(value, dict):
            print("confusion_matrix:")
            print(value.get("table", ""))
        else:
            print(f"{key}: {value}")


# ============================================================================
# PIPELINES
# ============================================================================

def run_test_pipeline(config: Q1ALabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on labeled test data."""
    test_data = load_json(config.test_path)

    scored_test_data = score_dataset(
        data=test_data,
        config=config,
        progress_desc="Scoring labeled clients",
    )
    save_json(scored_test_data, config.evaluated_output_path)

    results = evaluate_predictions(config)
    print_evaluation(results)

    return scored_test_data


def run_unlabeled_pipeline(config: Q1BUnlabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on unlabeled note data and save scored outputs."""
    unlabeled_data = load_json(config.unlabeled_path)

    scored_unlabeled_data = score_dataset(
        data=unlabeled_data,
        config=config,
        progress_desc="Scoring unlabeled clients",
    )
    save_json(scored_unlabeled_data, config.output_path)

    return scored_unlabeled_data


# ============================================================================
# ENTRY POINT
# ============================================================================
#
# HOW TO WORK THROUGH THIS FILE
# ──────────────────────────────
# There are three functions marked # TODO that you must implement:
#
#   1. build_prompt()      Write the prompt that tells the LLM what to do.
#   2. call_llm()          Wire up your LLM API (uncomment one of the three
#                          provider options and add your API key).
#   3. compute_metrics()   Define the metric(s) you will use to evaluate and
#                          compare prompt versions.
#
# Recommended order:
#   Step 1 — implement build_prompt(), call_llm(), and compute_metrics()
#   Step 2 — run run_test_pipeline(LABELED_CONFIG) to score the labeled set
#             and see your metrics + confusion matrix printed to the terminal
#   Step 3 — iterate on your prompt; re-run Step 2 to compare versions
#   Step 4 — once satisfied, run run_unlabeled_pipeline(UNLABELED_CONFIG)
#             to score all 300 clients → produces scored_notes.json for Q2
#
# TIP: before running at scale, test your prompt on a single client record:
#
#   import json
#   sample = load_json("data/labeled_notes.json")[0]
#   notes_str = json.dumps(sample["notes"], indent=2)
#   print(build_prompt(notes_str))           # inspect the prompt visually
#   print(call_llm(build_prompt(notes_str))) # check the raw model response
# ============================================================================

if __name__ == "__main__":
    LABELED_CONFIG = Q1ALabeledConfig(
        test_path="labeled_notes.json",
        evaluated_output_path="output/q1/evaluated_labeled_results.json",
    )
    UNLABELED_CONFIG = Q1BUnlabeledConfig(
        unlabeled_path="unlabeled_notes.json",
        output_path="output/q1/scored_notes.json",
    )

    # Step 2: validate your prompt on the labeled test set
    #run_test_pipeline(LABELED_CONFIG)

    # Step 4: score all unlabeled clients (only after prompt is validated)
    run_unlabeled_pipeline(UNLABELED_CONFIG)
