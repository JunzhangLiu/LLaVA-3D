"""
Score a model_scanqa.py output file against ScanQA ground truth.

Usage (run from LLaVA-3D/):
    python llava/eval/score_scanqa.py results/demo_kmeans512.json
    python llava/eval/score_scanqa.py results/scanqa_baseline.json --gt-file playground/data/annotations/ScanQA_v1.0_val.json

Output:
    File:         results/demo_kmeans512.json
    Scored:       84 questions  (skipped 0 corrupt lines)
    Raw EM@1:     34.52%  (29/84)
    Refined EM@1: 48.81%  (41/84)

Metrics:
    Raw EM@1     — exact match after lowercasing, stripping punctuation, and
                   normalising digits to words (e.g. "1" -> "one")
    Refined EM@1 — also counts if the prediction is a substring of any GT
                   answer, or vice versa (catches "wooden chair" vs "chair")
"""
import argparse
import json
import re
import sys


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def clean(s):
    s = s.lower()
    # Strip all punctuation except apostrophes, commas, hyphens, colons
    s = re.sub(r'[^a-zA-Z0-9,\'\s\-:]+', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Digit -> word (0-10 only, covers most ScanQA answers)
    for digit, word in [
        ('0', 'zero'), ('1', 'one'), ('2', 'two'), ('3', 'three'),
        ('4', 'four'), ('5', 'five'), ('6', 'six'), ('7', 'seven'),
        ('8', 'eight'), ('9', 'nine'), ('10', 'ten'),
    ]:
        s = re.sub(r'\b' + digit + r'\b', word, s)
    # Drop leading articles before a single word ("the chair" -> "chair")
    for article in ['the', 'a', 'an']:
        s = re.sub(r'\b' + article + r'\b ([a-zA-Z]+)', r'\g<1>', s)
    return s


# ---------------------------------------------------------------------------
# Matching: returns (raw_hit, refined_hit)
# ---------------------------------------------------------------------------

def match(pred, gts):
    # Raw: prediction must equal one of the GT answers exactly
    if pred in gts:
        return 1, 1
    # Refined: prediction is a substring of a GT answer, or vice versa
    for gt in gts:
        pred_nospace = ''.join(pred.split())
        gt_nospace   = ''.join(gt.split())
        if pred_nospace in gt_nospace or gt_nospace in pred_nospace:
            return 0, 1
    return 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Score a model_scanqa.py output against ScanQA ground truth.'
    )
    parser.add_argument(
        'answers_file',
        help='Path to the JSONL answers file produced by model_scanqa.py',
    )
    parser.add_argument(
        '--gt-file',
        default='playground/data/annotations/ScanQA_v1.0_val.json',
        help='Path to the ScanQA ground-truth JSON (default: playground/data/annotations/ScanQA_v1.0_val.json)',
    )
    args = parser.parse_args()

    # Load ground truth: question_id -> list of accepted answers
    gt_map = {q['question_id']: q['answers'] for q in json.load(open(args.gt_file))}

    # Load predictions (one JSON object per line; skip corrupt lines)
    preds, skipped = [], 0
    for line in open(args.answers_file):
        try:
            preds.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1

    # Match predictions to ground truth (ignore questions with no GT entry)
    matched = [(p, gt_map[p['question_id']]) for p in preds if p['question_id'] in gt_map]

    # Score
    raw_hits, refined_hits = 0, 0
    for pred, answers in matched:
        r, f = match(clean(pred['text']), [clean(a) for a in answers])
        raw_hits     += r
        refined_hits += f

    n = len(matched)
    print(f'File:         {args.answers_file}')
    print(f'Scored:       {n} questions  (skipped {skipped} corrupt lines)')
    print(f'Raw EM@1:     {raw_hits / n * 100:.2f}%  ({raw_hits}/{n})')
    print(f'Refined EM@1: {refined_hits / n * 100:.2f}%  ({refined_hits}/{n})')


if __name__ == '__main__':
    main()
