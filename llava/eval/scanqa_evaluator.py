import json
from tqdm import tqdm

import mmengine
from llava.eval.scanqa_text_utils import answer_match, clean_answer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE")
]


def calc_scanqa_score(preds, gts, tokenizer, scorers):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    print("Total samples:", len(preds))
    assert len(preds) == len(gts)  # number of samples
    for item_id, (pred, gt) in tqdm(enumerate(zip(preds, gts))):
        question_id = pred['question_id']
        gt_question_id = gt['question_id']
        assert question_id == gt_question_id
        pred_answer = pred['text']
        gt_answers = gt['text']
        # if len(pred) > 1:
        #     if pred[-1] == '.':
        #         pred = pred[:-1]
        #     pred = pred[0].lower() + pred[1:]
        pred_answer = clean_answer(pred_answer)
        ref_captions = [clean_answer(gt_answer) for gt_answer in gt_answers]
        tmp_acc, tmp_refined_acc = answer_match(pred_answer, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred_answer}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    acc = acc / len(preds)
    refined_acc = refined_acc / len(preds)
    val_scores["[scanqa] EM1"] = acc
    val_scores["[scanqa] EM1_refined"] = refined_acc
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scanqa] {m}"] = sc
        else:
            val_scores[f"[scanqa] {method}"] = score
    return val_scores

pred_json = 'llava-3d-7b-scanqa_answer_val.json'
preds = [json.loads(q) for q in open(pred_json, "r")]
gt_json = 'playground/data/annotations/llava3d_scanqa_val_answer.json'
gts = mmengine.load(gt_json)


val_scores = calc_scanqa_score(preds, gts, tokenizer, scorers)
print(val_scores)