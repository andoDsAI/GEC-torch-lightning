from typing import Dict, List

import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

rouge = evaluate.load("rouge")
wer = evaluate.load("wer")
bleu = evaluate.load("bleu")


def compute_metrics(
    predictions: List[str], references: List[str], type_compute: str = ""
) -> Dict[str, float]:
    """Compute the WER between the predictions and the targets.

    :param predictions: The predictions
    :param references: The targets
    :param type_compute: The type of compute
    :return: The metrics
    """
    results = {
        f"{type_compute}_wer": wer.compute(predictions=predictions, references=references),
        f"{type_compute}_rouge": rouge.compute(predictions=predictions, references=references),
    }

    # calculate BLEU score
    # bleu_score = np.mean(
    #     [
    #         sentence_bleu([prediction.split()], reference.split())
    #         for prediction, reference in zip(predictions, references)
    #     ]
    # )

    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )["bleu"]

    results[f"{type_compute}_bleu"] = bleu_score
    return results
