import torch

from mammo_benchmark.evaluation.core import EvalBuffer, aggregate_exams, record_batch


def test_exam_aggregation_uses_max_probability_and_max_label():
    buffer = EvalBuffer()
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 1.0, 6.0],
            [3.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2, 0])
    record_batch(buffer, "Bi-Rads", logits, labels, ["exam_a", "exam_a", "exam_b"])

    exam_logits, exam_labels, exam_ids, num_images = aggregate_exams(buffer)
    exam_probs = torch.softmax(exam_logits, dim=1)

    assert exam_ids == ["exam_a", "exam_b"]
    assert num_images == [2, 1]
    assert exam_labels.tolist() == [2, 0]
    assert int(exam_probs[0].argmax()) == 2
