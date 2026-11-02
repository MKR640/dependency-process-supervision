import json
import os.path as osp
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation, getcontext

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset


getcontext().prec = 50

_SUBJECT_FOLDERS = (
    'AtomicPhysics',
    'ClassicalElectromagnetism',
    'ClassicalMechanics',
    'Electrodynamics',
    'GeometricalOptics',
    'QuantumMechanics',
    'Relativity',
    'SemiconductorPhysics',
    'Solid-StatePhysics',
    'StatisticalMechanics',
    'TheoreticalMechanics',
    'Thermodynamics',
    'WaveOptics',
)
_ANSWER_TYPE_PATTERN = re.compile(
    r'(?<![A-Z])(?:MC|TF|NV|IN|EX|EQ)(?![A-Z])')
_MCQ_LABELS = ('A', 'B', 'C', 'D')
_PAREN_CHOICE_LABEL_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])\(([A-Ea-e])\)\s')
_LINE_CHOICE_LABEL_PATTERN = re.compile(
    r'(?im)(?:^|\n)\\?\s*([A-Ea-e])\)\s')
_TRUE_FALSE_LABELS = ('YES', 'NO')
_NUMERIC_FEW_SHOT_COUNT = 3


def _answer_type_codes(raw_answer_type):
    return tuple(
        _ANSWER_TYPE_PATTERN.findall(str(raw_answer_type).upper()))


def _normalize_problem_text(raw_problem):
    problem_text = str(raw_problem)
    return problem_text.replace(r'\\n', '\n').replace(r'\n', '\n').strip()


def _choice_labels_in_problem(problem):
    labels = {
        match.group(1).upper()
        for match in _PAREN_CHOICE_LABEL_PATTERN.finditer(problem)
    }
    labels.update({
        match.group(1).upper()
        for match in _LINE_CHOICE_LABEL_PATTERN.finditer(problem)
    })
    return tuple(sorted(labels))


def _last_boxed_content(value):
    text = '' if value is None else str(value)
    boxed_index = text.rfind(r'\boxed')
    if boxed_index < 0:
        return text.strip()

    open_brace_index = text.find('{', boxed_index)
    if open_brace_index < 0:
        return ''

    brace_depth = 0
    for character_index in range(open_brace_index, len(text)):
        character = text[character_index]
        if character == '{':
            brace_depth += 1
        elif character == '}':
            brace_depth -= 1
            if brace_depth == 0:
                return text[open_brace_index + 1:character_index].strip()
    return ''


def _normalize_short_answer(raw_answer):
    answer_text = _last_boxed_content(raw_answer)
    answer_text = answer_text.replace(r'\left', '').replace(r'\right', '')
    answer_text = re.sub(
        r'\\(?:mathrm|operatorname|text|textrm|mathbf|mathit)\{([^{}]*)\}',
        r'\1', answer_text, flags=re.IGNORECASE)
    return answer_text.strip().strip('$').strip()


def _normalize_choice_answer(raw_answer):
    answer_text = _normalize_short_answer(raw_answer).strip('[]()').upper()
    if answer_text in _MCQ_LABELS:
        return answer_text
    return None


def _normalize_true_false_answer(raw_answer):
    answer_text = _normalize_short_answer(raw_answer).upper()
    if answer_text in ('TRUE', 'T', 'YES', 'Y'):
        return 'YES'
    if answer_text in ('FALSE', 'F', 'NO', 'N'):
        return 'NO'
    return None


def _clean_numeric_text(value):
    numeric_text = _normalize_short_answer(value)
    numeric_text = re.sub(r'[\x00-\x1F\x7F]', '', numeric_text)
    numeric_text = numeric_text.replace('$', '')
    numeric_text = numeric_text.replace(r'\left', '').replace(r'\right', '')
    numeric_text = numeric_text.replace(r'\,', '')
    numeric_text = numeric_text.replace(' ', '')
    numeric_text = numeric_text.replace(',', '')
    numeric_text = numeric_text.replace('−', '-')
    numeric_text = numeric_text.replace('–', '-').replace('—', '-')
    numeric_text = numeric_text.replace('×', 'x')
    numeric_text = numeric_text.replace(r'\times', 'x')
    numeric_text = numeric_text.replace(r'\cdot', 'x')
    numeric_text = numeric_text.replace('∙', 'x').replace('·', 'x')
    numeric_text = numeric_text.replace(r'\%', '')
    numeric_text = re.sub(r'(?<!\\)frac', r'\\frac', numeric_text)
    numeric_text = re.sub(
        r'\\(?:mathrm|operatorname|text|textrm|mathbf|mathit)\{([^{}]*)\}',
        r'\1', numeric_text, flags=re.IGNORECASE)
    if numeric_text.startswith('(') and numeric_text.endswith(')'):
        numeric_text = numeric_text[1:-1]
    if numeric_text.count('=') == 1:
        left_side, right_side = numeric_text.split('=', 1)
        if left_side and len(left_side) <= 12:
            numeric_text = right_side
    return numeric_text


def _decimal_from_text(value):
    numeric_text = _clean_numeric_text(value)
    if not numeric_text:
        return None

    try:
        if re.fullmatch(r'[+-]?\d+(\.\d+)?', numeric_text):
            return Decimal(numeric_text)
        if re.fullmatch(r'[+-]?\d+(\.\d+)?[eE][+-]?\d+', numeric_text):
            return Decimal(numeric_text)
    except InvalidOperation:
        return None

    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\^?\{([+-]?\d+)\}',
                         numeric_text)
    if match and match.group(1) == '10':
        return Decimal(10) ** int(match.group(2))

    match = re.fullmatch(
        r'([+-]?\d+(?:\.\d+)?)x10\^?\{?([+-]?\d+)\}?', numeric_text)
    if match:
        return Decimal(match.group(1)) * (Decimal(10) ** int(match.group(2)))

    match = re.fullmatch(
        r'([+-]?)\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}',
        numeric_text)
    if match:
        denominator = Decimal(match.group(3))
        if denominator == 0:
            return None
        sign = -1 if match.group(1) == '-' else 1
        return Decimal(sign) * Decimal(match.group(2)) / denominator

    match = re.fullmatch(
        r'([+-]?)\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}'
        r'x10\^?\{?([+-]?\d+)\}?', numeric_text)
    if match:
        denominator = Decimal(match.group(3))
        if denominator == 0:
            return None
        sign = -1 if match.group(1) == '-' else 1
        value = Decimal(sign) * Decimal(match.group(2)) / denominator
        return value * (Decimal(10) ** int(match.group(4)))

    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)',
                         numeric_text)
    if match:
        denominator = Decimal(match.group(2))
        if denominator == 0:
            return None
        return Decimal(match.group(1)) / denominator

    return None


def _numeric_values_match(reference_text, prediction_text):
    reference_value = _decimal_from_text(reference_text)
    prediction_value = _decimal_from_text(prediction_text)
    if reference_value is None or prediction_value is None:
        return False

    quantum = Decimal(1).scaleb(reference_value.as_tuple().exponent)
    try:
        return prediction_value.quantize(quantum) == reference_value
    except InvalidOperation:
        return False


def _iter_english_records(path):
    local_path = get_data_path(path, local_mode=True)
    for subject_folder in _SUBJECT_FOLDERS:
        filename = osp.join(local_path, subject_folder, 'en.jsonl')
        with open(filename, encoding='utf-8') as jsonl_file:
            for line_number, line in enumerate(jsonl_file, 1):
                question_record = json.loads(line)
                if question_record.get('language') != 'EN':
                    raise ValueError(
                        f'UGPhysics {subject_folder}:{line_number} is not EN')
                yield subject_folder, line_number, question_record


def _common_example_fields(subject_folder, line_number, question_record,
                           answer):
    problem = _normalize_problem_text(question_record.get('problem', ''))
    if not problem:
        raise ValueError(
            f'UGPhysics {subject_folder}:{line_number} has empty problem')
    source_index = question_record.get('index', line_number)
    return {
        'id': f'ugphysics-{subject_folder}-{source_index}',
        'problem': problem,
        'answer': answer,
        'answer_type': str(question_record.get('answer_type', '')).strip(),
        'subject': str(question_record.get('subject', '')).strip(),
        'subject_folder': subject_folder,
        'domain': str(question_record.get('domain', '')).strip(),
        'topic': str(question_record.get('topic', '')).strip(),
        'level': str(question_record.get('level', '')).strip(),
        'source_index': source_index,
    }


def _closed_examples(path, target_answer_type):
    examples = []
    for subject_folder, line_number, question_record in _iter_english_records(
            path):
        answer_type_codes = _answer_type_codes(
            question_record.get('answer_type'))
        if answer_type_codes != (target_answer_type, ):
            continue
        if question_record.get('is_multiple_answer'):
            continue

        if target_answer_type == 'MC':
            problem = _normalize_problem_text(question_record.get('problem',
                                                                  ''))
            if _choice_labels_in_problem(problem) != _MCQ_LABELS:
                continue
            answer = _normalize_choice_answer(question_record.get('answers'))
        else:
            answer = _normalize_true_false_answer(
                question_record.get('answers'))
        if answer is None:
            continue

        examples.append(
            _common_example_fields(subject_folder, line_number,
                                   question_record, answer))
    if not examples:
        raise ValueError(f'UGPhysics has no {target_answer_type} examples')
    return examples


def _numeric_examples(path):
    examples = []
    for subject_folder, line_number, question_record in _iter_english_records(
            path):
        answer_type_codes = _answer_type_codes(
            question_record.get('answer_type'))
        if answer_type_codes != ('NV', ):
            continue
        if question_record.get('is_multiple_answer'):
            continue

        answer = _normalize_short_answer(question_record.get('answers'))
        if _decimal_from_text(answer) is None:
            continue

        unit = str(question_record.get('unit', '')).strip()
        example = _common_example_fields(subject_folder, line_number,
                                         question_record, answer)
        example['unit'] = unit
        example['unit_prompt'] = (
            f'Expected unit: {unit}'
            if unit else 'Expected unit: dimensionless or not specified')
        examples.append(example)
    if len(examples) <= _NUMERIC_FEW_SHOT_COUNT:
        raise ValueError('UGPhysics has no numeric test examples after shots')
    return examples


def _classification_metric_values(predictions, references, labels):
    if len(predictions) != len(references):
        return {'error': 'predictions and references length mismatch'}
    if not references:
        return {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'invalid_rate': 0.0,
            'per_label': {},
        }

    normalized_predictions = [
        str(prediction).strip().upper() for prediction in predictions
    ]
    normalized_references = [
        str(reference).strip().upper() for reference in references
    ]
    label_set = set(labels)
    correct_count = sum(
        prediction == reference for prediction, reference in zip(
            normalized_predictions, normalized_references))
    invalid_count = sum(
        prediction not in label_set for prediction in normalized_predictions)

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    per_label_metric_values = {}
    for label in labels:
        true_positive = sum(
            prediction == label and reference == label
            for prediction, reference in zip(normalized_predictions,
                                             normalized_references))
        false_positive = sum(
            prediction == label and reference != label
            for prediction, reference in zip(normalized_predictions,
                                             normalized_references))
        false_negative = sum(
            prediction != label and reference == label
            for prediction, reference in zip(normalized_predictions,
                                             normalized_references))
        support_count = sum(reference == label
                            for reference in normalized_references)
        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1_value = (2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1_value
        per_label_metric_values[label] = {
            'support': support_count,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1_value * 100,
        }

    label_count = len(labels)
    reference_count = len(normalized_references)
    return {
        'accuracy': correct_count / reference_count * 100,
        'macro_precision': precision_sum / label_count * 100,
        'macro_recall': recall_sum / label_count * 100,
        'macro_f1': f1_sum / label_count * 100,
        'invalid_rate': invalid_count / reference_count * 100,
        'per_label': per_label_metric_values,
    }


def _accuracy_by_field(predictions, references, test_set, field_name,
                       answer_matcher):
    field_counts = defaultdict(lambda: [0, 0])
    for prediction, reference, example in zip(predictions, references,
                                              test_set):
        field_value = str(example[field_name]).strip()
        field_counts[field_value][1] += 1
        if answer_matcher(reference, prediction):
            field_counts[field_value][0] += 1

    return {
        field_value: correct_count / total_count * 100
        for field_value, (correct_count, total_count) in sorted(
            field_counts.items())
    }


@TEXT_POSTPROCESSORS.register_module('ugphysics_boxed')
def ugphysics_boxed_postprocess(text):
    return _last_boxed_content(text)


@LOAD_DATASET.register_module()
class UGPhysicsMultipleChoiceDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        return Dataset.from_list(_closed_examples(path, 'MC'))


@LOAD_DATASET.register_module()
class UGPhysicsTrueFalseDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        return Dataset.from_list(_closed_examples(path, 'TF'))


@LOAD_DATASET.register_module()
class UGPhysicsNumericFewShotDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        examples = _numeric_examples(path)
        return DatasetDict({
            'train':
            Dataset.from_list(examples[:_NUMERIC_FEW_SHOT_COUNT]),
            'test':
            Dataset.from_list(examples[_NUMERIC_FEW_SHOT_COUNT:]),
        })


@ICL_EVALUATORS.register_module()
class UGPhysicsClosedEvaluator(BaseEvaluator):

    def __init__(self, labels):
        self.labels = tuple(str(label).strip().upper() for label in labels)
        super().__init__()

    def score(self, predictions, references, test_set):
        metric_values = _classification_metric_values(predictions, references,
                                                      self.labels)
        if 'error' in metric_values:
            return metric_values

        matcher = lambda reference, prediction: (
            str(reference).strip().upper() == str(prediction).strip().upper())
        metric_values['subject_accuracy'] = _accuracy_by_field(
            predictions, references, test_set, 'subject', matcher)
        metric_values['domain_accuracy'] = _accuracy_by_field(
            predictions, references, test_set, 'domain', matcher)
        metric_values['level_accuracy'] = _accuracy_by_field(
            predictions, references, test_set, 'level', matcher)
        return metric_values


@ICL_EVALUATORS.register_module()
class UGPhysicsNumericEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'predictions and references length mismatch'}
        if not references:
            return {
                'numeric_accuracy': 0.0,
                'invalid_rate': 0.0,
                'subject_accuracy': {},
                'domain_accuracy': {},
                'level_accuracy': {},
            }

        correct_count = 0
        invalid_count = 0
        details = []
        for prediction, reference, example in zip(predictions, references,
                                                  test_set):
            prediction_value = _decimal_from_text(prediction)
            reference_value = _decimal_from_text(reference)
            is_correct = _numeric_values_match(reference, prediction)
            if prediction_value is None or reference_value is None:
                invalid_count += 1
            if is_correct:
                correct_count += 1
            details.append({
                'id': example['id'],
                'prediction': prediction,
                'reference': reference,
                'correct': is_correct,
                'subject': example['subject'],
                'domain': example['domain'],
                'level': example['level'],
            })

        reference_count = len(references)
        return {
            'numeric_accuracy':
            correct_count / reference_count * 100,
            'invalid_rate':
            invalid_count / reference_count * 100,
            'subject_accuracy':
            _accuracy_by_field(predictions, references, test_set, 'subject',
                               _numeric_values_match),
            'domain_accuracy':
            _accuracy_by_field(predictions, references, test_set, 'domain',
                               _numeric_values_match),
            'level_accuracy':
            _accuracy_by_field(predictions, references, test_set, 'level',
                               _numeric_values_match),
            'details':
            details,
        }
