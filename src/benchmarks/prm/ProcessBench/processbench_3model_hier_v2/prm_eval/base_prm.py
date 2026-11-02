from abc import ABC, abstractmethod
from typing import Dict, List, Sequence


class BasePRM(ABC):
    """
    Base class for ProcessBench PRM evaluation.

    Subclasses should return one NON_ERROR probability for every
    reasoning step.

    Convention:
        score >= threshold -> NON_ERROR / correct step
        score <  threshold -> ERROR / incorrect step

    ProcessBench prediction:
        first incorrect step index
        -1 if no incorrect step is detected
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        threshold: float = 0.5,
        max_length: int = 4096,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_length = max_length

        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = self._load_tokenizer()

        print(f"Loading model: {self.model_name}")
        self.model = self._load_model()

        self.model.eval()

        self._post_load()

        print("Model loaded successfully.")

    @abstractmethod
    def _load_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    def _post_load(self):
        pass

    @abstractmethod
    def predict_batch(
        self,
        examples: Sequence[Dict],
    ) -> List[Dict]:
        """
        Input:
            [
                {
                    "problem": "...",
                    "steps": ["step1", "step2", ...]
                },
                ...
            ]

        Return:
            [
                {
                    "step_scores": [...],
                    "step_predictions": [...],
                    "prediction": first_error_index
                },
                ...
            ]
        """
        raise NotImplementedError

    def scores_to_prediction(
        self,
        step_scores: List[float],
    ):
        """
        Convert NON_ERROR probabilities into ProcessBench prediction.
        """

        step_predictions = [
            1 if score >= self.threshold else 0
            for score in step_scores
        ]

        prediction = -1

        for index, label in enumerate(step_predictions):
            if label == 0:
                prediction = index
                break

        return step_predictions, prediction