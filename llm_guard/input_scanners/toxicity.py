from __future__ import annotations

from enum import Enum
from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import calculate_risk_score, get_logger, split_text_by_sentences

from .base import Scanner

LOGGER = get_logger()

DEFAULT_CHINESE_MODEL = Model(
    path="thu-coai/roberta-base-cold",    # 更改为新模型路径
    revision=None,                         # 如果没有特定版本号可以设为 None
    onnx_path=None,                       # 如果没有 ONNX 版本可以设为 None
    onnx_revision=None,
    pipeline_kwargs={
        "padding": "max_length",
        "top_k": None,
        "function_to_apply": "sigmoid",
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

print()
DEFAULT_MODEL = Model(
    path="unitary/unbiased-toxic-roberta",
    revision="36295dd80b422dc49f40052021430dae76241adc",
    onnx_path="ProtectAI/unbiased-toxic-roberta-onnx",
    onnx_revision="34480fa958f6657ad835c345808475755b6974a7",
    pipeline_kwargs={
        "padding": "max_length",
        "top_k": None,
        "function_to_apply": "sigmoid",
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

_toxic_labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


class MatchType(Enum):
    SENTENCE = "sentence"
    FULL = "full"

    def get_inputs(self, prompt: str) -> list[str]:
        if self == MatchType.SENTENCE:
            return split_text_by_sentences(prompt)

        return [prompt]


class Toxicity(Scanner):
    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.5,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False
    ) -> None:
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        self._threshold = threshold
        self._match_type = match_type
        self._use_onnx = use_onnx
        self._model = model
        self._pipeline = None
        self._is_chinese_model = False

    def _initialize_pipeline(self, text: str):
        """根据是否是中文文本初始化对应的模型pipeline"""
        if self._model is not None:
            model = self._model
        else:
            self._is_chinese_model = is_chinese_text(text)
            model = DEFAULT_CHINESE_MODEL if self._is_chinese_model else DEFAULT_MODEL
            LOGGER.info(f"Using model: {model.path}")

        tf_tokenizer, tf_model = get_tokenizer_and_model_for_classification(
            model=model,
            use_onnx=self._use_onnx,
        )

        self._pipeline = pipeline(
            task="text-classification",
            model=tf_model,
            tokenizer=tf_tokenizer,
            **model.pipeline_kwargs,
        )

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if self._pipeline is None:
            self._initialize_pipeline(prompt)

        inputs = self._match_type.get_inputs(prompt)
        results_all = self._pipeline(inputs) # type: ignore

        if self._is_chinese_model:
            # 中文模型处理逻辑
            highest_toxicity_score = 0.0
            for results_chunk in results_all:
                # 获取预测的标签和得分
                scores = {r["label"]: r["score"] for r in results_chunk}
                # 如果 LABEL_0 的概率更大，说明是安全的
                if scores["LABEL_0"] > scores["LABEL_1"]:
                    toxicity_score = 0.0
                else:
                    toxicity_score = scores["LABEL_1"]

                highest_toxicity_score = max(highest_toxicity_score, toxicity_score)

            is_safe = highest_toxicity_score <= self._threshold
            if not is_safe:
                LOGGER.warning("Detected offensive content in Chinese text", score=highest_toxicity_score)
            else:
                LOGGER.debug("Text is safe", score=highest_toxicity_score)

            return prompt, is_safe, calculate_risk_score(highest_toxicity_score, self._threshold)
        else:
            # 英文模型的处理逻辑（多标签）
            highest_toxicity_score = 0.0
            toxicity_above_threshold = []
            for results_chunk in results_all:
                for result in results_chunk:
                    if result["label"] not in _toxic_labels:
                        continue

                    if result["score"] > self._threshold:
                        toxicity_above_threshold.append(result)

                    if result["score"] > highest_toxicity_score:
                        highest_toxicity_score = result["score"]

            if len(toxicity_above_threshold) > 0:
                LOGGER.warning("Detected toxicity in the English text", results=toxicity_above_threshold)
                return prompt, False, calculate_risk_score(highest_toxicity_score, self._threshold)

            LOGGER.debug("No toxicity found in the English text", results=results_all)
            return prompt, True, calculate_risk_score(highest_toxicity_score, self._threshold)

def is_chinese_text(text: str) -> bool:
    """
    检测文本语言，返回 'zh' 或 'en'
    """
    # 统计中文字符数量
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

    # 如果包含中文字符，则认为是中文文本
    # 可以根据需要调整阈值，比如要求中文字符占比超过某个值
    return chinese_chars > 0
