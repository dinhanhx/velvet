import re
from typing import Any, Tuple, Union


class InstructionTemplate:
    """Instruction Template

    language-related parameters must follow ISO 639-1
    """

    @staticmethod
    def make_instruction_response_pair(*args: Any, **kwds: Any) -> Tuple[str, str]:
        return "", ""

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        return re.Match()


class VisualQuestionAnswerTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(
        question: str, answer: str, language: str
    ) -> Tuple[str, str]:
        return (
            f"Generate answer in {language}: {question}",
            f"{answer}",
        )

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Generate answer in ([a-zA-Z]{2}):"
        return re.match(pattern, text)


class ReverseVisualQuestionAnswerTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(
        question: str, answer: str, language: str
    ) -> Tuple[str, str]:
        return (
            f"Generate question in {language}: {answer}",
            f"{question}",
        )

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Generate question in ([a-zA-Z]{2}):"
        return re.match(pattern, text)


class ImageCaptionTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(caption: str, language: str) -> Tuple[str, str]:
        return f"Generate caption in {language}:", f"{caption}"

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Generate caption in ([a-zA-Z]{2}):"
        return re.match(pattern, text)


class CompleteImageCaptionTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(
        head_caption: str, tail_caption: str, language: str
    ) -> Tuple[str, str]:
        return (
            f"Complete caption in {language}: {head_caption}",
            f"{tail_caption}",
        )

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Complete caption in ([a-zA-Z]{2}):"
        return re.match(pattern, text)


class TranslateTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(
        source_text: str, source_language: str, target_text: str, target_language: str
    ) -> Tuple[str, str]:
        return (
            f"Translate to {target_language}: {source_language}",
            f"{target_text}",
        )

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Translate to ([a-zA-Z]{2}):"
        return re.match(pattern, text)


class ImageTextMatchTemplate(InstructionTemplate):
    @staticmethod
    def make_instruction_response_pair(
        caption: str, language: str, label: bool
    ) -> Tuple[str, str]:
        return f"Is this a pair in {language}: {caption}", f"{label}"

    @staticmethod
    def regex_match(text: str) -> Union[re.Match, None]:
        pattern = r"^Is this a pair in ([a-zA-Z]{2}):"
        return re.match(pattern, text)
