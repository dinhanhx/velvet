from dataclasses import dataclass
from typing import Tuple


class InstructionTemplate:
    """Instruction Template

    language-related field should follow ISO 639-1
    """
    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return "", ""


@dataclass
class VisualQuestionAnswerTemplate(InstructionTemplate):
    question: str
    answer: str
    language: str

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return (
            f"Generate answer in {self.language}: {self.question}",
            f"{self.answer}",
        )


@dataclass
class ReverseVisualQuestionAnswerTemplate(InstructionTemplate):
    question: str
    answer: str
    language: str

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return (
            f"Generate question in {self.language}: {self.answer}",
            f"{self.question}",
        )


@dataclass
class ImageCaptionTemplate(InstructionTemplate):
    caption: str
    language: str

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return f"Generate caption in {self.language}:", f"{self.caption}"


@dataclass
class CompleteImageCaptionTemplate(InstructionTemplate):
    head_caption: str
    tail_caption: str
    language: str

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return (
            f"Complete caption in {self.language}: {self.head_caption}",
            f"{self.tail_caption}",
        )


@dataclass
class TranslateTemplate(InstructionTemplate):
    source_text: str
    source_language: str
    target_text: str
    target_language: str

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return (
            f"Translate to {self.target_language}: {self.source_language}",
            f"{self.target_text}",
        )


@dataclass
class ImageTextMatchTemplate(InstructionTemplate):
    caption: str
    language: str
    label: bool

    def make_instruction_response_pair(self) -> Tuple[str, str]:
        return f"Is this a pair in {self.language}: {self.caption}", f"{self.label}"
