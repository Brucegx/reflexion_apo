import re
import json
from src.models import WorkflowState, PromptComponents
from src.llm.llm_repository import LLMRepository
from pathlib import Path

class RoleExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "role_extraction_prompt.txt"

    @classmethod
    def _build_role_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                template = f.read()
            return template.replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Role prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_role(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_role_prompt(state.initial_prompt)
            llm_config = {
                "max_tokens": 75,  # Increased for better extraction
                "temperature": 0.05,
                "model_id": "anthropic.claude-v2"
            }
            result = LLMRepository.execute_prompt(prompt, llm_config)
            
            if result != 'N/A':
                state.extracted_components.role = result
                state.extracted_components.sop_text = state.initial_prompt
        except RuntimeError:
            role_pattern = r"(?i)\b(you are|acting as|we need a|role)\s*:\s*([a-z\s]+)(?=\b|\.|,)"
            match = re.search(role_pattern, state.initial_prompt)
            if match:
                state.extracted_components.role = match.group(2).strip()
        
        return state

class ContextExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "context_extraction_prompt.txt"

    @classmethod
    def _build_context_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Context prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_context(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_context_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 150,
                "temperature": 0.1,
                "model_id": "anthropic.claude-v2"
            })
            if result != 'N/A':
                state.extracted_components.context = result
        except RuntimeError:
            # Fallback to keyword matching
            keywords = ["context", "background", "environment"]
            sentences = [s.strip() for s in state.initial_prompt.split('.')]
            context_sentences = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
            if context_sentences:
                state.extracted_components.context = ". ".join(context_sentences)
        return state

class TaskExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "task_extraction_prompt.txt"

    @classmethod
    def _build_task_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Task prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_task(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_task_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 100,
                "temperature": 0.1,
                "model_id": "anthropic.claude-v2"
            })
            if result != 'N/A':
                # Clean up any prefix like "Task:"
                state.extracted_components.task = re.sub(r"^Task:\s*", "", result)
        except RuntimeError:
            task_keywords = ["task", "objective", "goal"]
            task_lines = [line.strip() for line in state.initial_prompt.split('\n') 
                        if any(kw in line.lower() for kw in task_keywords)]
            if task_lines:
                state.extracted_components.task = ". ".join(task_lines)
        return state

class InstructionExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "instruction_extraction_prompt.txt"

    @classmethod
    def _build_instruction_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Instruction prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_instructions(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_instruction_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 200,
                "temperature": 0.1,
                "model_id": "anthropic.claude-v2"
            })
            
            if result != 'N/A':
                # Extract numbered/bulleted items
                instructions = re.findall(r"\d+\.\s+(.+)|[\-\•]\s+(.+)", result)
                state.extracted_components.instructions = [max(i, default='').strip() for i in instructions]
        except RuntimeError:
            instruction_pattern = r"(?i)(\d+\.|•|\-)\s*(.+?)(?=\n\d+\.|•|\-|$)"
            matches = re.findall(instruction_pattern, state.initial_prompt)
            if matches:
                state.extracted_components.instructions = [m[1].strip() for m in matches]
        return state

class InputFormatExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "input_format_prompt.txt"

    @classmethod
    def _build_input_format_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Input format prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_input_format(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_input_format_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 100,
                "temperature": 0.05,
                "model_id": "anthropic.claude-v2"
            })
            if result != 'N/A':
                state.extracted_components.input_format = result
        except RuntimeError:
            input_pattern = r"(?i)input format:\s*(.+?)(?=\n\n|$)"
            match = re.search(input_pattern, state.initial_prompt)
            if match:
                state.extracted_components.input_format = match.group(1).strip()
        return state

class OutputFormatExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "output_format_prompt.txt"

    @classmethod
    def _build_output_format_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Output format prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_output_format(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_output_format_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 100,
                "temperature": 0.05,
                "model_id": "anthropic.claude-v2"
            })
            if result != 'N/A':
                state.extracted_components.output_format = result
        except RuntimeError:
            output_pattern = r"(?i)output format:\s*(.+?)(?=\n\n|$)"
            match = re.search(output_pattern, state.initial_prompt)
            if match:
                state.extracted_components.output_format = match.group(1).strip()
        return state

class FewShotExtractor:
    _PROMPT_PATH = Path(__file__).parent.parent / "prompt_store" / "fewshot_extraction_prompt.txt"

    @classmethod
    def _build_fewshot_prompt(cls, sop_text: str) -> str:
        try:
            with open(cls._PROMPT_PATH, "r") as f:
                return f.read().replace("{{sop_text}}", sop_text)
        except FileNotFoundError:
            raise RuntimeError(f"Fewshot prompt template not found at {cls._PROMPT_PATH}")

    @classmethod
    def extract_few_shot(cls, state: WorkflowState) -> WorkflowState:
        try:
            prompt = cls._build_fewshot_prompt(state.initial_prompt)
            result = LLMRepository.execute_prompt(prompt, {
                "max_tokens": 300,
                "temperature": 0.1,
                "model_id": "anthropic.claude-v2"
            })
            
            if result != 'N/A' and result.strip().startswith("["):
                try:
                    examples = json.loads(result)
                    if isinstance(examples, list):
                        state.extracted_components.few_shot_examples = [
                            ex for ex in examples 
                            if safe_eval_check(json.dumps(ex))
                        ]
                except json.JSONDecodeError:
                    pass
        except RuntimeError:
            example_pattern = r"Example:\s*(\{.*?\})"
            matches = re.findall(example_pattern, state.initial_prompt, re.DOTALL)
            if matches:
                state.extracted_components.few_shot_examples = [
                    eval(m) for m in matches if safe_eval_check(m)
                ]
        return state

def safe_eval_check(example_str: str) -> bool:
    """Enhanced safety check for example parsing"""
    return all(c in example_str for c in ["{", "}"]) and ":" in example_str and not "os." in example_str