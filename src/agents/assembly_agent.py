from src.models import WorkflowState

class PromptAssembler:
    TEMPLATE = """**Role:** {role}
**Context:** {context}
**Task:** {task}
**Instructions:**
{instructions}
**Input Format:** {input_format}
**Output Format:** {output_format}"""

    @staticmethod
    def assemble_prompt(state: WorkflowState) -> WorkflowState:
        components = state.extracted_components
        state.optimized_prompt = PromptAssembler.TEMPLATE.format(
            role=components.role or "Unspecified Role",
            context=components.context or "General Context",
            task=components.task or "Primary Task",
            instructions="\n- ".join(components.instructions),
            input_format=components.input_format or "Free-form text",
            output_format=components.output_format or "Unstructured response"
        )
        return state

class PromptValidator:
    @staticmethod
    def validate_prompt(state: WorkflowState) -> WorkflowState:
        required_fields = ['role', 'task']
        state.validation_errors = [
            f"Missing {field}" for field in required_fields
            if not getattr(state.extracted_components, field)
        ]
        return state 