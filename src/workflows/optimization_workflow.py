from langgraph.graph import StateGraph, END
from src.models import WorkflowState
from src.agents.extraction_agents import (
    RoleExtractor, 
    ContextExtractor,
    TaskExtractor,
    InstructionExtractor,
    InputFormatExtractor,
    OutputFormatExtractor,
    FewShotExtractor
)
from src.agents.assembly_agent import PromptAssembler, PromptValidator

def create_optimization_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Add all extraction nodes
    workflow.add_node("extract_role", RoleExtractor.extract_role)
    workflow.add_node("extract_context", ContextExtractor.extract_context)
    workflow.add_node("extract_task", TaskExtractor.extract_task)
    workflow.add_node("extract_instructions", InstructionExtractor.extract_instructions)
    workflow.add_node("extract_input", InputFormatExtractor.extract_input_format)
    workflow.add_node("extract_output", OutputFormatExtractor.extract_output_format)
    workflow.add_node("extract_examples", FewShotExtractor.extract_few_shot)
    workflow.add_node("assemble_prompt", PromptAssembler.assemble_prompt)
    workflow.add_node("validate_prompt", PromptValidator.validate_prompt)

    # Build sequential workflow
    workflow.set_entry_point("extract_role")
    workflow.add_edge("extract_role", "extract_context")
    workflow.add_edge("extract_context", "extract_task")
    workflow.add_edge("extract_task", "extract_instructions")
    workflow.add_edge("extract_instructions", "extract_input")
    workflow.add_edge("extract_input", "extract_output")
    workflow.add_edge("extract_output", "extract_examples")
    workflow.add_edge("extract_examples", "assemble_prompt")
    workflow.add_edge("assemble_prompt", "validate_prompt")
    workflow.add_edge("validate_prompt", END)

    return workflow.compile() 