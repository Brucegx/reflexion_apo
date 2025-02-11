import streamlit as st
from src.workflows.optimization_workflow import create_optimization_workflow
from src.models import PromptOptimizationState
from src.agents.extraction_agents import RoleExtractor, ContextExtractor, TaskExtractor, InstructionExtractor, InputFormatExtractor, OutputFormatExtractor, FewShotExtractor

# Initialize workflow
optimization_workflow = create_optimization_workflow()

def reset_state():
    st.session_state.clear()
    st.session_state.PromptOptimizationState = PromptOptimizationState()

# UI Configuration
st.set_page_config(page_title="Prompt Optimizer", layout="wide")
st.title("AI Prompt Optimization System")

# Create main tabs
label_free_optimizer_tab = st.tabs(["Label Data Free Optimizer"])[0]

# Label-Free Optimization Tab
with label_free_optimizer_tab:
    with st.container():
        st.header("Label-Free Optimization")
        optimization_input = st.text_area(
            "Paste your raw content here:",
            height=300,
            help="Paste SOPs, existing prompts, or any unstructured text"
        )
        
        # Placeholder for future trigger logic
        if st.button("🚀 Analyze Content", key="label_free_trigger"):
            st.info("Optimization trigger clicked - implementation coming soon!")
            # Future implementation will handle the content analysis here

def process_sop(raw_text: str) -> PromptOptimizationState:
    state = PromptOptimizationState(raw_input=raw_text)
    
    # Ordered extraction pipeline
    extractors = [
        RoleExtractor.extract_role,
        ContextExtractor.extract_context,
        TaskExtractor.extract_task,
        InstructionExtractor.extract_instructions,
        InputFormatExtractor.extract_input_format,
        OutputFormatExtractor.extract_output_format,
        FewShotExtractor.extract_few_shot
    ]
    
    for extractor in extractors:
        state = extractor(state)
        if state.validation_errors:  # Simple error tracking
            break
    
    return state