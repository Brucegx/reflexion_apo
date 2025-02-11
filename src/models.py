from pydantic import BaseModel
from typing import List, Dict, Optional

class PromptComponents(BaseModel):
    role: Optional[str] = None
    context: Optional[str] = None
    sop_text: Optional[str] = None
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    quality_metrics: List[str] = []

class PromptOptimizationState(BaseModel):
    initial_prompt: str
    extracted_components: PromptComponents = PromptComponents()
    optimized_prompt: Optional[str] = None
    validation_errors: List[str] = [] 