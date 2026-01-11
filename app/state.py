from typing import TypedDict, List, Dict, Annotated
import operator
from langchain_core.messages import BaseMessage

class DeepAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]
    current_step_index: int
    scratchpad: Dict
    task_complete: bool
    user_id: str
    next_worker: str
