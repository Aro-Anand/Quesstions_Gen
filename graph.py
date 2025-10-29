# graph.py
"""
LangGraph workflow definition for the question paper generation pipeline.
"""
import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from agents import (
    WorkflowState,
    create_generator_agent,
    create_validator_agent,
    format_output_node
)
from utils import query_syllabus_context

logger = logging.getLogger(__name__)


def create_workflow_graph(pinecone_index):
    """
    Create the LangGraph workflow for question generation.
    
    Args:
        pinecone_index: Pinecone index for syllabus context retrieval
        
    Returns:
        Compiled StateGraph
    """
    
    # Initialize agents
    generator = create_generator_agent()
    validator = create_validator_agent()
    
    # Create state graph
    workflow = StateGraph(WorkflowState)
    
    def retrieve_context(state: WorkflowState) -> WorkflowState:
        """Retrieve relevant syllabus context from Pinecone."""
        try:
            inputs = state["user_inputs"]
            query_text = f"{inputs['topic']} {inputs['chapter']}"
            
            filters = {
                "class": inputs["class"],
                "subject": inputs["subject"]
            }
            
            contexts = query_syllabus_context(
                pinecone_index,
                query_text,
                filters=filters,
                top_k=3
            )
            
            state["context_snippets"] = [ctx["text"] for ctx in contexts]
            logger.info(f"Retrieved {len(contexts)} context snippets")
            
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
            state["context_snippets"] = []
        
        return state
    
    def should_retry(state: WorkflowState) -> Literal["generator", "end"]:
        """Determine if questions need regeneration."""
        validated = state.get("validated_questions", [])
        raw = state.get("raw_generated_questions", [])
        retry_count = state.get("retry_count", 0)
        
        if not raw:
            return "end"
        
        pass_rate = len(validated) / len(raw) if raw else 0
        
        # Retry if less than 50% pass and haven't retried yet
        if pass_rate < 0.5 and retry_count == 0:
            logger.info(f"Pass rate {pass_rate:.1%} < 50%, retrying generation...")
            state["retry_count"] = 1
            return "generator"
        
        return "end"
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generator", generator)
    workflow.add_node("validator", validator)
    workflow.add_node("format_output", format_output_node)
    
    # Define edges
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generator")
    workflow.add_edge("generator", "validator")
    
    # Conditional edge for retry logic
    workflow.add_conditional_edges(
        "validator",
        should_retry,
        {
            "generator": "generator",
            "end": "format_output"
        }
    )
    
    workflow.add_edge("format_output", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app


def test_workflow():
    """Test the workflow with sample inputs."""
    from utils import initialize_pinecone, setup_syllabus_index, upsert_dummy_data
    
    print("Initializing Pinecone...")
    pc = initialize_pinecone()
    index = setup_syllabus_index(pc)
    
    print("Setting up dummy data...")
    upsert_dummy_data(index)
    
    print("Creating workflow...")
    app = create_workflow_graph(index)
    
    print("Testing workflow...")
    initial_state: WorkflowState = {
        "user_inputs": {
            "class": "Class 10",
            "subject": "Math",
            "chapter": "Algebra",
            "topic": "Quadratic Equations",
            "num_questions": 3,
            "difficulty": 3,
            "question_type": "Objective",
            "choice_type": "Single Choice"
        },
        "raw_generated_questions": [],
        "validated_questions": [],
        "output": "",
        "output_latex": "",
        "retry_count": 0,
        "context_snippets": []
    }
    
    print("Running workflow...")
    result = app.invoke(initial_state)
    
    print("\n" + "="*80)
    print("WORKFLOW RESULT")
    print("="*80)
    print(f"\nGenerated Questions: {len(result['raw_generated_questions'])}")
    print(f"Validated Questions: {len(result['validated_questions'])}")
    print(f"Retry Count: {result['retry_count']}")
    
    print("\n" + "="*80)
    print("OUTPUT (Plain Text)")
    print("="*80)
    print(result['output'])
    
    print("\n" + "="*80)
    print("OUTPUT (LaTeX)")
    print("="*80)
    print(result['output_latex'][:500] + "...")
    
    return result


if __name__ == "__main__":
    test_workflow()