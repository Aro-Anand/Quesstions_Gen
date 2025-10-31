# graph.py
"""
LangGraph workflow definition with timing measurements.
"""
import logging
import time
from typing import Literal
from langgraph.graph import StateGraph, END
from agents import (
    WorkflowState,
    create_generator_agent,
    create_validator_agent,
    format_output_node
)
from utils import query_syllabus_context
from timing_decorator import TimingStats, time_stage

logger = logging.getLogger(__name__)


def create_workflow_graph(pinecone_index):
    """
    Create the LangGraph workflow for question generation with timing.
    
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
        """Retrieve relevant syllabus context from Pinecone with timing."""
        timing_stats = state.get("timing_stats") or TimingStats()
        
        with time_stage(timing_stats, "Context Retrieval"):
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
        
        state["timing_stats"] = timing_stats
        return state
    
    def timed_generator(state: WorkflowState) -> WorkflowState:
        """Generator with timing."""
        timing_stats = state.get("timing_stats") or TimingStats()
        
        with time_stage(timing_stats, "Question Generation"):
            state = generator(state)
        
        state["timing_stats"] = timing_stats
        return state
    
    def timed_validator(state: WorkflowState) -> WorkflowState:
        """Validator with timing."""
        timing_stats = state.get("timing_stats") or TimingStats()
        
        with time_stage(timing_stats, "Question Validation"):
            state = validator(state)
        
        state["timing_stats"] = timing_stats
        return state
    
    def timed_format_output(state: WorkflowState) -> WorkflowState:
        """Format output with timing."""
        timing_stats = state.get("timing_stats") or TimingStats()
        
        with time_stage(timing_stats, "Output Formatting"):
            state = format_output_node(state)
        
        # Add timing summary to state
        timing_summary = timing_stats.get_summary()
        state["timing_summary"] = timing_summary
        
        # Calculate per-question metrics
        num_questions = len(state.get("validated_questions", []))
        if num_questions > 0:
            avg_time = timing_summary["total_time"] / num_questions
            timing_summary["avg_time_per_question"] = f"{avg_time:.2f}s"
        
        logger.info(f"⏱️  Total generation time: {timing_summary['total_formatted']}")
        logger.info(f"⏱️  Questions generated: {num_questions}")
        if num_questions > 0:
            logger.info(f"⏱️  Average time per question: {timing_summary['avg_time_per_question']}")
        
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
    
    # Add nodes with timing
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generator", timed_generator)
    workflow.add_node("validator", timed_validator)
    workflow.add_node("format_output", timed_format_output)
    
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