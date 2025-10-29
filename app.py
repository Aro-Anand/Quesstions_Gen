# app.py
"""
Main Streamlit application for AI Question Paper Generator.
"""
import streamlit as st
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv
from utils import (
    initialize_pinecone,
    setup_syllabus_index,
    upsert_dummy_data,
    get_curriculum_options
)
from graph import create_workflow_graph
from agents import WorkflowState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .validation-metric {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_app():
    """Initialize Pinecone and set up the workflow (cached)."""
    try:
        with st.spinner("🔧 Initializing application..."):
            # Check for API keys
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OPENAI_API_KEY not found in environment variables!")
                st.stop()
            
            if not os.getenv("PINECONE_API_KEY"):
                st.error("❌ PINECONE_API_KEY not found in environment variables!")
                st.stop()
            
            # Initialize Pinecone
            pc = initialize_pinecone()
            index = setup_syllabus_index(pc)
            
            # Check if index needs population
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                logger.info("Index empty, upserting dummy data...")
                upsert_dummy_data(index)
            
            # Create workflow
            workflow = create_workflow_graph(index)
            
            logger.info("Application initialized successfully")
            return workflow, index
    
    except Exception as e:
        st.error(f"❌ Initialization error: {e}")
        logger.error(f"Initialization error: {e}")
        st.stop()


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with user input controls.
    
    Returns:
        Dictionary of user inputs
    """
    st.sidebar.markdown("## 📚 Question Paper Configuration")
    st.sidebar.markdown("---")
    
    # Get curriculum options
    curriculum = get_curriculum_options()
    
    # Class selection
    class_level = st.sidebar.selectbox(
        "🎓 Class",
        options=list(curriculum.keys()),
        help="Select the class level"
    )
    
    # Subject selection (dynamic based on class)
    subjects = list(curriculum[class_level].keys())
    subject = st.sidebar.selectbox(
        "📖 Subject",
        options=subjects,
        help="Select the subject"
    )
    
    # Chapter selection (dynamic based on subject)
    chapters = list(curriculum[class_level][subject].keys())
    chapter = st.sidebar.selectbox(
        "📑 Chapter",
        options=chapters,
        help="Select the chapter"
    )
    
    # Topic selection (dynamic based on chapter)
    topics = curriculum[class_level][subject][chapter]
    topic = st.sidebar.selectbox(
        "🎯 Topic",
        options=topics,
        help="Select the specific topic"
    )
    
    st.sidebar.markdown("---")
    
    # Number of questions
    num_questions = st.sidebar.slider(
        "🔢 Number of Questions",
        min_value=1,
        max_value=50,
        value=10,
        help="How many questions to generate"
    )
    
    # Difficulty level
    difficulty = st.sidebar.slider(
        "⚡ Difficulty Level",
        min_value=1,
        max_value=5,
        value=3,
        format="%d",
        help="1=Easy, 5=Extremely Difficult"
    )
    
    difficulty_labels = {
        1: "Easy",
        2: "Moderate",
        3: "Medium",
        4: "Difficult",
        5: "Extremely Difficult"
    }
    st.sidebar.caption(f"Selected: **{difficulty_labels[difficulty]}**")
    
    st.sidebar.markdown("---")
    
    # Question type
    question_type = st.sidebar.radio(
        "📝 Question Type",
        options=["Objective", "Descriptive"],
        help="Choose between objective (MCQ) or descriptive questions"
    )
    
    # Choice type (only for objective)
    choice_type = None
    if question_type == "Objective":
        choice_type = st.sidebar.radio(
            "✅ Choice Type",
            options=["Single Choice", "Multiple Choice"],
            help="Single correct answer or multiple correct answers"
        )
    
    st.sidebar.markdown("---")
    
    return {
        "class": class_level,
        "subject": subject,
        "chapter": chapter,
        "topic": topic,
        "num_questions": num_questions,
        "difficulty": difficulty,
        "question_type": question_type,
        "choice_type": choice_type or "Single Choice"
    }


def display_validation_metrics(result: Dict[str, Any]):
    """Display validation metrics in a nice format."""
    raw_count = len(result.get("raw_generated_questions", []))
    validated_count = len(result.get("validated_questions", []))
    retry_count = result.get("retry_count", 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Generated",
            value=raw_count,
            help="Total questions generated"
        )
    
    with col2:
        st.metric(
            label="✅ Validated",
            value=validated_count,
            help="Questions that passed validation"
        )
    
    with col3:
        pass_rate = (validated_count / raw_count * 100) if raw_count > 0 else 0
        st.metric(
            label="📈 Pass Rate",
            value=f"{pass_rate:.1f}%",
            help="Percentage of questions that passed"
        )
    
    if retry_count > 0:
        st.info(f"🔄 Workflow retried {retry_count} time(s) to improve quality")


def display_question_details(result: Dict[str, Any]):
    """Display detailed validation scores and feedback."""
    questions = result.get("validated_questions", [])
    
    if not questions:
        st.warning("⚠️ No validated questions available")
        return
    
    st.markdown("### 📋 Question Validation Details")
    
    for i, q in enumerate(questions, 1):
        with st.expander(f"Question {i} - Score: {q.get('validation_score', 0):.2f}"):
            st.markdown(f"**Question:** {q['question']}")
            
            if q.get("options"):
                st.markdown("**Options:**")
                for opt in q["options"]:
                    st.markdown(f"- {opt}")
            
            score = q.get("validation_score", 0)
            feedback = q.get("feedback", "No feedback available")
            
            # Color code based on score
            if score >= 0.8:
                score_color = "🟢"
            elif score >= 0.7:
                score_color = "🟡"
            else:
                score_color = "🔴"
            
            st.markdown(f"{score_color} **Validation Score:** {score:.2f}")
            st.markdown(f"**Feedback:** {feedback}")
            
            # LaTeX preview
            if q.get("question_latex"):
                st.markdown("**LaTeX Preview:**")
                st.code(q["question_latex"], language="latex")


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<div class="main-header">📝 AI Question Paper Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Generate high-quality educational assessments with AI-powered multi-agent workflow</div>',
        unsafe_allow_html=True
    )
    
    # Initialize app
    try:
        workflow, index = initialize_app()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.stop()
    
    # Sidebar inputs
    user_inputs = render_sidebar()
    
    # Display current configuration
    with st.expander("📌 Current Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **Class:** {user_inputs['class']}
            - **Subject:** {user_inputs['subject']}
            - **Chapter:** {user_inputs['chapter']}
            - **Topic:** {user_inputs['topic']}
            """)
        with col2:
            st.markdown(f"""
            - **Questions:** {user_inputs['num_questions']}
            - **Difficulty:** {user_inputs['difficulty']}/5
            - **Type:** {user_inputs['question_type']}
            - **Choice:** {user_inputs['choice_type']}
            """)
    
    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("🚀 Generate Question Paper", use_container_width=True)
    
    # Generate questions when button is clicked
    if generate_button:
        try:
            # Create initial state
            initial_state: WorkflowState = {
                "user_inputs": user_inputs,
                "raw_generated_questions": [],
                "validated_questions": [],
                "output": "",
                "output_latex": "",
                "retry_count": 0,
                "context_snippets": []
            }
            
            # Run workflow with progress indicator
            with st.spinner("🔄 Generating questions... This may take a moment."):
                result = workflow.invoke(initial_state)
            
            # Check if generation was successful
            if not result.get("validated_questions"):
                st.error("❌ Failed to generate valid questions. Please try again with different parameters.")
                return
            
            # Success message
            st.success(f"✅ Successfully generated {len(result['validated_questions'])} questions!")
            
            # Display metrics
            st.markdown("---")
            display_validation_metrics(result)
            
            # Main output tabs
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs([
                "📄 Plain Text Output",
                "🔤 LaTeX Output",
                "📊 Validation Details",
                "💾 Download Options"
            ])
            
            with tab1:
                st.markdown("### Question Paper (Plain Text)")
                output_text = result.get("output", "No output generated")
                st.text_area(
                    "Generated Questions",
                    value=output_text,
                    height=600,
                    label_visibility="collapsed"
                )
            
            with tab2:
                st.markdown("### Question Paper (LaTeX)")
                st.markdown("""
                Copy the LaTeX code below and compile it with your favorite LaTeX editor 
                (e.g., Overleaf, TeXShop, or pdflatex).
                """)
                output_latex = result.get("output_latex", "No LaTeX output generated")
                st.code(output_latex, language="latex", line_numbers=True)
            
            with tab3:
                display_question_details(result)
            
            with tab4:
                st.markdown("### 💾 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download plain text
                    st.download_button(
                        label="📥 Download Plain Text (.txt)",
                        data=result.get("output", ""),
                        file_name="question_paper.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Download LaTeX
                    st.download_button(
                        label="📥 Download LaTeX (.tex)",
                        data=result.get("output_latex", ""),
                        file_name="question_paper.tex",
                        mime="text/x-tex",
                        use_container_width=True
                    )
                
                st.info("""
                **💡 Tip:** To compile the LaTeX file:
                1. Download the .tex file
                2. Upload to [Overleaf](https://www.overleaf.com) or use a local LaTeX compiler
                3. Compile to generate a PDF
                """)
            
            # Store result in session state for reference
            st.session_state["last_result"] = result
            
        except Exception as e:
            st.error(f"❌ An error occurred during generation: {str(e)}")
            logger.error(f"Generation error: {e}", exc_info=True)
            
            with st.expander("🔍 Error Details"):
                st.code(str(e))
    


if __name__ == "__main__":
    main()