# app.py
"""
Enhanced Streamlit application with timing and HTML export.
"""
import streamlit as st
import logging
import os
import tempfile
from typing import Dict, Any
from dotenv import load_dotenv
from utils import (
    initialize_pinecone,
    setup_syllabus_index,
    get_curriculum_options,
    upload_pdf_to_vectorstore,
    get_uploaded_documents,
    delete_document_from_vectorstore,
    get_vectorstore_stats
)
from graph import create_workflow_graph
from agents import WorkflowState
from timing_decorator import TimingStats

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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
    .timing-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stage-timing {
        background-color: rgba(255,255,255,0.2);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
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
            
            # Create workflow
            workflow = create_workflow_graph(index)
            
            logger.info("Application initialized successfully")
            return workflow, index
    
    except Exception as e:
        st.error(f"❌ Initialization error: {e}")
        logger.error(f"Initialization error: {e}")
        st.stop()


def render_document_management_tab(index):
    """Render document upload and management interface."""
    st.markdown("## 📚 Document Management")
    st.markdown("Upload syllabus PDFs to enhance question generation quality")
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        stats = get_vectorstore_stats(index)
        st.metric(
            label="📊 Total Vectors",
            value=f"{stats['total_vectors']:,}",
            help="Number of document chunks in vector store"
        )
    
    with col2:
        uploaded_docs = get_uploaded_documents(index)
        st.metric(
            label="📄 Documents",
            value=len(uploaded_docs),
            help="Number of unique documents uploaded"
        )
    
    st.markdown("---")
    
    # Upload Section
    st.markdown("### 📤 Upload New Documents")
    
    curriculum = get_curriculum_options()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        upload_class = st.selectbox(
            "Class",
            options=list(curriculum.keys()),
            key="upload_class"
        )
    
    with col2:
        upload_subject = st.selectbox(
            "Subject",
            options=list(curriculum[upload_class].keys()),
            key="upload_subject"
        )
    
    with col3:
        upload_chapter = st.selectbox(
            "Chapter",
            options=list(curriculum[upload_class][upload_subject].keys()),
            key="upload_chapter"
        )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files containing syllabus content"
    )
    
    if uploaded_files:
        st.info(f"📋 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Upload and Process", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Define progress callback
                    def update_progress(msg):
                        status_text.text(f"{uploaded_file.name}: {msg}")
                    
                    # Upload to vector store
                    result = upload_pdf_to_vectorstore(
                        index=index,
                        pdf_path=tmp_path,
                        class_level=upload_class,
                        subject=upload_subject,
                        chapter=upload_chapter,
                        progress_callback=update_progress
                    )
                    
                    results.append(result)
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display results
            status_text.empty()
            progress_bar.empty()
            
            success_count = sum(1 for r in results if r["status"] == "success")
            
            if success_count == len(results):
                st.success(f"✅ Successfully uploaded {success_count} document(s)!")
            else:
                st.warning(f"⚠️ Uploaded {success_count}/{len(results)} document(s)")
            
            # Show details
            for result in results:
                if result["status"] == "success":
                    st.success(
                        f"✅ **{result['filename']}**: "
                        f"{result['chunks_uploaded']} chunks uploaded"
                    )
                else:
                    st.error(f"❌ **{result['filename']}**: {result.get('error', 'Unknown error')}")
            
            # Clear cache to refresh document list
            st.rerun()
    
    st.markdown("---")
    
    # Existing Documents
    st.markdown("### 📋 Uploaded Documents")
    
    uploaded_docs = get_uploaded_documents(index)
    
    if not uploaded_docs:
        st.info("No documents uploaded yet. Upload PDFs above to get started!")
    else:
        for doc in uploaded_docs:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 1rem; border-radius: 8px; 
                     border-left: 4px solid #1f77b4; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong>📄 {doc['filename']}</strong><br>
                    <small>Class: {doc['class']} | Subject: {doc['subject']} | Chapter: {doc['chapter']}</small><br>
                    <small>Hash: {doc['file_hash'][:16]}...</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🗑️ Delete", key=f"del_{doc['file_hash']}"):
                    with st.spinner("Deleting..."):
                        success = delete_document_from_vectorstore(index, doc['file_hash'])
                        if success:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed")


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with user input controls."""
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


def display_timing_metrics(timing_summary: Dict[str, Any]):
    """Display timing metrics in an attractive format."""
    st.markdown("### ⏱️ Generation Performance Metrics")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⏱️ Total Time",
            value=timing_summary.get("total_formatted", "N/A"),
            help="Total time for complete generation process"
        )
    
    with col2:
        st.metric(
            label="📝 Per Question",
            value=timing_summary.get("avg_time_per_question", "N/A"),
            help="Average time per question"
        )
    
    with col3:
        total_time = timing_summary.get("total_time", 0)
        questions_per_min = (60 / (total_time / len(st.session_state.get("last_result", {}).get("validated_questions", [1])))) if total_time > 0 else 0
        st.metric(
            label="🚀 Questions/Min",
            value=f"{questions_per_min:.1f}",
            help="Generation rate"
        )
    
    with col4:
        # Calculate efficiency score (higher is better)
        efficiency = min(100, int((1 / total_time * 100) if total_time > 0 else 100))
        st.metric(
            label="⚡ Efficiency",
            value=f"{efficiency}%",
            help="Overall generation efficiency"
        )
    
    # Stage-wise breakdown
    if "stages_formatted" in timing_summary:
        st.markdown("#### 📊 Stage-wise Breakdown")
        
        stages = timing_summary["stages_formatted"]
        stage_times = timing_summary.get("stages", {})
        total = timing_summary.get("total_time", 1)
        
        for stage, formatted_time in stages.items():
            percentage = (stage_times.get(stage, 0) / total * 100) if total > 0 else 0
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.progress(percentage / 100)
            with col2:
                st.markdown(f"**{stage}**")
            with col3:
                st.markdown(f"`{formatted_time}` ({percentage:.1f}%)")


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
    
    # Initialize session state for storing results
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "html_content" not in st.session_state:
        st.session_state["html_content"] = None
    
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
    
    # Main tabs
    tab1, tab2 = st.tabs(["📝 Generate Questions", "📚 Manage Documents"])
    
    with tab1:
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
            generate_button = st.button("🚀 Generate Question Paper", use_container_width=True, key="generate_btn")
        
        # Generate questions when button is clicked
        if generate_button:
            # Clear previous HTML content
            st.session_state["html_content"] = None
            
            try:
                # Create initial state with timing
                initial_state: WorkflowState = {
                    "user_inputs": user_inputs,
                    "raw_generated_questions": [],
                    "validated_questions": [],
                    "output": "",
                    "output_latex": "",
                    "retry_count": 0,
                    "context_snippets": [],
                    "timing_stats": TimingStats(),
                    "timing_summary": {}
                }
                
                # Run workflow with progress indicator
                with st.spinner("🔄 Generating questions... This may take a moment."):
                    result = workflow.invoke(initial_state)
                
                # Check if generation was successful
                if not result.get("validated_questions"):
                    st.error("❌ Failed to generate valid questions. Please try again with different parameters.")
                else:
                    # Store result in session state
                    st.session_state["last_result"] = result
                    
                    # Success message
                    st.success(f"✅ Successfully generated {len(result['validated_questions'])} questions!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred during generation: {str(e)}")
                logger.error(f"Generation error: {e}", exc_info=True)
                
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
        
        # Display results if they exist in session state
        if st.session_state.get("last_result"):
            result = st.session_state["last_result"]
            
            # Display timing metrics
            st.markdown("---")
            if "timing_summary" in result:
                display_timing_metrics(result["timing_summary"])
            
            # Display validation metrics
            st.markdown("---")
            display_validation_metrics(result)
            
            # Main output tabs
            st.markdown("---")
            output_tab1, output_tab2, output_tab3, output_tab4, output_tab5 = st.tabs([
                "📄 Plain Text Output",
                "📤 LaTeX Output",
                "🌐 HTML Preview",
                "📊 Validation Details",
                "💾 Download Options"
            ])
            
            with output_tab1:
                st.markdown("### Question Paper (Plain Text)")
                output_text = result.get("output", "No output generated")
                st.text_area(
                    "Generated Questions",
                    value=output_text,
                    height=600,
                    label_visibility="collapsed"
                )
            
            with output_tab2:
                st.markdown("### Question Paper (LaTeX)")
                st.markdown("""
                Copy the LaTeX code below and compile it with your favorite LaTeX editor 
                (e.g., Overleaf, TeXShop, or pdflatex).
                """)
                output_latex = result.get("output_latex", "No LaTeX output generated")
                st.code(output_latex, language="latex", line_numbers=True)
            
            with output_tab3:
                st.markdown("### Question Paper (HTML from LaTeX)")
                
                st.info("🔄 Convert your LaTeX document to HTML while preserving all formatting")
                
                # Convert button with unique key
                if st.button("🌐 Convert LaTeX to HTML", key="convert_latex_html", use_container_width=True):
                    try:
                        logger.info("Starting LaTeX to HTML conversion process")
                        from html_exporter import LaTeXToHTMLConverter
                        
                        # Initialize converter
                        converter = LaTeXToHTMLConverter()
                        logger.info("LaTeX converter initialized")
                        
                        # Get LaTeX content
                        latex_content = result.get("output_latex", "")
                        if not latex_content:
                            logger.error("No LaTeX content available")
                            st.error("❌ No LaTeX content available")
                        else:
                            logger.info(f"Got LaTeX content (length: {len(latex_content)})")
                            
                            with st.spinner("Converting LaTeX to HTML with Pandoc..."):
                                logger.info("Starting Pandoc conversion")
                                # Convert LaTeX to HTML
                                success, html_content = converter.latex_to_html(
                                    latex_content=latex_content,
                                    include_mathjax=True
                                )
                                
                                if success and html_content:
                                    logger.info("HTML conversion successful")
                                    # Store in session state
                                    st.session_state["html_content"] = html_content
                                    st.success("✅ LaTeX converted to HTML successfully!")
                                    st.rerun()
                                else:
                                    logger.error("HTML conversion failed")
                                    st.error("❌ Failed to convert LaTeX to HTML")
                                    st.info("💡 Make sure Pandoc is installed: https://pandoc.org/installing.html")
                    
                    except Exception as e:
                        logger.error(f"Conversion error: {str(e)}", exc_info=True)
                        st.error(f"❌ Error during conversion: {str(e)}")
                        
                        with st.expander("🔍 Error Details"):
                            st.code(str(e))
                
                # Display HTML if it exists
                if st.session_state.get("html_content"):
                    st.markdown("---")
                    st.markdown("### 👀 HTML Preview")
                    st.components.v1.html(st.session_state["html_content"], height=600, scrolling=True)
            
            with output_tab4:
                display_question_details(result)
            
            with output_tab5:
                st.markdown("### 💾 Download Options")
                
                col1, col2, col3 = st.columns(3)
                
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
                
                with col3:
                    # Download HTML if generated
                    if st.session_state.get("html_content"):
                        st.download_button(
                            label="📥 Download HTML (.html)",
                            data=st.session_state["html_content"],
                            file_name="question_paper.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    else:
                        st.info("💡 Convert to HTML first", icon="ℹ️")
                
                st.markdown("---")
                st.info("""
                **💡 Tips:**
                - **Plain Text**: Ready to copy-paste into any document
                - **LaTeX**: Upload to [Overleaf](https://www.overleaf.com) or use local compiler
                - **HTML**: Open in browser for a beautiful, shareable blog-style page
                """)
    
    with tab2:
        render_document_management_tab(index)


if __name__ == "__main__":
    main()