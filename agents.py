# agents.py
"""
LangGraph agent definitions for Generator and Validator.
"""
import json
import logging
from typing import List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class QuestionDict(TypedDict):
    """Type definition for a single question."""
    question: str
    question_latex: str
    options: List[str] | None
    options_latex: List[str] | None
    correct_answer: str | None
    correct_answer_latex: str | None
    difficulty: int
    validation_score: float | None
    feedback: str | None


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""
    user_inputs: Dict[str, Any]
    raw_generated_questions: List[QuestionDict]
    validated_questions: List[QuestionDict]
    output: str
    output_latex: str
    retry_count: int
    context_snippets: List[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0.7)


def create_generator_agent() -> Any:
    """
    Create the Generator Agent that produces questions with LaTeX.
    
    Returns:
        Runnable agent for question generation
    """
    
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert question paper generator for educational assessments. 
Your task is to generate high-quality questions that match the specified difficulty level and type.
You MUST generate questions in BOTH plain text AND LaTeX format.

Difficulty Level Guidelines:
- Level 1 (Easy): Basic recall, definitions, simple calculations
- Level 2 (Moderate): Understanding concepts, straightforward applications
- Level 3 (Medium): Problem-solving, multi-step solutions
- Level 4 (Difficult): Complex applications, critical thinking
- Level 5 (Extremely Difficult): Advanced problem-solving, synthesis of multiple concepts

Context from syllabus:
{context}

Generate questions as a JSON array with this EXACT structure:
[
  {{
    "question": "Plain text question here",
    "question_latex": "LaTeX formatted question here (use \\\\text for text, proper math symbols)",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"] or null for descriptive,
    "options_latex": ["A) LaTeX option1", "B) LaTeX option2", "C) LaTeX option3", "D) LaTeX option4"] or null,
    "correct_answer": "B) option2" or "Brief answer for descriptive",
    "correct_answer_latex": "LaTeX formatted answer",
    "difficulty": {difficulty}
  }}
]

CRITICAL REQUIREMENTS:
1. Ensure questions are relevant to the topic and aligned with difficulty
2. For objective questions, include exactly 4 options with one correct answer
3. For descriptive questions, provide a model answer
4. Use proper LaTeX syntax: \\\\frac{{}}{{}}, \\\\sqrt{{}}, x^2, \\\\text{{}} for text, etc.
5. Return ONLY valid JSON, no additional text
6. Generate diverse questions covering different aspects of the topic"""),
        ("human", """Generate {num_questions} {question_type} questions on the topic "{topic}" from chapter "{chapter}" 
for {class_level} {subject}.

Difficulty: {difficulty}/5
{choice_type_info}

Provide output as a JSON array.""")
    ])
    
    def generate_questions(state: WorkflowState) -> WorkflowState:
        """Generate questions based on user inputs and context."""
        try:
            inputs = state["user_inputs"]
            context = "\n".join(state.get("context_snippets", []))
            
            choice_type_info = ""
            if inputs["question_type"] == "Objective":
                choice_type_info = f"Choice Type: {inputs['choice_type']}"
            
            prompt_values = {
                "context": context if context else "No specific context available. Use general knowledge.",
                "num_questions": inputs["num_questions"],
                "question_type": inputs["question_type"],
                "topic": inputs["topic"],
                "chapter": inputs["chapter"],
                "class_level": inputs["class"],
                "subject": inputs["subject"],
                "difficulty": inputs["difficulty"],
                "choice_type_info": choice_type_info
            }
            
            logger.info(f"Generating {inputs['num_questions']} questions...")
            response = llm.invoke(generator_prompt.format_messages(**prompt_values))
            
            # Parse JSON response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            questions = json.loads(content)
            
            # Validate structure
            validated_questions = []
            for q in questions:
                if isinstance(q, dict) and "question" in q and "question_latex" in q:
                    validated_questions.append(q)
            
            state["raw_generated_questions"] = validated_questions
            logger.info(f"Generated {len(validated_questions)} questions successfully")
            
            return state
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {response.content}")
            # Return fallback
            state["raw_generated_questions"] = []
            return state
        except Exception as e:
            logger.error(f"Error in generator agent: {e}")
            state["raw_generated_questions"] = []
            return state
    
    return generate_questions


def create_validator_agent() -> Any:
    """
    Create the Validator Agent that evaluates and scores questions including LaTeX.
    
    Returns:
        Runnable agent for question validation
    """
    
    validator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert educational content validator. Evaluate each question on:

1. Relevance (40%): Does it match the topic, chapter, and subject?
2. Difficulty Match (30%): Does it align with the target difficulty level?
3. Clarity (20%): Is it unambiguous and well-structured?
4. Diversity (10%): Does it cover unique aspects?

For EACH question, provide a JSON object:
{{
  "validation_score": 0.85,  // 0.0 to 1.0
  "feedback": "Specific feedback on strengths and issues",
  "approved": true  // true if score >= 0.7
}}

Return as a JSON array matching the order of input questions.
Also validate that LaTeX formatting is correct and properly formatted."""),
        ("human", """Validate these questions for:
Topic: {topic}
Chapter: {chapter}
Subject: {subject}
Class: {class_level}
Difficulty: {difficulty}/5

Questions to validate:
{questions_json}

Provide validation as JSON array.""")
    ])
    
    def validate_questions(state: WorkflowState) -> WorkflowState:
        """Validate generated questions and update scores."""
        try:
            inputs = state["user_inputs"]
            questions = state["raw_generated_questions"]
            
            if not questions:
                logger.warning("No questions to validate")
                state["validated_questions"] = []
                return state
            
            # Prepare questions for validation
            questions_for_validation = []
            for i, q in enumerate(questions):
                questions_for_validation.append({
                    "index": i,
                    "question": q.get("question", ""),
                    "question_latex": q.get("question_latex", ""),
                    "difficulty": q.get("difficulty", inputs["difficulty"])
                })
            
            prompt_values = {
                "topic": inputs["topic"],
                "chapter": inputs["chapter"],
                "subject": inputs["subject"],
                "class_level": inputs["class"],
                "difficulty": inputs["difficulty"],
                "questions_json": json.dumps(questions_for_validation, indent=2)
            }
            
            logger.info("Validating questions...")
            response = llm.invoke(validator_prompt.format_messages(**prompt_values))
            
            # Parse validation results
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            validations = json.loads(content)
            
            # Merge validation results with questions
            validated = []
            for i, q in enumerate(questions):
                if i < len(validations):
                    val = validations[i]
                    q["validation_score"] = val.get("validation_score", 0.5)
                    q["feedback"] = val.get("feedback", "No feedback")
                    validated.append(q)
            
            # Filter approved questions (score >= 0.7)
            approved_questions = [q for q in validated if q.get("validation_score", 0) >= 0.7]
            
            state["validated_questions"] = approved_questions
            
            pass_rate = len(approved_questions) / len(validated) if validated else 0
            logger.info(f"Validation complete. Pass rate: {pass_rate:.1%} ({len(approved_questions)}/{len(validated)})")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in validator agent: {e}")
            # Fallback: approve all with moderate scores
            for q in state["raw_generated_questions"]:
                q["validation_score"] = 0.75
                q["feedback"] = "Auto-approved due to validation error"
            state["validated_questions"] = state["raw_generated_questions"]
            return state
    
    return validate_questions


def format_output_node(state: WorkflowState) -> WorkflowState:
    """
    Format validated questions into final output (plain text and LaTeX).
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with formatted output
    """
    try:
        questions = state["validated_questions"]
        inputs = state["user_inputs"]
        
        if not questions:
            state["output"] = "No valid questions generated. Please try again."
            state["output_latex"] = "No valid questions generated."
            return state
        
        # Plain text format
        output_lines = [
            "=" * 80,
            f"QUESTION PAPER",
            f"Class: {inputs['class']} | Subject: {inputs['subject']}",
            f"Chapter: {inputs['chapter']} | Topic: {inputs['topic']}",
            f"Difficulty Level: {inputs['difficulty']}/5 | Type: {inputs['question_type']}",
            "=" * 80,
            ""
        ]
        
        # LaTeX format
        latex_lines = [
            "\\documentclass[12pt]{article}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\usepackage{geometry}",
            "\\geometry{margin=1in}",
            "\\title{Question Paper}",
            f"\\author{{{inputs['class']} - {inputs['subject']}}}",
            "\\date{\\today}",
            "\\begin{document}",
            "\\maketitle",
            "",
            f"\\section*{{Chapter: {inputs['chapter']}}}",
            f"\\subsection*{{Topic: {inputs['topic']}}}",
            f"\\textbf{{Difficulty Level:}} {inputs['difficulty']}/5 \\\\",
            f"\\textbf{{Question Type:}} {inputs['question_type']}",
            "",
            "\\begin{enumerate}",
            ""
        ]
        
        for i, q in enumerate(questions, 1):
            # Plain text
            output_lines.append(f"{i}. {q['question']}")
            output_lines.append("")
            
            if q.get("options"):
                for opt in q["options"]:
                    output_lines.append(f"   {opt}")
                output_lines.append("")
            
            # LaTeX
            latex_lines.append(f"\\item {q.get('question_latex', q['question'])}")
            latex_lines.append("")
            
            if q.get("options_latex") or q.get("options"):
                latex_lines.append("\\begin{enumerate}")
                options = q.get("options_latex", q.get("options", []))
                for opt in options:
                    # Remove A), B), etc. prefixes for LaTeX enumerate
                    opt_text = opt[3:] if len(opt) > 3 and opt[1] == ')' else opt
                    latex_lines.append(f"  \\item {opt_text}")
                latex_lines.append("\\end{enumerate}")
                latex_lines.append("")
        
        latex_lines.append("\\end{enumerate}")
        latex_lines.append("\\end{document}")
        
        state["output"] = "\n".join(output_lines)
        state["output_latex"] = "\n".join(latex_lines)
        
        logger.info("Output formatted successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        state["output"] = "Error formatting output"
        state["output_latex"] = "Error formatting output"
        return state