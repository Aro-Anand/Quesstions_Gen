# utils.py
"""
Utility functions for Pinecone setup, embeddings, and dummy data management.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding for given text using OpenAI.
    
    Args:
        text: Input text to embed
        model: OpenAI embedding model name
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def initialize_pinecone() -> Pinecone:
    """
    Initialize Pinecone client and create/connect to index.
    
    Returns:
        Pinecone client instance
    """
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise


def setup_syllabus_index(pc: Pinecone, index_name: str = "syllabus-vectors") -> Any:
    """
    Create or connect to Pinecone index and populate with dummy syllabus data.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index to create/use
        
    Returns:
        Pinecone index object
    """
    try:
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Connect to index
        index = pc.Index(index_name)
        logger.info(f"Connected to index: {index_name}")
        
        return index
    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        raise


def get_dummy_syllabus_data() -> List[Dict[str, Any]]:
    """
    Generate dummy syllabus content for pre-population.
    
    Returns:
        List of syllabus content dictionaries
    """
    return [
        {
            "id": "math10_algebra_quadratic_1",
            "text": "Quadratic equations involve solving equations of the form ax² + bx + c = 0. Methods include factoring, completing the square, and using the quadratic formula.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Algebra", "topic": "Quadratic Equations"}
        },
        {
            "id": "math10_algebra_quadratic_2",
            "text": "The discriminant (b² - 4ac) determines the nature of roots. Positive discriminant gives two real roots, zero gives one repeated root, negative gives complex roots.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Algebra", "topic": "Quadratic Equations"}
        },
        {
            "id": "math10_geometry_triangles_1",
            "text": "Similar triangles have equal corresponding angles and proportional sides. The Pythagorean theorem applies to right triangles: a² + b² = c².",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Geometry", "topic": "Triangles"}
        },
        {
            "id": "math10_geometry_circles_1",
            "text": "Circle properties include radius, diameter, chord, tangent, and secant. The area is πr² and circumference is 2πr.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Geometry", "topic": "Circles"}
        },
        {
            "id": "math12_calculus_derivatives_1",
            "text": "Derivatives represent the rate of change. Basic rules include power rule: d/dx(x^n) = nx^(n-1), product rule, and chain rule.",
            "metadata": {"class": "Class 12", "subject": "Math", "chapter": "Calculus", "topic": "Derivatives"}
        },
        {
            "id": "math12_calculus_integrals_1",
            "text": "Integration is the reverse of differentiation. Definite integrals calculate area under curves. Fundamental theorem connects derivatives and integrals.",
            "metadata": {"class": "Class 12", "subject": "Math", "chapter": "Calculus", "topic": "Integration"}
        },
        {
            "id": "science10_physics_motion_1",
            "text": "Newton's laws of motion: First law (inertia), Second law (F=ma), Third law (action-reaction). These govern motion and forces.",
            "metadata": {"class": "Class 10", "subject": "Science", "chapter": "Physics", "topic": "Motion and Force"}
        },
        {
            "id": "science10_chemistry_acids_1",
            "text": "Acids are proton donors with pH < 7. Bases are proton acceptors with pH > 7. Neutralization reaction: acid + base → salt + water.",
            "metadata": {"class": "Class 10", "subject": "Science", "chapter": "Chemistry", "topic": "Acids and Bases"}
        },
        {
            "id": "science10_biology_cells_1",
            "text": "Cells are the basic unit of life. Prokaryotic cells lack a nucleus, eukaryotic cells have membrane-bound organelles including nucleus, mitochondria, and chloroplasts.",
            "metadata": {"class": "Class 10", "subject": "Science", "chapter": "Biology", "topic": "Cell Structure"}
        },
        {
            "id": "math10_algebra_polynomials_1",
            "text": "Polynomials are algebraic expressions with terms containing variables raised to whole number powers. Operations include addition, subtraction, multiplication, and division.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Algebra", "topic": "Polynomials"}
        },
        {
            "id": "math10_trigonometry_ratios_1",
            "text": "Trigonometric ratios in right triangles: sin(θ) = opposite/hypotenuse, cos(θ) = adjacent/hypotenuse, tan(θ) = opposite/adjacent. These relate angles to side lengths.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Trigonometry", "topic": "Trigonometric Ratios"}
        },
        {
            "id": "science12_physics_electricity_1",
            "text": "Ohm's law states V = IR where V is voltage, I is current, R is resistance. Power is given by P = VI. Series and parallel circuits have different resistance combinations.",
            "metadata": {"class": "Class 12", "subject": "Science", "chapter": "Physics", "topic": "Electricity"}
        },
        {
            "id": "science12_chemistry_organic_1",
            "text": "Organic chemistry studies carbon compounds. Hydrocarbons include alkanes (C-C single bonds), alkenes (C=C double bonds), and alkynes (C≡C triple bonds). Functional groups determine reactivity.",
            "metadata": {"class": "Class 12", "subject": "Science", "chapter": "Chemistry", "topic": "Organic Chemistry"}
        },
        {
            "id": "math12_vectors_1",
            "text": "Vectors have magnitude and direction. Operations include addition (triangle/parallelogram law), scalar multiplication, dot product (scalar result), and cross product (vector result).",
            "metadata": {"class": "Class 12", "subject": "Math", "chapter": "Vectors", "topic": "Vector Operations"}
        },
        {
            "id": "math10_statistics_mean_1",
            "text": "Measures of central tendency include mean (average), median (middle value), and mode (most frequent). Standard deviation measures spread of data.",
            "metadata": {"class": "Class 10", "subject": "Math", "chapter": "Statistics", "topic": "Central Tendency"}
        }
    ]


def upsert_dummy_data(index: Any) -> None:
    """
    Populate Pinecone index with dummy syllabus embeddings.
    
    Args:
        index: Pinecone index object
    """
    try:
        syllabus_data = get_dummy_syllabus_data()
        vectors = []
        
        logger.info("Generating embeddings for dummy data...")
        for item in syllabus_data:
            embedding = get_embedding(item["text"])
            vectors.append({
                "id": item["id"],
                "values": embedding,
                "metadata": {
                    **item["metadata"],
                    "text": item["text"]
                }
            })
        
        # Upsert in batches
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
        index.upsert(vectors=vectors)
        logger.info("Dummy data upserted successfully")
    except Exception as e:
        logger.error(f"Error upserting dummy data: {e}")
        raise


def query_syllabus_context(
    index: Any,
    query_text: str,
    filters: Optional[Dict[str, str]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for relevant syllabus context.
    
    Args:
        index: Pinecone index object
        query_text: Text to search for
        filters: Metadata filters (class, subject, etc.)
        top_k: Number of results to return
        
    Returns:
        List of matching context dictionaries
    """
    try:
        query_embedding = get_embedding(query_text)
        
        # Build filter dict for Pinecone
        filter_dict = {}
        if filters:
            for key, value in filters.items():
                if value:
                    filter_dict[key] = value
        
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filter_dict:
            query_params["filter"] = filter_dict
        
        results = index.query(**query_params)
        
        contexts = []
        for match in results.matches:
            contexts.append({
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "metadata": match.metadata
            })
        
        return contexts
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []


def get_curriculum_options() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Get hierarchical curriculum options for UI dropdowns.
    
    Returns:
        Nested dictionary of class -> subject -> chapter -> topics
    """
    return {
        "Class 10": {
            "Math": {
                "Algebra": ["Quadratic Equations", "Polynomials", "Linear Equations"],
                "Geometry": ["Triangles", "Circles", "Coordinate Geometry"],
                "Trigonometry": ["Trigonometric Ratios", "Heights and Distances"],
                "Statistics": ["Central Tendency", "Probability"]
            },
            "Science": {
                "Physics": ["Motion and Force", "Light", "Sound"],
                "Chemistry": ["Acids and Bases", "Metals and Non-metals", "Chemical Reactions"],
                "Biology": ["Cell Structure", "Heredity", "Life Processes"]
            }
        },
        "Class 12": {
            "Math": {
                "Calculus": ["Derivatives", "Integration", "Differential Equations"],
                "Vectors": ["Vector Operations", "3D Geometry"],
                "Probability": ["Conditional Probability", "Distributions"]
            },
            "Science": {
                "Physics": ["Electricity", "Magnetism", "Optics", "Modern Physics"],
                "Chemistry": ["Organic Chemistry", "Chemical Kinetics", "Electrochemistry"],
                "Biology": ["Genetics", "Evolution", "Ecology"]
            }
        }
    }