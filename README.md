# AI Question Paper Generator with PDF Upload

A production-ready Streamlit application that generates high-quality educational question papers using LangGraph multi-agent workflows and PDF-based knowledge retrieval.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Question Generation**: LangGraph orchestrates Generator and Validator agents
- **PDF Knowledge Base**: Upload syllabus PDFs to enhance question quality
- **Smart Chunking**: Intelligent document processing with token-aware splitting
- **Vector Search**: Pinecone-powered semantic search for relevant context
- **Quality Validation**: Automated scoring and feedback for generated questions
- **LaTeX Support**: Generate questions in both plain text and LaTeX formats
- **Interactive UI**: User-friendly Streamlit interface with document management

### Question Generation
- Configurable difficulty levels (1-5)
- Multiple question types (Objective/Descriptive)
- Single or multiple choice options
- Automatic validation and retry logic
- Context-aware generation from uploaded PDFs

### Document Management
- Upload multiple PDFs simultaneously
- Automatic text extraction and chunking
- Metadata tagging (class, subject, chapter)
- View and delete uploaded documents
- Real-time vector store statistics

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Pinecone account and API key
- 2GB+ RAM (for PDF processing)

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ai-question-generator
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1  # Your Pinecone environment
```

### 5. Initialize Pinecone Index

The application will automatically create the index on first run. The index name is `syllabus-vectors` with:
- Dimension: 1536 (OpenAI text-embedding-3-small)
- Metric: Cosine similarity
- Cloud: AWS (us-east-1)

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Uploading Syllabus PDFs

1. Navigate to the **"Manage Documents"** tab
2. Select Class, Subject, and Chapter from dropdowns
3. Click "Choose PDF file(s)" and select one or more PDFs
4. Click "🚀 Upload and Process"
5. Wait for processing (progress shown in real-time)
6. Documents are now available for question generation

### Generating Questions

1. Go to the **"Generate Questions"** tab
2. Configure parameters in the sidebar:
   - **Class**: Select grade level
   - **Subject**: Choose subject area
   - **Chapter**: Pick specific chapter
   - **Topic**: Select topic within chapter
   - **Number of Questions**: 1-50 questions
   - **Difficulty**: 1 (Easy) to 5 (Extremely Difficult)
   - **Question Type**: Objective or Descriptive
   - **Choice Type**: Single or Multiple choice (for Objective)
3. Click "🚀 Generate Question Paper"
4. View results in multiple formats:
   - **Plain Text**: Formatted text output
   - **LaTeX**: Compilable LaTeX document
   - **Validation Details**: Quality scores and feedback
   - **Download Options**: Save as .txt or .tex

## 📁 Project Structure

```
ai-question-generator/
├── app.py                      # Main Streamlit application
├── agents.py                   # LangGraph agent definitions
├── graph.py                    # Workflow graph orchestration
├── utils.py                    # Utility functions (Pinecone, embeddings)
├── document_processor.py       # PDF processing and chunking
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

## 🏗️ Architecture

### Multi-Agent Workflow

```
User Input → Retrieve Context → Generator Agent → Validator Agent → Format Output
                                      ↑                    |
                                      |                    ↓
                                      └─── Retry (if <50% pass)
```

### PDF Processing Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding → Pinecone Storage
```

### Key Components

1. **PDFProcessor**: Extracts and chunks text with token awareness
2. **EmbeddingManager**: Generates OpenAI embeddings in batches
3. **Generator Agent**: Creates questions using LLM + vector context
4. **Validator Agent**: Scores questions on relevance, difficulty, clarity
5. **StateGraph**: Orchestrates workflow with conditional retry logic

## 🔧 Configuration

### Chunking Parameters

Edit in `document_processor.py`:

```python
processor = PDFProcessor(
    chunk_size=1000,      # Max tokens per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### LLM Configuration

Edit in `agents.py`:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)
```

### Vector Search

Edit in `utils.py`:

```python
query_syllabus_context(
    index=index,
    query_text=query,
    top_k=5  # Number of relevant chunks to retrieve
)
```

## 📊 Monitoring & Debugging

### View Logs

The application logs to console. For file logging:

```python
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Vector Store Statistics

Check in the "Manage Documents" tab:
- Total vectors (chunks) stored
- Number of unique documents
- Index fullness percentage

### Question Validation Metrics

After generation, view:
- Generated vs. validated question counts
- Pass rate percentage
- Individual question scores and feedback

## 🧪 Testing

### Test Workflow Directly

```bash
python graph.py
```

This runs a test workflow with sample inputs and displays results.

### Test PDF Processing

```python
from document_processor import process_and_embed_pdf

vectors = process_and_embed_pdf(
    pdf_path="path/to/test.pdf",
    api_key=os.getenv("OPENAI_API_KEY"),
    class_level="Class 10",
    subject="Math",
    chapter="Algebra"
)

print(f"Generated {len(vectors)} vectors")
```

## 🐛 Troubleshooting

### PDF Extraction Issues

**Problem**: "Failed to extract text from PDF"

**Solutions**:
- Ensure PDF is not password-protected
- Check if PDF contains actual text (not scanned images)
- Try converting scanned PDFs with OCR first
- Install additional dependencies: `pip install pdf2image tesseract`

### Pinecone Connection Errors

**Problem**: "Failed to connect to Pinecone"

**Solutions**:
- Verify API key in `.env` file
- Check Pinecone environment/region matches your account
- Ensure index exists or will be auto-created
- Check Pinecone dashboard for quota limits

### OpenAI Rate Limits

**Problem**: "Rate limit exceeded"

**Solutions**:
- Reduce batch size in `EmbeddingManager`
- Add delays between API calls
- Upgrade OpenAI plan for higher limits
- Use caching for repeated queries

### Memory Issues

**Problem**: "Out of memory when processing large PDFs"

**Solutions**:
- Reduce `chunk_size` parameter
- Process PDFs in smaller batches
- Increase system RAM
- Use streaming/incremental processing

## 🔒 Security Best Practices

1. **Never commit `.env` file** - Add to `.gitignore`
2. **Rotate API keys regularly** - Update in Pinecone/OpenAI dashboards
3. **Use environment-specific keys** - Separate dev/prod credentials
4. **Validate file uploads** - Check file types and sizes
5. **Sanitize user inputs** - Prevent prompt injection attacks

## 🚀 Production Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in dashboard:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
4. Deploy!

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:

```bash
docker build -t question-generator .
docker run -p 8501:8501 --env-file .env question-generator
```

## 📈 Performance Optimization

### Caching

- Streamlit caches Pinecone connections with `@st.cache_resource`
- Workflow graph is compiled once and reused
- Consider caching frequently queried contexts

### Batch Processing

- Process multiple PDFs in parallel (use `concurrent.futures`)
- Batch embed chunks (already implemented with batch_size=100)
- Use Pinecone batched upserts

### Index Optimization

- Use Pinecone metadata filtering for faster queries
- Periodically clean up outdated vectors
- Consider using namespaces for multi-tenancy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- LangChain for the agent framework
- LangGraph for workflow orchestration
- Pinecone for vector storage
- OpenAI for LLMs and embeddings
- Streamlit for the UI framework

## 📞 Support

For issues and questions:
- Open a GitHub issue
- Check existing issues for solutions
- Review logs in `app.log`
- Contact maintainers

---

**Built with ❤️ for education technology**