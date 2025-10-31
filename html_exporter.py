# html_exporter.py
"""
LaTeX to HTML conversion using Pandoc while preserving formatting.
"""
import os
import subprocess
import tempfile
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LaTeXToHTMLConverter:
    """Convert LaTeX documents to HTML while preserving formatting."""
    
    def __init__(self):
        self.check_pandoc_installed()
    
    def check_pandoc_installed(self) -> bool:
        """Check if Pandoc is installed."""
        try:
            result = subprocess.run(
                ["pandoc", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ Pandoc is installed")
                return True
        except FileNotFoundError:
            logger.warning("⚠️  Pandoc not found. Install from https://pandoc.org/installing.html")
        return False
    
    def latex_to_html(
        self,
        latex_content: str,
        output_path: Optional[str] = None,
        include_mathjax: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Convert LaTeX document to HTML using Pandoc.
        
        Args:
            latex_content: LaTeX document content
            output_path: Path for output HTML file (optional)
            include_mathjax: Include MathJax for math rendering
            
        Returns:
            Tuple of (success: bool, html_content: str or None)
        """
        try:
            logger.info("🚀 Starting LaTeX to HTML conversion...")
            logger.info(f"Input LaTeX content length: {len(latex_content)} characters")
            
            # Create temporary LaTeX file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.tex',
                delete=False,
                encoding='utf-8'
            ) as tmp_tex:
                tmp_tex.write(latex_content)
                tmp_tex_path = tmp_tex.name
                logger.info(f"✓ Created temporary LaTeX file: {tmp_tex_path}")
            
            # Create temporary or use specified output path
            if output_path is None:
                tmp_html = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.html',
                    delete=False
                )
                output_path = tmp_html.name
                tmp_html.close()
            
            # Build Pandoc command
            cmd = [
                "pandoc",
                tmp_tex_path,
                "-f", "latex",  # From LaTeX
                "-t", "html5",  # To HTML5
                "-o", output_path,
                "--standalone",  # Complete HTML document
                "--self-contained",  # Embed resources
                "--css", self._get_custom_css_path(),
            ]
            
            # Add MathJax for math rendering
            if include_mathjax:
                cmd.extend([
                    "--mathjax",
                    "--mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
                ])
            
            # Add metadata
            cmd.extend([
                "--metadata", "pagetitle=Question Paper",
                "--metadata", f"date={datetime.now().strftime('%B %d, %Y')}"
            ])
            
            # Log Pandoc command
            logger.info("📝 Executing Pandoc command:")
            logger.info(" ".join(cmd))
            
            # Run Pandoc
            logger.info("⚙️ Running Pandoc conversion...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temp LaTeX file
            os.unlink(tmp_tex_path)
            logger.info("✓ Cleaned up temporary LaTeX file")
            
            if result.returncode == 0:
                logger.info("✅ Pandoc conversion successful")
                # Read HTML content
                with open(output_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Enhance HTML with additional styling
                html_content = self._enhance_html(html_content)
                
                # Write back enhanced HTML
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"✓ Successfully converted LaTeX to HTML: {output_path}")
                return True, html_content
            else:
                logger.error("❌ Pandoc conversion failed!")
                logger.error(f"Pandoc stderr: {result.stderr}")
                logger.error(f"Pandoc stdout: {result.stdout}")
                return False, None
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Pandoc conversion timed out after 30 seconds")
            return False, None
        except Exception as e:
            logger.error("❌ Error during LaTeX to HTML conversion:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return False, None
    
    def _get_custom_css_path(self) -> str:
        """Create and return path to custom CSS file."""
        css_content = """
/* Base Styles */
body {
    font-family: 'Latin Modern Roman', 'Computer Modern', 'Times New Roman', serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 30px;
    line-height: 1.8;
    color: #2c3e50;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Document Container */
.container {
    background: white;
    padding: 40px;
    border-radius: 10px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

/* Title Styling - Preserve LaTeX Title Format */
h1.title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #1a237e;
    margin-bottom: 0.5em;
    padding-bottom: 20px;
    border-bottom: 3px solid #1a237e;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Author/Metadata */
.author, .date {
    text-align: center;
    font-size: 1.2em;
    color: #555;
    margin: 10px 0;
    font-style: italic;
}

/* Section Headers */
h2, .section {
    color: #1565c0;
    font-size: 1.8em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    padding-bottom: 10px;
    border-bottom: 2px solid #e0e0e0;
    font-weight: 600;
}

h3, .subsection {
    color: #1976d2;
    font-size: 1.4em;
    margin-top: 1.2em;
    margin-bottom: 0.6em;
    font-weight: 500;
}

/* Preserve LaTeX Formatting */
.textbf, strong, b {
    font-weight: bold;
    color: #1a237e;
}

.textit, em, i {
    font-style: italic;
}

/* Enumerated Lists - Preserve LaTeX enumerate */
ol {
    counter-reset: item;
    list-style-type: none;
    padding-left: 0;
    margin: 20px 0;
}

ol > li {
    counter-increment: item;
    margin-bottom: 25px;
    padding: 15px;
    background: #f8f9fa;
    border-left: 4px solid #1565c0;
    border-radius: 5px;
    position: relative;
}

ol > li::before {
    content: counter(item) ". ";
    font-weight: bold;
    font-size: 1.2em;
    color: #1565c0;
    margin-right: 10px;
}

/* Nested enumerate (for options) */
ol ol {
    margin: 10px 0;
    padding-left: 20px;
}

ol ol > li {
    background: white;
    border-left: 3px solid #90caf9;
    padding: 8px 12px;
    margin-bottom: 8px;
}

ol ol > li::before {
    content: "(" counter(item, lower-alpha) ") ";
    font-weight: normal;
    color: #1976d2;
}

/* Item Lists */
ul {
    list-style-type: none;
    padding-left: 0;
}

ul > li {
    padding: 8px 0 8px 25px;
    position: relative;
}

ul > li::before {
    content: "▸";
    position: absolute;
    left: 0;
    color: #1565c0;
    font-weight: bold;
}

/* Math Styling */
.math, .display-math {
    font-family: 'Latin Modern Math', 'STIX Two Math', 'Cambria Math', serif;
    color: #1a237e;
}

/* Display Math */
.display-math, .equation {
    display: block;
    margin: 20px auto;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 5px;
    text-align: center;
    overflow-x: auto;
}

/* Inline Math */
.inline-math {
    padding: 2px 4px;
    background: #f0f0f0;
    border-radius: 3px;
}

/* Horizontal Rules */
hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 30px 0;
}

/* Paragraphs */
p {
    margin: 15px 0;
    text-align: justify;
    hyphens: auto;
}

/* Metadata Section */
.metadata {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
}

.metadata p {
    margin: 5px 0;
    text-align: left;
}

/* Code/Verbatim */
code, .verbatim {
    font-family: 'Courier New', 'Courier', monospace;
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background: white;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background: #1565c0;
    color: white;
    font-weight: 600;
}

tr:hover {
    background: #f5f5f5;
}

/* Footer */
footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #e0e0e0;
    color: #777;
    font-style: italic;
}

/* Print Styles */
@media print {
    body {
        background: white;
        margin: 0;
        padding: 20px;
    }
    
    .container {
        box-shadow: none;
    }
    
    h1.title {
        color: black;
        border-bottom-color: black;
    }
    
    h2, h3 {
        color: black;
    }
    
    ol > li {
        break-inside: avoid;
        page-break-inside: avoid;
    }
}

/* Mobile Responsive */
@media (max-width: 768px) {
    body {
        padding: 15px;
        margin: 0;
    }
    
    .container {
        padding: 20px;
    }
    
    h1.title {
        font-size: 1.8em;
    }
    
    h2 {
        font-size: 1.5em;
    }
    
    ol > li {
        padding: 12px;
    }
}

/* MathJax Styling */
.MathJax {
    font-size: 1.1em !important;
}

.MathJax_Display {
    margin: 1em 0 !important;
}

/* Preserve LaTeX spacing */
.quad {
    display: inline-block;
    width: 1em;
}

.qquad {
    display: inline-block;
    width: 2em;
}

/* LaTeX-style section numbering */
.section-number {
    font-weight: bold;
    margin-right: 0.5em;
}

/* Theorem-like environments */
.theorem, .definition, .proof {
    margin: 20px 0;
    padding: 15px;
    border-left: 4px solid #1565c0;
    background: #e3f2fd;
}

.theorem::before {
    content: "Theorem: ";
    font-weight: bold;
}

.definition::before {
    content: "Definition: ";
    font-weight: bold;
}

.proof::before {
    content: "Proof: ";
    font-style: italic;
}
"""
        
        # Create temp CSS file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.css',
            delete=False,
            encoding='utf-8'
        ) as css_file:
            css_file.write(css_content)
            return css_file.name
    
    def _enhance_html(self, html_content: str) -> str:
        """Add additional enhancements to HTML."""
        # Add viewport meta tag for mobile responsiveness
        viewport_meta = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        
        if '<head>' in html_content and viewport_meta not in html_content:
            html_content = html_content.replace('<head>', f'<head>\n{viewport_meta}')
        
        # Wrap body content in container if not already wrapped
        if '<body>' in html_content and '<div class="container">' not in html_content:
            html_content = html_content.replace(
                '<body>',
                '<body>\n<div class="container">'
            )
            html_content = html_content.replace(
                '</body>',
                '</div>\n</body>'
            )
        
        return html_content
    
    def convert_latex_with_timing(
        self,
        latex_content: str,
        timing_stats: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Convert LaTeX to HTML and optionally add timing information.
        
        Args:
            latex_content: LaTeX document content
            timing_stats: Optional timing statistics to append
            output_path: Path for output HTML file
            
        Returns:
            Tuple of (success: bool, html_content: str or None)
        """
        # If timing stats provided, append to LaTeX before conversion
        if timing_stats:
            timing_section = self._create_timing_latex(timing_stats)
            # Insert before \end{document}
            latex_content = latex_content.replace(
                '\\end{document}',
                fr'\n{timing_section}\n\end{{document}}'
            )
        
        return self.latex_to_html(latex_content, output_path)
    
    def _create_timing_latex(self, timing_stats: Dict[str, Any]) -> str:
        """Create LaTeX section for timing statistics."""
        latex_lines = [
            '\\section*{Generation Performance Metrics}',
            '',
            '\\begin{itemize}',
            f'  \\item \\textbf{{Total Generation Time:}} {timing_stats.get("total_formatted", "N/A")}',
            f'  \\item \\textbf{{Average Time per Question:}} {timing_stats.get("avg_time_per_question", "N/A")}',
        ]
        
        if 'stages_formatted' in timing_stats:
            latex_lines.append('  \\item \\textbf{Stage-wise Breakdown:}')
            latex_lines.append('  \\begin{enumerate}')
            for stage, duration in timing_stats['stages_formatted'].items():
                latex_lines.append(f'    \\item {stage}: {duration}')
            latex_lines.append('  \\end{enumerate}')
        
        latex_lines.extend([
            '\\end{itemize}',
            ''
        ])
        
        return '\n'.join(latex_lines)


def convert_latex_to_html(
    latex_content: str,
    output_path: Optional[str] = None,
    timing_stats: Optional[Dict[str, Any]] = None
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to convert LaTeX to HTML.
    
    Args:
        latex_content: LaTeX document content
        output_path: Optional output path
        timing_stats: Optional timing statistics
        
    Returns:
        Tuple of (success: bool, html_content: str or None)
    """
    converter = LaTeXToHTMLConverter()
    
    if timing_stats:
        return converter.convert_latex_with_timing(
            latex_content,
            timing_stats,
            output_path
        )
    else:
        return converter.latex_to_html(latex_content, output_path)