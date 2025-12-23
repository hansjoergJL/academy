#!/usr/bin/env python3
"""
Convert Markdown to PDF for Academy documentation
"""

import markdown
import pdfkit
import sys

def markdown_to_pdf(md_path, pdf_path):
    """Convert Markdown file to PDF"""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'toc'])
    
    # Add CSS styling
    css_style = """
    <style>
    body { 
        font-family: Arial, sans-serif; 
        line-height: 1.6; 
        margin: 40px; 
        font-size: 12pt;
    }
    h1 { 
        color: #2c3e50; 
        border-bottom: 2px solid #3498db; 
        padding-bottom: 10px;
    }
    h2 { 
        color: #34495e; 
        border-bottom: 1px solid #ecf0f1; 
        padding-bottom: 5px;
    }
    h3 { color: #7f8c8d; }
    code { 
        background-color: #f8f9fa; 
        padding: 2px 4px; 
        border-radius: 3px; 
        font-family: 'Courier New', monospace;
    }
    pre { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 5px; 
        overflow-x: auto;
        border-left: 4px solid #3498db;
    }
    blockquote { 
        border-left: 4px solid #3498db; 
        margin-left: 0; 
        padding-left: 20px; 
        font-style: italic;
        color: #7f8c8d;
    }
    table { 
        border-collapse: collapse; 
        width: 100%; 
        margin: 20px 0;
    }
    th, td { 
        border: 1px solid #ddd; 
        padding: 12px; 
        text-align: left;
    }
    th { 
        background-color: #f2f2f2; 
        font-weight: bold;
    }
    .toc {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 30px;
        border-left: 4px solid #3498db;
    }
    </style>
    """
    
    # Combine HTML and CSS
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>LLM Fine-Tuning Leitfaden</title>
        {css_style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    try:
        pdfkit.from_string(full_html, pdf_path, options=options)
        print(f"✅ PDF erstellt: {pdf_path}")
        return True
    except Exception as e:
        print(f"❌ Fehler bei PDF-Erstellung: {e}")
        return False

if __name__ == "__main__":
    md_file = "/home/ad/dev/academy/data/raw/LLM_Fine_Tuning_Leitfaden.md"
    pdf_file = "/home/ad/dev/academy/data/raw/LLM_Fine_Tuning_Leitfaden.pdf"
    
    success = markdown_to_pdf(md_file, pdf_file)
    if success:
        print("📄 Markdown → PDF Konvertierung abgeschlossen")
        print(f"📍 Pfad: {pdf_file}")
    else:
        sys.exit(1)