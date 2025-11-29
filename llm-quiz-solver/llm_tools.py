import logging
import json
import base64
import io
import requests
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

import threading

class ToolContext(threading.local):
    """Holds the context for the tools (browser page, etc.) thread-locally."""
    def __init__(self):
        self.page = None

# Thread-local context
context = ToolContext()

def set_page(page):
    context.page = page

@tool
def get_page_content(include_links: bool = True) -> str:
    """
    Returns the text content of the current web page. 
    Useful to understand the quiz question and available data.
    """
    if not context.page:
        return "Error: No active browser page."
    
    try:
        # Get text content
        text = context.page.inner_text("body")
        
        # Get links if requested
        links_summary = ""
        if include_links:
            links = context.page.query_selector_all("a")
            links_data = []
            for link in links:
                href = link.get_attribute("href")
                text_content = link.inner_text().strip()
                if href:
                    links_data.append(f"[{text_content}]({href})")
            if links_data:
                links_summary = "\n\nLinks found:\n" + "\n".join(links_data)
        
        return text + links_summary
    except Exception as e:
        return f"Error getting page content: {e}"

@tool
def navigate_to_url(url: str) -> str:
    """
    Navigates the browser to the specified URL.
    Use this to visit links found on the page.
    """
    if not context.page:
        return "Error: No active browser page."
    
    try:
        logger.info(f"Navigating to: {url}")
        context.page.goto(url, timeout=60000)
        context.page.wait_for_load_state("networkidle")
        return f"Successfully navigated to {url}"
    except Exception as e:
        return f"Error navigating to {url}: {e}"

@tool
def download_file(url: str) -> str:
    """
    Downloads a file from a URL and returns its content as a string (if text) or summary (if binary).
    For PDFs, it extracts text.
    """
    try:
        # Handle relative URLs
        if not url.startswith("http"):
            # We might need the current URL to resolve relative paths, 
            # but for now let's assume the agent passes full URLs or we try to resolve if context.page.url is available
            if context.page:
                from urllib.parse import urljoin
                url = urljoin(context.page.url, url)
        
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            return f"Error: Failed to download {url}, status code {response.status_code}"
            
        content_type = response.headers.get("Content-Type", "").lower()
        
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            import io
            try:
                reader = PdfReader(io.BytesIO(response.content))
                text = []
                for p in reader.pages:
                    text.append(p.extract_text() or "")
                return f"PDF Content:\n{' '.join(text)}"
            except Exception as e:
                return f"Error reading PDF: {e}"
        
        elif "csv" in content_type or url.lower().endswith(".csv"):
             # Save to file to avoid flooding context and logs
             filename = "downloaded_data.csv"
             try:
                 with open(filename, "w", encoding="utf-8") as f:
                     f.write(response.text)
                 
                 lines = response.text.splitlines()
                 head = "\n".join(lines[:5])
                 return f"CSV content saved to '{filename}'. You MUST write Python code to read this file.\nFirst 5 lines:\n{head}\n...({len(lines)} lines total)"
             except Exception as e:
                 return f"Error saving CSV: {e}"
             
        elif "text" in content_type or "json" in content_type:
            return response.text
            
        else:
            return f"Downloaded binary file of size {len(response.content)} bytes. Content-Type: {content_type}"
            
    except Exception as e:
        return f"Error downloading file: {e}"

@tool
def run_python_code(code: str) -> str:
    """
    Executes the given Python code and returns the output (stdout).
    The code can use pandas, requests, json, re, math.
    The code should print the result to stdout.
    """
    import sys
    import io
    import pandas as pd
    import numpy as np
    import requests
    import json
    import re
    import math
    
    # Capture stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Wrap code to ensure it runs
        exec_globals = {
            "pd": pd,
            "np": np,
            "requests": requests,
            "json": json,
            "re": re,
            "math": math
        }
        exec(code, exec_globals)
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code: {e}"

class SubmitAnswerSchema(BaseModel):
    answer: Any = Field(description="The final answer to the quiz question.")

@tool(args_schema=SubmitAnswerSchema, return_direct=True)
def submit_answer(answer: Any) -> str:
    """
    Submits the final answer to the quiz. 
    This tool should be called when you have calculated the answer.
    """
    # This tool is special; it signals the agent loop to stop and submit.
    # In our implementation, we might just return the answer string and let the loop handle submission,
    # or we can store it in context.
    return f"FINAL_ANSWER:{answer}"

ALL_TOOLS = [get_page_content, navigate_to_url, download_file, run_python_code, submit_answer]
