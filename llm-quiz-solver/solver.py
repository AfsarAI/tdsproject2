import logging
import time
import json
import re
import requests
import os
from playwright.sync_api import sync_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from llm_tools import ALL_TOOLS, set_page
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Configure Rich Console
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "url": "bold blue underline"
})
console = Console(theme=custom_theme)

# Configure logging to use Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("quiz_solver")

class QuizSolver:
    def __init__(self, secret):
        self.secret = secret
        # Initialize LLM
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[error]GEMINI_API_KEY not found in environment variables.[/error]")
            raise ValueError("GEMINI_API_KEY is required.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        self.tools = ALL_TOOLS

    def solve(self, start_url, email):
        current_url = start_url
        
        console.print(f"[bold magenta]Starting Quiz Solver for {email}[/bold magenta]")
        
        while current_url:
            console.print(f"\n[bold green]We send you to url:[/bold green] [url]{current_url}[/url]")
            try:
                result = self._process_single_quiz_agent(current_url, email)
                if not result:
                    console.print("[warning]Quiz processing finished or failed (no result returned).[/warning]")
                    break
                
                # Check if we got a new URL to visit
                if result.get("correct") and result.get("url"):
                    next_url = result.get("url")
                    console.print(f"[bold green]You solve it correctly. You get url:[/bold green] [url]{next_url}[/url]")
                    current_url = next_url
                elif not result.get("correct"):
                    reason = result.get('reason', 'Unknown')
                    console.print(f"[error]You solve it wrongly.[/error] Reason: {reason}")
                    # Stop to avoid infinite loops
                    break
                else:
                    console.print("[bold green]You solve it correctly and get no new URL, ending the quiz.[/bold green]")
                    break
            except Exception as e:
                console.print(f"[error]Error processing quiz at {current_url}: {e}[/error]")
                break

    def _get_llm(self, provider="gemini"):
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return None
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True
            )
        elif provider == "aipipe":
            api_key = os.environ.get("AIPIPE_API_KEY")
            if not api_key:
                return None
            # AIPIPE uses OpenAI-compatible endpoint
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="google/gemini-2.0-flash-lite-001", # Default for AIPIPE
                api_key=api_key,
                base_url="https://aipipe.org/openrouter/v1",
                temperature=0,
                default_headers={"HTTP-Referer": "http://localhost:5000", "X-Title": "LLM Quiz Solver"}
            )
        return None

    def _process_single_quiz_agent(self, url, email):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")
                
                # Set page context for tools
                set_page(page)

                # Try to find submit URL *before* agent runs (in case agent navigates away)
                content = page.content()
                text_content = page.inner_text("body")
                initial_submit_url = self._find_submit_url(page, url, text_content, content)
                
                # Define Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an autonomous quiz-solving agent.
Your goal is to solve the quiz at the current URL.
You have access to a browser (via tools) to read the page, navigate, and download files.
You can also run Python code to analyze data.

Your specific instructions:
1. **Read the page content** to understand the question.
2. **Analyze the task**:
   - If it involves a file (PDF, CSV), download it. **IMPORTANT**: The `download_file` tool will save the file to disk and return the filename. You MUST then use `run_python_code` to read that file (e.g., `pandas.read_csv('downloaded_data.csv')`) and calculate the answer. Do NOT try to read the file content directly from the tool output.
   - If it involves scraping, navigate to the target pages.
   - If it involves visualization, generate the chart code.
3. **Calculate the answer**.
4. **Call the `submit_answer` tool** with the final answer.

**CRITICAL RULES**:
- **Time Limit**: You must be efficient. Do not loop unnecessarily.
- **Answer Format**: The answer can be a string, number, or JSON.
- **Submission**: Do NOT submit via HTTP POST yourself. ALWAYS use the `submit_answer` tool.
- **Output Noise**: Do NOT print large amounts of data (like whole CSVs or long lists) to stdout. Use `head()` or summary statistics if you need to inspect data.
- **Code Execution**: When using Python, always print the result so you can see it in the logs.

Context:
- User Email: {email}
- Quiz Secret: {self.secret}
- Current Quiz URL: {url}
"""),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                # Try Gemini first, then AIPIPE
                llm = self._get_llm("gemini")
                if not llm:
                    console.print("[warning]Gemini API Key missing, trying AIPIPE...[/warning]")
                    llm = self._get_llm("aipipe")
                
                if not llm:
                    console.print("[error]No valid LLM provider found (Gemini or AIPIPE).[/error]")
                    return None

                # Create Agent
                agent = create_tool_calling_agent(llm, self.tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                
                # Run Agent
                console.print("[info]Agent is thinking...[/info]")
                try:
                    response = agent_executor.invoke({"input": "Solve the quiz on the current page."})
                except Exception as e:
                    console.print(f"[error]Agent failed with primary LLM: {e}. Trying fallback...[/error]")
                    # Fallback logic
                    if isinstance(llm, ChatGoogleGenerativeAI):
                        fallback_llm = self._get_llm("aipipe")
                        if fallback_llm:
                            console.print("[info]Switching to AIPIPE...[/info]")
                            agent = create_tool_calling_agent(fallback_llm, self.tools, prompt)
                            agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                            response = agent_executor.invoke({"input": "Solve the quiz on the current page."})
                        else:
                            raise e
                    else:
                        raise e

                output = response["output"]
                
                # Extract answer from output
                answer = None
                if "FINAL_ANSWER:" in output:
                    answer = output.split("FINAL_ANSWER:")[1].strip()
                else:
                    # Fallback
                    console.print(f"[warning]Agent did not return FINAL_ANSWER tag. Using raw output.[/warning]")
                    answer = output.strip()

                console.print(f"[bold cyan]Agent determined answer:[/bold cyan] {answer}")
                
                # Submit the answer
                # Use initial_submit_url if available, otherwise try to find it again (though we might be on a different page now)
                submit_url = initial_submit_url
                if not submit_url:
                    # Try finding it on current page (maybe we navigated to the submission page?)
                    content = page.content()
                    text_content = page.inner_text("body")
                    submit_url = self._find_submit_url(page, url, text_content, content)
                
                if not submit_url:
                    console.print("[error]Could not find submit URL.[/error]")
                    return None
                
                return self._submit_answer(submit_url, email, answer, url)

            except Exception as e:
                console.print(f"[error]Error in Agent session: {e}[/error]")
                return None
            finally:
                browser.close()

    def _find_submit_url(self, page, base_url, text_content, html_content):
        # Logic to find submit URL (copied/adapted from original)
        submit_match = re.search(r'https?://[^\s"<]+/submit', text_content)
        if not submit_match:
            submit_match = re.search(r'https?://[^\s"<]+/submit', html_content)
        
        if submit_match:
            return submit_match.group(0)
            
        links = page.query_selector_all("a")
        for link in links:
            href = link.get_attribute("href")
            if href and "submit" in href:
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    return urljoin(base_url, href)
                return href
        return None

    def _submit_answer(self, submit_url, email, answer, original_url):
        payload = {
            "email": email,
            "secret": self.secret,
            "url": original_url,
            "answer": answer
        }
        
        logger.info(f"Submitting to {submit_url} with payload: {payload}")
        try:
            response = requests.post(submit_url, json=payload, timeout=30)
            try:
                resp_json = response.json()
            except:
                resp_json = {"correct": False, "reason": f"Invalid JSON: {response.text}"}
            
            # Handle secret mismatch retry logic (same as before)
            if response.status_code == 200 and not resp_json.get("correct", False):
                reason = (resp_json.get("reason") or "").lower()
                if "secret mismatch" in reason:
                     logger.info("Secret mismatch detected. Retrying with answer as secret...")
                     # Try to extract 5-digit secret
                     import re
                     secret_match = re.search(r'\b\d{5}\b', str(answer))
                     new_secret = secret_match.group(0) if secret_match else str(answer)
                     
                     payload["secret"] = new_secret
                     payload["answer"] = new_secret # Sometimes the answer IS the secret
                     
                     resp2 = requests.post(submit_url, json=payload, timeout=30)
                     return resp2.json()
            
            return resp_json
            
        except Exception as e:
            logger.error(f"Submission error: {e}")
            return None

