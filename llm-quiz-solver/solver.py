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
            # User provided specific key
            api_key = os.environ.get("AIPIPE_API_KEY") or "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDIwMjNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.FsZQpaXV3lzcIPYelSZ5ZH83MFBb-DUq86sOidLPJrI"
            
            # AIPIPE uses OpenAI-compatible endpoint
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini", # User requested GPT model
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
                    ("system", f"""You are an intelligent agent built to solve a specific quiz.
You have access to a web browser and python code execution tools.

YOUR GOAL:
1.  Understand the task on the current page.
2.  If the task requires data processing (e.g., from a CSV), DOWNLOAD the file and use PYTHON to process it.
3.  Submit the answer using the `submit_answer` tool.

CRITICAL INSTRUCTIONS FOR CSV TASKS:
- **READ THE QUESTION CAREFULLY**: Does it ask for the **COUNT** of items or the **SUM** of a column?
- **CHECK CONDITIONS**: Does it ask for items **greater than**, **less than**, or **equal to** a value?
- **FILTERING**: If the question says "sum of numbers > 5000", you MUST filter the data first (df[df[0] > 5000]) and THEN take the sum. Do NOT sum the entire column.
- **VERIFY**: Before submitting, double-check your logic. Did you calculate count instead of sum? Did you apply the filter?

ERROR HANDLING:
- If you receive an error like "Wrong sum of numbers", it means your calculation was incorrect.
- **DO NOT** just try the same code again.
- **ANALYZE**:
    - Did I calculate the sum of the *entire* column instead of the *filtered* values?
    - Did I calculate the *count* instead of the *sum*?
    - Did I read the wrong column?
- **CORRECT**: Adjust your Python code to fix the logic error and try again.

GENERAL:
- Use `get_page_content` to read instructions.
- Use `run_python_code` for ALL calculations. Do not do math in your head.
- Always return the final answer using the `submit_answer` tool.
- **IMPORTANT**: If the page explicitly says "Post your answer to [URL]", pass that URL to the `submit_answer` tool as the `submit_url` argument.
- **IMPORTANT**: If the page says "with url = [URL]" or "using url = [URL]", pass that URL to the `submit_answer` tool as the `task_url` argument.
    - `submit_url`: The HTTP endpoint to send the POST request TO. **Defaults to the initial submit URL. Only change this if the page explicitly says "Post to [URL]".**
- **IMPORTANT**: If the page says "with url = [URL]" or "using url = [URL]", pass that URL to the `submit_answer` tool as the `task_url` argument.
    - `submit_url`: The HTTP endpoint to send the POST request TO. **Defaults to the initial submit URL. Only change this if the page explicitly says "Post to [URL]".**
    - `task_url`: The value to put inside the JSON payload's "url" field. **Do NOT use this as the submit_url unless the page explicitly says "Post to [URL]".**
- **IMPORTANT**: If the page says "Start by POSTing..." or implies starting the quiz, and no specific question is asked, the answer is usually just an empty string "" or "start". Do NOT submit a long explanation.

Context:
- User Email: {email}
- Quiz Secret: {self.secret}
- Current Quiz URL: {url}
"""),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                # Try AIPIPE first (to avoid Gemini rate limits), then Gemini
                llm = self._get_llm("aipipe")
                if not llm:
                    console.print("[warning]AIPIPE API Key missing or failed, trying Gemini...[/warning]")
                    llm = self._get_llm("gemini")
                
                if not llm:
                    console.print("[error]No valid LLM provider found (AIPIPE or Gemini).[/error]")
                    return None

                # Create Agent
                agent = create_tool_calling_agent(llm, self.tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                
                # Run Agent with Retry Loop for Wrong Answers
                console.print("[info]Agent is thinking...[/info]")
                max_wrong_answer_retries = 3
                wrong_answer_count = 0
                current_input = "Solve the quiz on the current page."
                
                while wrong_answer_count < max_wrong_answer_retries:
                    # Execute Agent Chain
                    max_retries = 3
                    retry_count = 0
                    response = None
                    
                    while retry_count < max_retries:
                        try:
                            response = agent_executor.invoke({"input": current_input})
                            break # Success
                        except Exception as e:
                            error_str = str(e)
                            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                                retry_count += 1
                                wait_time = 15 * retry_count
                                console.print(f"[warning]Rate limit hit (429). Waiting {wait_time}s before retry {retry_count}/{max_retries}...[/warning]")
                                import time
                                time.sleep(wait_time)
                                if retry_count == max_retries:
                                    # Fallback logic (simplified for brevity, keeping existing logic structure)
                                    if isinstance(llm, ChatGoogleGenerativeAI):
                                        fallback_llm = self._get_llm("aipipe")
                                        if fallback_llm:
                                            console.print("[info]Switching to AIPIPE...[/info]")
                                            agent = create_tool_calling_agent(fallback_llm, self.tools, prompt)
                                            agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                                            try:
                                                response = agent_executor.invoke({"input": current_input})
                                                break
                                            except Exception as fallback_e:
                                                raise fallback_e
                                        else:
                                            raise e
                                    else:
                                        raise e
                            else:
                                # Immediate fallback logic
                                if isinstance(llm, ChatGoogleGenerativeAI):
                                    fallback_llm = self._get_llm("aipipe")
                                    if fallback_llm:
                                        console.print("[info]Switching to AIPIPE...[/info]")
                                        agent = create_tool_calling_agent(fallback_llm, self.tools, prompt)
                                        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                                        try:
                                            response = agent_executor.invoke({"input": current_input})
                                            break
                                        except Exception as fallback_e:
                                            raise fallback_e
                                    else:
                                        raise e
                                else:
                                    raise e

                    if not response:
                        console.print("[error]Agent failed to produce a response.[/error]")
                        return None

                    output = response["output"]
                    
                    # Extract answer, submit_url, and task_url from output
                    answer = None
                    agent_submit_url = None
                    agent_task_url = None
                    
                    if "FINAL_ANSWER:" in output:
                        final_part = output.split("FINAL_ANSWER:")[1].strip()
                        
                        # Parse optional parts
                        if "|SUBMIT_URL:" in final_part:
                            parts = final_part.split("|SUBMIT_URL:")
                            answer = parts[0].strip()
                            remaining = parts[1].strip()
                            if "|TASK_URL:" in remaining:
                                subparts = remaining.split("|TASK_URL:")
                                agent_submit_url = subparts[0].strip()
                                agent_task_url = subparts[1].strip()
                            else:
                                agent_submit_url = remaining
                        elif "|TASK_URL:" in final_part:
                            parts = final_part.split("|TASK_URL:")
                            answer = parts[0].strip()
                            agent_task_url = parts[1].strip()
                        else:
                            answer = final_part
                    else:
                        console.print(f"[warning]Agent did not return FINAL_ANSWER tag. Using raw output.[/warning]")
                        answer = output.strip()

                    console.print(f"[bold cyan]Agent determined answer:[/bold cyan] {answer}")
                    if agent_submit_url:
                        console.print(f"[bold cyan]Agent determined submit URL:[/bold cyan] {agent_submit_url}")
                    if agent_task_url:
                        console.print(f"[bold cyan]Agent determined task URL (for payload):[/bold cyan] {agent_task_url}")
                    
                    # Submit the answer
                    submit_url = agent_submit_url or initial_submit_url
                    if not submit_url:
                        content = page.content()
                        text_content = page.inner_text("body")
                        submit_url = self._find_submit_url(page, url, text_content, content)
                    
                    if not submit_url:
                        console.print("[error]Could not find submit URL.[/error]")
                        return None
                
                    # Use agent_task_url if provided, otherwise use original url
                    payload_url = agent_task_url if agent_task_url else url
                    submission_result = self._submit_answer(submit_url, email, answer, payload_url)
                    
                    console.print(f"[bold yellow]DEBUG: Submission Result: {submission_result}[/bold yellow]")

                    if not submission_result:
                        console.print("[error]Submission failed (no result).[/error]")
                        return None
                    elif submission_result.get("correct"):
                        return submission_result
                    else:
                        reason = submission_result.get("reason", "Unknown error") if submission_result else "Submission failed"
                        console.print(f"[error]Wrong answer submitted. Reason: {reason}[/error]")
                        wrong_answer_count += 1
                        if wrong_answer_count < max_wrong_answer_retries:
                            console.print(f"[info]Retrying with feedback ({wrong_answer_count}/{max_wrong_answer_retries})...[/info]")
                            current_input = f"Your previous answer '{answer}' was WRONG. The server responded: '{reason}'. Please analyze the task again, fix your mistake, and submit the correct answer."
                        else:
                            console.print("[error]Max wrong answer retries reached.[/error]")
                            return submission_result # Return the last (failed) result

            except Exception as e:
                console.print(f"[error]Error in Agent session: {e}[/error]")
                return None
            finally:
                browser.close()

    def _find_submit_url(self, page, base_url, text_content, html_content):
        # Logic to find submit URL
        # 1. Try strict regex for .../submit
        submit_match = re.search(r'https?://[^\s"<>]+?/submit\b', text_content)
        if not submit_match:
            submit_match = re.search(r'https?://[^\s"<>]+?/submit\b', html_content)
        
        if submit_match:
            return submit_match.group(0)

        # 2. Look for links with "submit" in href
        links = page.query_selector_all("a")
        for link in links:
            href = link.get_attribute("href")
            if href and "submit" in href:
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    return urljoin(base_url, href)
                return href
        
        # 3. Fallback: Look for ANY url in text that looks like a submit endpoint
        # e.g. https://tds-llm-analysis.s-anand.net/project2/submit
        fallback_match = re.search(r'https?://[^\s"<>]+?s-anand\.net/[^\s"<>]+', text_content)
        if fallback_match:
             # This is risky, but might be better than nothing if we are desperate
             pass

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
