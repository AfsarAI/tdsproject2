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

logger = logging.getLogger(__name__)

class QuizSolver:
    def __init__(self, secret):
        self.secret = secret
        # Initialize LLM
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True # Sometimes needed for Gemini
        )
        
        self.tools = ALL_TOOLS

    def solve(self, start_url, email):
        current_url = start_url
        
        while current_url:
            logger.info(f"Visiting: {current_url}")
            try:
                result = self._process_single_quiz_agent(current_url, email)
                if not result:
                    logger.info("Quiz processing finished or failed.")
                    break
                
                # Check if we got a new URL to visit
                if result.get("correct") and result.get("url"):
                    current_url = result.get("url")
                    logger.info(f"Correct! Proceeding to next URL: {current_url}")
                elif not result.get("correct"):
                    logger.warning(f"Incorrect answer. Reason: {result.get('reason')}")
                    # Stop to avoid infinite loops
                    break
                else:
                    logger.info("Quiz completed successfully!")
                    break
            except Exception as e:
                logger.error(f"Error processing quiz at {current_url}: {e}")
                break

    def _process_single_quiz_agent(self, url, email):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")
                
                # Set page context for tools
                set_page(page)
                
                # Define Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an autonomous quiz-solving agent.
Your goal is to solve the quiz at the current URL.
You have access to a browser (via tools) to read the page, navigate, and download files.
You can also run Python code to analyze data.

Your specific instructions:
1. Read the page content to understand the question.
2. If there are links to files (PDF, CSV, etc.), download and analyze them.
3. If you need to scrape another page, use the navigation or download tools.
4. Calculate the answer.
5. Call the `submit_answer` tool with the final answer.

Context:
- User Email: {email}
- Quiz Secret: {self.secret}
- Current Quiz URL: {url}

Do NOT submit the answer via HTTP POST yourself. Just call `submit_answer` with the value.
If you cannot solve it, try your best to guess or return a reasonable failure message.
"""),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                # Create Agent
                agent = create_tool_calling_agent(self.llm, self.tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                
                # Run Agent
                logger.info("Starting Agent execution...")
                response = agent_executor.invoke({"input": "Solve the quiz on the current page."})
                output = response["output"]
                
                # Extract answer from output or tool calls
                # The agent might have called submit_answer, which returns "FINAL_ANSWER:..."
                # Or the output itself might contain it if we parse the intermediate steps (which AgentExecutor does but hides by default unless we check intermediate_steps)
                # But since we defined submit_answer as a tool, the agent *should* have called it.
                # However, AgentExecutor returns the string response of the agent.
                
                # Let's look for the answer in the tool execution logs or force the agent to return it.
                # A better way with AgentExecutor is to parse the 'output' if the agent just says it, 
                # but we want the specific value passed to submit_answer.
                
                # Actually, since we are inside the loop, we can't easily intercept the tool call return value 
                # unless we use a callback or parse the full log.
                # SIMPLIFICATION: We will parse the 'output' string for "FINAL_ANSWER:..." 
                # because our submit_answer tool returns that string, and the agent usually repeats it or it becomes the final observation.
                
                answer = None
                if "FINAL_ANSWER:" in output:
                    answer = output.split("FINAL_ANSWER:")[1].strip()
                else:
                    # Fallback: try to find it in the text
                    # Or maybe the agent didn't call the tool?
                    logger.warning("Agent did not return FINAL_ANSWER. Output: " + output)
                    # Try to extract anything that looks like an answer
                    answer = output.strip()

                logger.info(f"Agent determined answer: {answer}")
                
                # Submit the answer
                # We need to find the submission URL from the page first (Agent could have done this, but let's do it reliably here or ask agent to return it?)
                # Reliable way: Re-use the logic to find submit URL from the page, or ask agent to find it.
                # Let's stick to the robust regex logic for finding submit URL, as it's purely mechanical.
                
                content = page.content()
                text_content = page.inner_text("body")
                submit_url = self._find_submit_url(page, url, text_content, content)
                
                if not submit_url:
                    logger.error("Could not find submit URL.")
                    return None
                
                return self._submit_answer(submit_url, email, answer, url)

            except Exception as e:
                logger.error(f"Error in Agent session: {e}")
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

