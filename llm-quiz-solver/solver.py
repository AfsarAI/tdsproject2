import logging
import time
import json
import re
import requests
import io
from playwright.sync_api import sync_playwright
from PyPDF2 import PdfReader
import pandas as pd

logger = logging.getLogger(__name__)

class QuizSolver:
    def __init__(self, secret):
        self.secret = secret

    def solve(self, start_url, email):
        current_url = start_url
        
        while current_url:
            logger.info(f"Visiting: {current_url}")
            try:
                result = self._process_single_quiz(current_url, email)
                if not result:
                    logger.info("Quiz processing finished or failed.")
                    break
                
                # Check if we got a new URL to visit
                if result.get("correct") and result.get("url"):
                    current_url = result.get("url")
                    logger.info(f"Correct! Proceeding to next URL: {current_url}")
                elif not result.get("correct"):
                    logger.warning(f"Incorrect answer. Reason: {result.get('reason')}")
                    # In a real scenario, we might want to retry or stop. 
                    # For now, we stop to avoid infinite loops if logic is flawed.
                    break
                else:
                    logger.info("Quiz completed successfully!")
                    break
            except Exception as e:
                logger.error(f"Error processing quiz at {current_url}: {e}")
                break

    def _process_single_quiz(self, url, email):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")

                # 1. Extract Instructions and Data
                content = page.content()
                text_content = page.inner_text("body")
                
                # Check for PDF links
                pdf_link = None
                links = page.query_selector_all("a")
                for link in links:
                    href = link.get_attribute("href")
                    if href and href.lower().endswith(".pdf"):
                        pdf_link = href
                        break
                
                # Check for submission URL
                submit_url = None
                submit_match = re.search(r'https?://[^\s"<]+/submit', text_content)
                if not submit_match:
                    submit_match = re.search(r'https?://[^\s"<]+/submit', content)
                
                if submit_match:
                    submit_url = submit_match.group(0)
                
                if not submit_url:
                    for link in links:
                        href = link.get_attribute("href")
                        if href and "submit" in href:
                            submit_url = href
                            break
                
                if submit_url and not submit_url.startswith("http"):
                    from urllib.parse import urljoin
                    submit_url = urljoin(url, submit_url)
                
                if not submit_url:
                    code_blocks = page.query_selector_all("pre, code")
                    for block in code_blocks:
                        block_text = block.inner_text()
                        if "submit" in block_text:
                            match = re.search(r'https?://[^\s"<]+/submit', block_text)
                            if match:
                                submit_url = match.group(0)
                                break
    
                if not submit_url:
                    logger.error("Could not find submit URL.")
                    return None
    
                # 2. Solve the Task
                answer = None
                
                # Case 1: PDF Task
                if pdf_link:
                    if not pdf_link.startswith("http"):
                        from urllib.parse import urljoin
                        pdf_link = urljoin(url, pdf_link)
                            
                    logger.info(f"Found PDF link: {pdf_link}")
                    pdf_content = requests.get(pdf_link).content
                    
                    if "sum" in text_content.lower() and "value" in text_content.lower():
                        answer = self._solve_pdf_sum_task(pdf_content)
                    else:
                        answer = self._extract_text_from_pdf(pdf_content)
                        if answer is None:
                            answer = 0 
    
                # Case 2: Simple Math or Text extraction
                elif "sum" in text_content.lower() or "calculate" in text_content.lower():
                    # Placeholder - expand if needed
                    pass
                
                # Check for Audio Task
                if "audio" in url or "mp3" in content:
                    logger.info("Detected Audio Task.")
                    logger.info(f"Page text content: {text_content}")
                    # Check for audio tag
                    audio_src = page.get_attribute("audio", "src")
                    if audio_src:
                         logger.info(f"Found audio source: {audio_src}")
                    
                    # Log all links
                    csv_link = None
                    all_links = page.query_selector_all("a")
                    for l in all_links:
                        href = l.get_attribute("href")
                        logger.info(f"Link: {href}")
                        if href and href.lower().endswith(".csv"):
                            csv_link = href
                    
                    if csv_link:
                        if not csv_link.startswith("http"):
                            from urllib.parse import urljoin
                            csv_link = urljoin(url, csv_link)
                        
                        logger.info(f"Found CSV link: {csv_link}")
                        csv_resp = requests.get(csv_link)
                        if csv_resp.status_code == 200:
                            # Parse CSV
                            import io
                            import pandas as pd
                            df = pd.read_csv(io.StringIO(csv_resp.text), header=None)
                            logger.info(f"CSV Head:\n{df.head()}")
                            
                            # Check for cutoff in text
                            cutoff = 0
                            cutoff_match = re.search(r'Cutoff:\s*(\d+)', text_content)
                            if cutoff_match:
                                cutoff = int(cutoff_match.group(1))
                                logger.info(f"Found cutoff: {cutoff}")
                            
                            # Sum numbers > cutoff
                            # Assuming the CSV has a column with numbers. Let's try to find a numeric column.
                            # If multiple, maybe sum all? Or look for specific column?
                            # Heuristic: Sum the last column if numeric? Or all numeric cells?
                            # Let's sum all numeric values in the dataframe > cutoff
                            
                            total_sum = 0
                            for col in df.select_dtypes(include=['number']).columns:
                                total_sum += df[col][df[col] > cutoff].sum()
                            
                            answer = int(total_sum) # Assuming integer answer?
                            logger.info(f"Computed CSV sum: {answer}")
                        else:
                            logger.error(f"Failed to download CSV: {csv_resp.status_code}")

                
                # Case 3: Scraping Task
                scrape_match = re.search(r'Scrape\s+([^\s]+)', text_content)
                if scrape_match:
                    scrape_path = scrape_match.group(1)
                    if not scrape_path.startswith("http"):
                        from urllib.parse import urljoin
                        scrape_url = urljoin(url, scrape_path)
                    else:
                        scrape_url = scrape_path
                    logger.info(f"Found scraping task: {scrape_url}")
                    try:
                        # Use requests first
                        scrape_resp = requests.get(scrape_url, timeout=30)
                        if scrape_resp.status_code == 200:
                            scraped_data = scrape_resp.text.strip()
                            logger.info(f"Initial scraped data (len={len(scraped_data)}): {scraped_data[:200]}...")
                            
                            # Check if it looks like HTML/JS
                            if "<html" in scraped_data.lower() or "<div" in scraped_data.lower() or "<script" in scraped_data.lower():
                                logger.info("Scraped data looks like HTML/JS, using Playwright...")
                                # Use a new page or the current one? Current one is busy. Use a new one.
                                # But we are in a context manager, so we can make a new page.
                                scrape_page = browser.new_page()
                                try:
                                    scrape_page.goto(scrape_url, timeout=30000)
                                    scrape_page.wait_for_load_state("networkidle")
                                    # Try to get the text content of the body or a specific element if known
                                    # For the demo, it seems the data is just text in the body or a specific div?
                                    # Let's try getting inner_text of body first.
                                    scraped_data = scrape_page.inner_text("body").strip()
                                finally:
                                    scrape_page.close()
                            
                            logger.info(f"Scraped data: {scraped_data}")
                            if answer is None: # Only set answer if not already found by other means
                                answer = scraped_data
                        else:
                            logger.error(f"Failed to scrape {scrape_url}: {scrape_resp.status_code}")
                    except Exception as e:
                        logger.error(f"Error scraping: {e}")
    
                # Try to parse any JSON embedded in page text
                try:
                    start_idx = text_content.find('{')
                    end_idx = text_content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = text_content[start_idx:end_idx+1]
                        json_data = json.loads(json_str)
                        if isinstance(json_data, dict) and "answer" in json_data:
                            # Only use JSON answer if we haven't found one yet (e.g. from scraping)
                            if answer is None:
                                answer = json_data["answer"]
                                logger.info(f"Found answer in JSON: {answer}")
                            else:
                                logger.info(f"Ignoring answer in JSON ({json_data['answer']}) because we already have an answer: {answer}")
                except json.JSONDecodeError:
                    pass
    
                if answer is None:
                    answer = 0
    
                logger.info(f"Computed answer: {answer}")
    
                # 3. Submit Answer WITH retry-for-secret-mismatch
                payload = {
                    "email": email,
                    "secret": self.secret,
                    "url": url,
                    "answer": answer
                }
                
                # helper to safely parse response
                def _parse_resp(resp):
                    try:
                        return resp.status_code, resp.json()
                    except Exception:
                        return resp.status_code, {"correct": False, "reason": f"Invalid JSON: {resp.text}"}
    
                logger.info(f"Submitting to {submit_url} with payload: {payload}")
                response = requests.post(submit_url, json=payload, timeout=30)
                status_code, resp_json = _parse_resp(response)
    
                if status_code == 200:
                    # If incorrect and reason mentions secret mismatch, try resubmitting with secret=answer
                    if not resp_json.get("correct", False):
                        reason = (resp_json.get("reason") or "").lower()
                        if "secret mismatch" in reason and answer:
                            logger.info("Server reported secret mismatch — retrying submit with secret set to the computed answer.")
                            payload_retry = payload.copy()
                            
                            # Try to extract a 5-digit number from the answer
                            secret_match = re.search(r'\b\d{5}\b', str(answer))
                            if secret_match:
                                extracted_secret = secret_match.group(0)
                                logger.info(f"Extracted potential secret {extracted_secret} from answer.")
                                payload_retry["secret"] = extracted_secret
                                # Also try setting the answer to the secret, in case that's what is expected
                                payload_retry["answer"] = extracted_secret
                            else:
                                # Fallback to using the whole answer as secret
                                payload_retry["secret"] = str(answer)
                                
                            logger.info(f"Retry payload: {payload_retry}")
                            resp2 = requests.post(submit_url, json=payload_retry, timeout=30)
                            status2, json2 = _parse_resp(resp2)
                            logger.info(f"Retry response: {status2} {json2}")
                            return json2
                    # Return original response JSON
                    return resp_json
                else:
                    logger.error(f"Submission failed: {status_code} - {response.text}")
                    return {"correct": False, "reason": f"HTTP {status_code}: {response.text}"}
    
            except Exception as e:
                logger.error(f"Error in Playwright session: {e}")
                return None
            finally:
                browser.close()


    def _solve_pdf_sum_task(self, pdf_bytes):
        """
        Extracts text from PDF, looks for a table-like structure or just numbers associated with 'value'.
        This is a simplified solver for the demo.
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            # The demo asks for page 2 usually, but let's check all or specific page if parsed from text.
            # For robustness, let's look at all pages.
            
            all_text = ""
            for page in reader.pages:
                all_text += page.extract_text() + "\n"
            
            # Heuristic: look for lines with numbers.
            # If it's a table, lines might look like "Item Name 12345"
            # We want to sum the "value" column.
            
            # Very naive extraction: find all numbers and sum them? No, that's dangerous.
            # Let's try to find a pattern.
            
            # If we had `tabula-py` or `camelot`, we could extract tables. 
            # Since we only have `pypdf2`, we rely on text.
            
            lines = all_text.split('\n')
            total_sum = 0
            for line in lines:
                # Look for numbers at the end of the line (common in simple tables)
                # e.g. "Item A   100"
                parts = line.split()
                if not parts:
                    continue
                
                # Check if last part is a number
                try:
                    val = float(parts[-1].replace(',', '').replace('$', ''))
                    # Heuristic: ignore small integers that might be page numbers or indices if they are sequential?
                    # For now, just sum everything that looks like a value row.
                    total_sum += val
                except ValueError:
                    continue
            
            return total_sum
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return 0

    def _extract_text_from_pdf(self, pdf_bytes):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
