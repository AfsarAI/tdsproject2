from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, encode_image_to_base64
)
from tools.multimodal_tools import analyze_audio_with_llm, analyze_image_with_llm
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
import json
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


# -------------------------------------------------
# STATE
# -------------------------------------------------
# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    url: str
    start_time: float

TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, encode_image_to_base64,
    analyze_audio_with_llm, analyze_image_with_llm
]


# -------------------------------------------------
# LLM INIT
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- **MODALITY DETECTION & TOOL USAGE (CRITICAL)**:
    - **AUDIO**: If you see an audio file (mp3, wav, opus, ogg) or an audio tag:
        1. Call `download_file` with the audio URL.
        2. Call `analyze_audio_with_llm` with the saved filename.
        3. Use the analysis/transcription to answer.
    - **IMAGE**: If you see an image that requires analysis:
        1. Call `download_file` (if needed).
        2. Call `analyze_image_with_llm` with the saved filename.
    - **CODE**: If you need to run Python code, use `run_code`.
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, multimodal analysis, or running code.
- If no specific answer is required or found in the instructions, submit "start" or "attempt" as the answer. NEVER submit an empty string or null.
- Include:
    email = {EMAIL}
    secret = {SECRET}
"""


# -------------------------------------------------
# NEW NODE: HANDLE MALFORMED JSON
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    """
    If the LLM generates invalid JSON, this node sends a correction message
    so the LLM can try again.
    """
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user", 
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Please rewrite the code and try again. Ensure you escape newlines and quotes correctly inside the JSON."
            }
        ]
    }


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    # --- STATE UPDATE FROM TOOL OUTPUT ---
    # Check if the last message was a post_request tool output that contains a new URL
    if state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, ToolMessage) and last_msg.name == "post_request":
            try:
                content = json.loads(last_msg.content)
                if isinstance(content, dict):
                    new_url = content.get("url")
                    if new_url and new_url != state.get("url"):
                        print(f"Updating URL state to: {new_url}")
                        state["url"] = new_url
            except Exception as e:
                print(f"Failed to parse tool output for URL update: {e}")

    # --- TIME HANDLING START ---
    cur_time = time.time()
    cur_url = state.get("url")
    start_time = state.get("start_time", cur_time)
    
    # SAFE GET: Prevents crash if url is None or not in dict
    prev_time = url_time.get(cur_url) 
    
    # We use the start_time from state to track total duration if needed, 
    # but the original logic used url_time for per-url tracking.
    # We'll keep using url_time for now but it's still global. 
    # Ideally url_time should be in state too, but it's imported from shared_store.
    # For now, we fix the cur_url retrieval.

    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time

        if diff >= 180:
            print(f"Timeout exceeded ({diff}s) — instructing LLM to purposely submit wrong answer.")

            fail_instruction = """
            You have exceeded the time limit for this task (over 180 seconds).
            Immediately call the `post_request` tool and submit a WRONG answer for the CURRENT quiz.
            """

            # Using HumanMessage (as you correctly implemented)
            fail_msg = HumanMessage(content=fail_instruction)

            # We invoke the LLM immediately with this new instruction
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result], "url": cur_url}
    # --- TIME HANDLING END ---

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm, 
    )
    
    # Better check: Does it have a HumanMessage?
    has_human = any(msg.type == "human" for msg in trimmed_messages)
    
    if not has_human:
        print("WARNING: Context was trimmed too far. Injecting state reminder.")
        # We remind the agent of the current URL from the state
        current_url = state.get("url", "Unknown URL")
        reminder = HumanMessage(content=f"Context cleared due to length. Continue processing URL: {current_url}")
        
        # We append this to the trimmed list (temporarily for this invoke)
        trimmed_messages.append(reminder)
    # ----------------------------------------

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    
    result = llm.invoke(trimmed_messages)

    return {"messages": [result], "url": state.get("url")}


# -------------------------------------------------
# ROUTE LOGIC (UPDATED FOR MALFORMED CALLS)
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    
    # 1. CHECK FOR MALFORMED FUNCTION CALLS
    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    # 2. CHECK FOR VALID TOOLS
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"

    # 3. CHECK FOR END
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and len(content) and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("Route → agent")
    return "agent"


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

# Add Nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node) # Add the repair node

# Add Edges
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent") # Retry loop

# Conditional Edges
graph.add_conditional_edges(
    "agent", 
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed", # Map the new route
        END: END
    }
)

app = graph.compile()


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    # system message is seeded ONCE here
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    app.invoke(
        {
            "messages": initial_messages,
            "url": url,
            "start_time": time.time()
        },
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("Tasks completed successfully!")