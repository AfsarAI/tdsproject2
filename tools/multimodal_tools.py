import os
import base64
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize the model for multimodal analysis
# We use a separate instance or the same one, but we need to ensure it's configured for multimodal
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def encode_file_to_base64(file_path: str) -> str:
    """Helper to encode file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@tool
def analyze_audio_with_llm(file_path: str) -> str:
    """
    Analyze an audio file using the Gemini multimodal LLM.
    Use this tool to transcribe audio, extract passphrases, or answer questions about audio content.
    
    Args:
        file_path (str): The path to the audio file (e.g., 'LLMFiles/audio.mp3').
    """
    try:
        # Ensure path is correct
        if not os.path.exists(file_path):
             # Try prepending LLMFiles if not present
            if os.path.exists(os.path.join("LLMFiles", file_path)):
                file_path = os.path.join("LLMFiles", file_path)
            else:
                return f"Error: File {file_path} not found."

        b64_data = encode_file_to_base64(file_path)
        
        # Determine mime type based on extension
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = "audio/mpeg" # default
        if ext == ".wav":
            mime_type = "audio/wav"
        elif ext == ".mp3":
            mime_type = "audio/mp3"
        elif ext == ".opus":
            mime_type = "audio/ogg" # Gemini often handles opus as ogg or generic audio
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Listen to this audio carefully. Transcribe it exactly and extract any passphrases, codes, or specific instructions mentioned. If it's a passphrase, provide just the passphrase or the specific format requested."
                },
                {
                    "type": "media",
                    "mime_type": mime_type,
                    "data": b64_data
                }
            ]
        )
        
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error analyzing audio: {e}"

@tool
def analyze_image_with_llm(file_path: str) -> str:
    """
    Analyze an image file using the Gemini multimodal LLM.
    Use this tool to extract text (OCR), describe visual elements, or answer questions about the image.
    
    Args:
        file_path (str): The path to the image file.
    """
    try:
        # Ensure path is correct
        if not os.path.exists(file_path):
             # Try prepending LLMFiles if not present
            if os.path.exists(os.path.join("LLMFiles", file_path)):
                file_path = os.path.join("LLMFiles", file_path)
            else:
                return f"Error: File {file_path} not found."

        b64_data = encode_file_to_base64(file_path)
        
        # Determine mime type
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = "image/jpeg" # default
        if ext == ".png":
            mime_type = "image/png"
        elif ext == ".jpg" or ext == ".jpeg":
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
            
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this image. Extract all visible text and describe any visual elements relevant to the task. If there is a question or instruction in the image, answer it."
                },
                {
                    "type": "media",
                    "mime_type": mime_type,
                    "data": b64_data
                }
            ]
        )
        
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error analyzing image: {e}"
