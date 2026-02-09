import streamlit as st
import streamlit.components.v1
from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from groq import Groq
import os
import time
import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Safe import for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Suno Lyric Engineer (Hybrid)", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS FOR POLISH ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold;}
    .success-box { padding: 1rem; background-color: #d4edda; color: #155724; border-radius: 8px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION ---
load_dotenv()

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_key(name):
    """Safely retrieve API keys from secrets or environment variables."""
    try:
        key = st.secrets.get(name)
        if key:
            return key
    except Exception as e:
        logger.debug(f"Could not get key from secrets: {e}")
    
    key = os.getenv(name)
    if key:
        return key
    
    logger.warning(f"API key '{name}' not found in secrets or environment")
    return None

def validate_api_key(key, service_name):
    """Validate API key format and presence."""
    if not key:
        return False, f"{service_name} API key is missing"
    
    if len(key.strip()) < 10:
        return False, f"{service_name} API key appears to be invalid (too short)"
    
    return True, f"{service_name} API key validated"

GEMINI_KEY = get_key("GEMINI_API_KEY") or get_key("GOOGLE_API_KEY")  # Support both key names
GROQ_KEY = get_key("GROQ_API_KEY")

# Validate API keys
gemini_valid, gemini_msg = validate_api_key(GEMINI_KEY, "Gemini")
groq_valid, groq_msg = validate_api_key(GROQ_KEY, "Groq")

if not gemini_valid:
    st.error(f"⛔ **Setup Error:** {gemini_msg}")
    logger.error(gemini_msg)
    st.stop()

# Display API key status
if groq_valid:
    st.success(f"✅ Both API keys configured - Hybrid mode enabled")
    logger.info("Both API keys available - running in hybrid mode")
else:
    st.warning(f"⚠️ Groq API key missing - Running in Gemini-only mode")
    logger.warning(groq_msg)

# --- ENHANCED DUAL BRAIN ENGINE WITH ROBUST FALLBACK ---
class ModelInfo:
    """Data class for model information."""
    def __init__(self, name: str, provider: str, priority: int, context_limit: int = 4096):
        self.name = name
        self.provider = provider
        self.priority = priority
        self.context_limit = context_limit
        self.working = True
        self.last_error = None

class HybridEngine:
    def __init__(self, gemini_key, groq_key=None):
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.gemini_client = None
        self.groq_client = None
        self.available_models = []
        self.model_status = {}
        
        # Initialize clients and discover models
        self._initialize_clients()
        self._discover_all_models()

    def _initialize_clients(self):
        """Initialize API clients with error handling."""
        # Initialize Gemini client
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_key)
            logger.info("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            st.error(f"Failed to initialize Gemini: {e}")
            st.stop()

        # Initialize Groq client if key is available
        if self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
                logger.info("✅ Groq client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")
                self.groq_client = None

    def _discover_gemini_models(self) -> List[ModelInfo]:
        """Discover available Gemini models with adaptive priority ranking."""
        models = []
        try:
            model_list = self.gemini_client.models.list()
            
            # Dynamic priority assignment based on model characteristics
            def get_dynamic_priority(model_name: str) -> int:
                name_lower = model_name.lower()
                
                # Highest priority for latest/pro versions
                if any(x in name_lower for x in ['pro-latest', 'pro-002', 'pro-exp']):
                    return 100
                elif 'pro' in name_lower and '1.5' in name_lower:
                    return 95
                elif 'pro' in name_lower and '2.0' in name_lower:
                    return 98  # Future-proof for Gemini 2.0
                elif 'flash' in name_lower and '1.5' in name_lower:
                    return 90
                elif 'flash' in name_lower and '2.0' in name_lower:
                    return 92  # Future-proof for Gemini 2.0 Flash
                elif 'pro' in name_lower:
                    return 85
                elif 'flash' in name_lower:
                    return 80
                elif 'ultra' in name_lower:
                    return 105  # Highest priority for Ultra models
                else:
                    # Any other model gets base priority
                    return 70
            
            def get_context_limit(model_name: str) -> int:
                name_lower = model_name.lower()
                if 'pro' in name_lower:
                    return 8192
                elif 'ultra' in name_lower:
                    return 32768  # Ultra models typically have larger context
                elif '2.0' in name_lower:
                    return 16384  # Future Gemini 2.0 models likely have larger context
                else:
                    return 4096
            
            available_models = []
            for model in model_list:
                # Updated for new Google GenAI API - check if model can generate content
                try:
                    model_name = model.name.replace("models/", "")
                    # Try to filter only text generation models
                    if any(term in model_name.lower() for term in ['gemini', 'pro', 'flash', 'ultra']):
                        priority = get_dynamic_priority(model_name)
                        context_limit = get_context_limit(model_name)
                        
                        available_models.append(ModelInfo(
                            name=model_name, 
                            provider="gemini", 
                            priority=priority,
                            context_limit=context_limit
                        ))
                except Exception as e:
                    logger.debug(f"Skipping model {getattr(model, 'name', 'unknown')}: {e}")
                    continue
            
            # Sort by priority (highest first)
            available_models.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"✅ Discovered {len(available_models)} Gemini models")
            if available_models:
                logger.info(f"Top Gemini model: {available_models[0].name} (priority: {available_models[0].priority})")
            
            models = available_models
            
        except Exception as e:
            logger.error(f"❌ Failed to discover Gemini models: {e}")
            # Create fallback models that might exist
            fallback_models = [
                ("gemini-1.5-pro-latest", 100),
                ("gemini-1.5-pro", 95),
                ("gemini-1.5-flash", 90),
                ("gemini-pro", 85),
                ("gemini-1.0-pro", 80)
            ]
            
            # Test each fallback model to see if it actually works
            for model_name, priority in fallback_models:
                try:
                    # Try a minimal test to see if model exists
                    test_config = types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=10
                    )
                    self.gemini_client.models.generate_content(
                        model=f"models/{model_name}", 
                        contents="Hi", 
                        config=test_config
                    )
                    # If we get here, the model works
                    models.append(ModelInfo(model_name, "gemini", priority))
                    logger.info(f"✅ Verified fallback model: {model_name}")
                    break  # Use the first working model
                except Exception:
                    logger.debug(f"Fallback model {model_name} not available")
                    continue
            
            if not models:
                # Last resort - add a generic model that might exist
                models = [ModelInfo("gemini-pro", "gemini", 75)]
                logger.warning("Using generic fallback model: gemini-pro")
        
        return models

    def _discover_groq_models(self) -> List[ModelInfo]:
        """Discover available Groq models with adaptive priority ranking."""
        models = []
        if not self.groq_client:
            return models
        
        try:
            # Get available models from Groq
            model_list = self.groq_client.models.list()
            
            # Dynamic priority assignment based on model characteristics
            def get_dynamic_priority(model_id: str) -> int:
                id_lower = model_id.lower()
                
                # Priority based on model capabilities and recency
                if 'llama' in id_lower and '70b' in id_lower:
                    return 100  # Large Llama models get highest priority
                elif 'llama' in id_lower and ('3.1' in id_lower or '3.2' in id_lower):
                    return 95   # Latest Llama versions
                elif 'llama' in id_lower and '8b' in id_lower and 'instant' in id_lower:
                    return 90   # Fast Llama models
                elif 'mixtral' in id_lower and '8x7b' in id_lower:
                    return 85   # Mixtral models
                elif 'mixtral' in id_lower and '8x22b' in id_lower:
                    return 88   # Larger Mixtral models
                elif 'gemma' in id_lower and '7b' in id_lower:
                    return 80   # Gemma models
                elif 'gemma' in id_lower and '2b' in id_lower:
                    return 75   # Smaller Gemma models
                elif 'llama' in id_lower:
                    return 85   # Any other Llama model
                elif 'claude' in id_lower:
                    return 92   # Claude models if available on Groq
                else:
                    return 70   # Unknown models get base priority
            
            def get_context_limit(model_id: str, model_obj) -> int:
                # Try to get context from model object first
                if hasattr(model_obj, 'context_window') and model_obj.context_window:
                    return model_obj.context_window
                elif hasattr(model_obj, 'max_tokens') and model_obj.max_tokens:
                    return model_obj.max_tokens
                
                # Fallback to inference from model name
                id_lower = model_id.lower()
                if '32768' in id_lower or '32k' in id_lower:
                    return 32768
                elif '8192' in id_lower or '8k' in id_lower:
                    return 8192
                elif '4096' in id_lower or '4k' in id_lower:
                    return 4096
                elif 'mixtral' in id_lower:
                    return 32768  # Mixtral typically has large context
                elif 'llama' in id_lower and '70b' in id_lower:
                    return 8192   # Large Llama models
                else:
                    return 4096   # Safe default
            
            available_models = []
            for model in model_list.data:
                try:
                    model_id = model.id
                    priority = get_dynamic_priority(model_id)
                    context_limit = get_context_limit(model_id, model)
                    
                    available_models.append(ModelInfo(
                        name=model_id,
                        provider="groq",
                        priority=priority,
                        context_limit=context_limit
                    ))
                except Exception as e:
                    logger.debug(f"Skipping model {getattr(model, 'id', 'unknown')}: {e}")
                    continue
            
            # Sort by priority (highest first)
            available_models.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"✅ Discovered {len(available_models)} Groq models")
            if available_models:
                logger.info(f"Top Groq model: {available_models[0].name} (priority: {available_models[0].priority})")
            
            models = available_models
            
        except Exception as e:
            logger.error(f"❌ Failed to discover Groq models: {e}")
            # Create fallback models that might exist and test them
            fallback_models = [
                ("llama-3.1-70b-versatile", 100),
                ("llama3-70b-8192", 95),
                ("llama-3.1-8b-instant", 90),
                ("mixtral-8x7b-32768", 85),
                ("gemma-7b-it", 80),
                ("llama2-70b-4096", 75)
            ]
            
            # Test each fallback model to see if it actually works
            for model_name, priority in fallback_models:
                try:
                    # Try a minimal test to see if model exists
                    test_completion = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5,
                        timeout=10.0
                    )
                    # If we get here, the model works
                    models.append(ModelInfo(model_name, "groq", priority))
                    logger.info(f"✅ Verified fallback model: {model_name}")
                    break  # Use the first working model
                except Exception:
                    logger.debug(f"Fallback model {model_name} not available")
                    continue
            
            if not models:
                # Last resort - add generic models that might exist
                models = [
                    ModelInfo("llama3-70b-8192", "groq", 95),
                    ModelInfo("mixtral-8x7b-32768", "groq", 85)
                ]
                logger.warning("Using generic fallback models for Groq")
        
        return models

    def _discover_all_models(self, force_refresh: bool = False):
        """Discover all available models from both providers."""
        with st.spinner("🔍 Discovering available AI models..."):
            # Get models from both providers
            gemini_models = self._discover_gemini_models()
            groq_models = self._discover_groq_models()
            
            # Combine and sort all models
            self.available_models = gemini_models + groq_models
            self.available_models.sort(key=lambda x: x.priority, reverse=True)
            
            # Initialize or update model status tracking
            new_model_status = {}
            for model in self.available_models:
                key = f"{model.provider}:{model.name}"
                if key in self.model_status and not force_refresh:
                    # Keep existing status if not forcing refresh
                    new_model_status[key] = self.model_status[key]
                else:
                    # Initialize new model or reset on force refresh
                    new_model_status[key] = {
                        "working": True,
                        "attempts": 0,
                        "last_success": None,
                        "last_error": None,
                        "discovery_time": time.time()
                    }
            
            self.model_status = new_model_status
            
            logger.info(f"✅ Total models discovered: {len(self.available_models)}")
            
            # Log model summary
            gemini_count = len(gemini_models)
            groq_count = len(groq_models)
            logger.info(f"Model breakdown: {gemini_count} Gemini, {groq_count} Groq")
            
            # Display model status in UI
            self._display_model_status()
    
    def refresh_models(self):
        """Refresh model discovery and reset all model statuses."""
        logger.info("🔄 Refreshing model discovery...")
        try:
            self._discover_all_models(force_refresh=True)
            st.success("✅ Models refreshed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to refresh models: {e}")
            st.error(f"Failed to refresh models: {e}")
            return False

    def _display_model_status(self):
        """Display current model availability in the UI with refresh option."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with st.expander(f"🤖 Available Models ({len(self.available_models)} total)", expanded=False):
                if not self.available_models:
                    st.warning("No models discovered. Please check your API keys.")
                    return
                
                # Group models by provider
                gemini_models = [m for m in self.available_models if m.provider == "gemini"]
                groq_models = [m for m in self.available_models if m.provider == "groq"]
                
                if gemini_models:
                    st.markdown("**🟣 Gemini Models:**")
                    for model in gemini_models:
                        key = f"{model.provider}:{model.name}"
                        status = self.model_status.get(key, {})
                        
                        status_emoji = "✅" if status.get("working", True) else "❌"
                        attempts = status.get("attempts", 0)
                        last_error = status.get("last_error")
                        
                        status_text = f"{status_emoji} **{model.name}** (Priority: {model.priority}, Context: {model.context_limit:,})"
                        
                        if attempts > 0:
                            status_text += f" - Failures: {attempts}"
                        if last_error and not status.get("working", True):
                            status_text += f" - Error: {last_error[:50]}..."
                        
                        st.write(status_text)
                
                if groq_models:
                    st.markdown("**🟢 Groq Models:**")
                    for model in groq_models:
                        key = f"{model.provider}:{model.name}"
                        status = self.model_status.get(key, {})
                        
                        status_emoji = "✅" if status.get("working", True) else "❌"
                        attempts = status.get("attempts", 0)
                        last_error = status.get("last_error")
                        
                        status_text = f"{status_emoji} **{model.name}** (Priority: {model.priority}, Context: {model.context_limit:,})"
                        
                        if attempts > 0:
                            status_text += f" - Failures: {attempts}"
                        if last_error and not status.get("working", True):
                            status_text += f" - Error: {last_error[:50]}..."
                        
                        st.write(status_text)
                
                # Show total working models
                working_count = len([m for m in self.available_models 
                                   if self.model_status.get(f"{m.provider}:{m.name}", {}).get("working", True)])
                
                if working_count < len(self.available_models):
                    st.warning(f"⚠️ {len(self.available_models) - working_count} models are currently unavailable")
        
        with col2:
            if st.button("🔄 Refresh Models", help="Rediscover available models"):
                self.refresh_models()
                st.rerun()

    def _retry_with_backoff(self, func, max_retries=3, base_delay=1):
        """Retry function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

    def _generate_with_gemini(self, prompt: str, system_instruction: str, model_name: str, temperature: float = 0.7) -> Tuple[Optional[str], str]:
        """Generate content using Gemini model."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")
                ],
                max_output_tokens=4096
            )
            
            def make_request():
                try:
                    # Try with models/ prefix first
                    response = self.gemini_client.models.generate_content(
                        model=f"models/{model_name}", 
                        contents=prompt, 
                        config=config
                    )
                    return response.text
                except Exception as e:
                    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                        # Try without models/ prefix in case naming changed
                        logger.debug(f"Trying {model_name} without models/ prefix")
                        response = self.gemini_client.models.generate_content(
                            model=model_name, 
                            contents=prompt, 
                            config=config
                        )
                        return response.text
                    else:
                        raise e
            
            result = self._retry_with_backoff(make_request)
            
            # Mark model as working
            self.model_status[f"gemini:{model_name}"]["working"] = True
            self.model_status[f"gemini:{model_name}"]["last_success"] = time.time()
            
            return result, f"Gemini ({model_name})"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Gemini {model_name} failed: {error_msg}")
            
            # Mark model as potentially problematic
            self.model_status[f"gemini:{model_name}"]["attempts"] += 1
            self.model_status[f"gemini:{model_name}"]["last_error"] = error_msg
            
            # More intelligent model disabling logic
            attempts = self.model_status[f"gemini:{model_name}"]["attempts"]
            if attempts >= 3:
                self.model_status[f"gemini:{model_name}"]["working"] = False
                logger.warning(f"❌ Disabled Gemini model {model_name} after {attempts} failures")
            elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                # Immediately disable if model doesn't exist (likely renamed/removed)
                self.model_status[f"gemini:{model_name}"]["working"] = False
                logger.warning(f"❌ Disabled non-existent Gemini model {model_name}")
            
            return None, error_msg

    def _generate_with_groq(self, prompt: str, system_instruction: str, model_name: str, temperature: float = 0.1) -> Tuple[Optional[str], str]:
        """Generate content using Groq model."""
        if not self.groq_client:
            return None, "Groq client not available"
        
        try:
            def make_request():
                completion = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                    timeout=30.0
                )
                return completion.choices[0].message.content
            
            result = self._retry_with_backoff(make_request)
            
            # Mark model as working
            self.model_status[f"groq:{model_name}"]["working"] = True
            self.model_status[f"groq:{model_name}"]["last_success"] = time.time()
            
            return result, f"Groq ({model_name})"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Groq {model_name} failed: {error_msg}")
            
            # Mark model as potentially problematic
            self.model_status[f"groq:{model_name}"]["attempts"] += 1
            self.model_status[f"groq:{model_name}"]["last_error"] = error_msg
            
            # More intelligent model disabling logic
            attempts = self.model_status[f"groq:{model_name}"]["attempts"]
            if attempts >= 3:
                self.model_status[f"groq:{model_name}"]["working"] = False
                logger.warning(f"❌ Disabled Groq model {model_name} after {attempts} failures")
            elif "not found" in error_msg.lower() or "model_not_found" in error_msg.lower():
                # Immediately disable if model doesn't exist (likely renamed/removed)
                self.model_status[f"groq:{model_name}"]["working"] = False
                logger.warning(f"❌ Disabled non-existent Groq model {model_name}")
            
            return None, error_msg

    def generate_with_fallback(self, prompt: str, system_instruction: str, prefer_creative: bool = True, temperature: float = None) -> Tuple[str, str]:
        """Generate content with intelligent fallback across all available models."""
        
        # Set temperature based on task type if not specified
        if temperature is None:
            temperature = 0.9 if prefer_creative else 0.1
        
        # Filter working models
        working_models = [m for m in self.available_models 
                         if self.model_status[f"{m.provider}:{m.name}"]["working"]]
        
        if not working_models:
            # First try: Reset all models if none are working (maybe temporary issues)
            logger.warning("No working models found, attempting reset...")
            for key in self.model_status:
                self.model_status[key]["working"] = True
                self.model_status[key]["attempts"] = 0
            working_models = self.available_models
            
            # If still no working models after reset, try refreshing model discovery
            if not working_models:
                logger.warning("Still no models available, refreshing model discovery...")
                st.warning("⚠️ Refreshing available models...")
                self._discover_all_models(force_refresh=True)
                working_models = [m for m in self.available_models 
                                if self.model_status[f"{m.provider}:{m.name}"]["working"]]
                
                if not working_models:
                    raise Exception("❌ No working models available after refresh. Please check your API keys and network connection.")
        
        # Sort by preference: creative tasks prefer Gemini, logic tasks prefer Groq
        if prefer_creative:
            # For creative tasks, prefer Gemini models
            working_models.sort(key=lambda x: (x.provider == "gemini", x.priority), reverse=True)
        else:
            # For logic tasks, prefer Groq models
            working_models.sort(key=lambda x: (x.provider == "groq", x.priority), reverse=True)
        
        last_error = None
        
        # Try each model until one succeeds
        for i, model in enumerate(working_models):
            status_msg = f"🔄 Trying {model.provider.capitalize()} {model.name} (attempt {i+1}/{len(working_models)})"
            
            with st.status(status_msg, expanded=False) as status:
                logger.info(f"Attempting generation with {model.provider}:{model.name}")
                
                if model.provider == "gemini":
                    result, error = self._generate_with_gemini(
                        prompt, system_instruction, model.name, temperature
                    )
                elif model.provider == "groq":
                    result, error = self._generate_with_groq(
                        prompt, system_instruction, model.name, temperature
                    )
                else:
                    continue
                
                if result:
                    status.update(label=f"✅ Success with {model.provider.capitalize()} {model.name}", state="complete")
                    logger.info(f"✅ Successfully generated with {model.provider}:{model.name}")
                    return result, f"{model.provider.capitalize()} ({model.name})"
                else:
                    status.update(label=f"❌ Failed with {model.provider.capitalize()} {model.name}: {error}", state="error")
                    last_error = error
        
        # If we get here, all models failed
        error_msg = f"❌ All {len(working_models)} available models failed. Last error: {last_error}"
        logger.error(error_msg)
        st.error(error_msg)
        raise Exception(error_msg)

    def generate_art(self, prompt: str, system_instruction: str) -> Tuple[str, str]:
        """Generate creative content, preferring Gemini models."""
        return self.generate_with_fallback(prompt, system_instruction, prefer_creative=True, temperature=0.9)

    def generate_logic(self, prompt: str, system_instruction: str) -> Tuple[str, str]:
        """Generate logical/structured content, preferring Groq models."""
        return self.generate_with_fallback(prompt, system_instruction, prefer_creative=False, temperature=0.1)
    
    def health_check(self) -> Dict[str, bool]:
        """Perform a quick health check on all models."""
        health_status = {}
        
        with st.status("🔍 Performing model health check...", expanded=False) as status:
            for model in self.available_models[:3]:  # Check only top 3 models to save time
                key = f"{model.provider}:{model.name}"
                
                try:
                    if model.provider == "gemini":
                        result, _ = self._generate_with_gemini(
                            "Hi", "You are a helpful assistant. Respond with just 'OK'.", 
                            model.name, 0.1
                        )
                    elif model.provider == "groq":
                        result, _ = self._generate_with_groq(
                            "Hi", "You are a helpful assistant. Respond with just 'OK'.", 
                            model.name, 0.1
                        )
                    
                    health_status[key] = bool(result)
                    
                except Exception:
                    health_status[key] = False
            
            status.update(label="✅ Health check completed", state="complete")
        
        return health_status

# Initialize Engine with enhanced error handling
try:
    with st.spinner("🚀 Initializing AI Engine..."):
        engine = HybridEngine(GEMINI_KEY, GROQ_KEY)
    st.success("🤖 AI Engine ready!")
except Exception as e:
    st.error(f"❌ Failed to initialize AI engine: {e}")
    logger.error(f"Engine initialization failed: {e}")
    st.stop()

# Artist presets for vocal styles
artist_presets = {
    "Custom": "",
    "The Arijit Heartbreak (Soulful)": "Bollywood Sad Ballad, Acoustic Guitar, Piano, Raw Male Vocals, Emotional, 65 BPM",
    "The Coke Studio Fusion (Classy)": "Sufi Fusion, Coke Studio Style, Harmonium + Electric Guitar, Deep Texture, 95 BPM",
    "The Retro Nostalgia (90s Feel)": "90s Bollywood Melody, Kumar Sanu Vibe, Violin Strings, Dholak, Melodic, 85 BPM",
    "The Ethereal Dream (Lo-fi)": "Indian Lo-fi, Reverb-heavy vocals, Rain sounds, Slow Trap Beat, Nostalgic, 80 BPM",
    "The Power Anthem (Inspiring)": "Cinematic Orchestral, War Drums, Hans Zimmer Style, Powerful Chorus, 130 BPM"
}

# --- SESSION STATE & INPUTS WITH CRASH PROTECTION ---
try:
    # Initialize session state with error recovery
    if "step" not in st.session_state: st.session_state.step = 1
    if "concept" not in st.session_state: st.session_state.concept = ""
    if "lyrics" not in st.session_state: st.session_state.lyrics = ""
    if "final_prompt" not in st.session_state: st.session_state.final_prompt = ""
    if "error_count" not in st.session_state: st.session_state.error_count = 0
except Exception as e:
    logger.error(f"Session state initialization failed: {e}")
    # Reset to safe defaults
    st.session_state.step = 1
    st.session_state.concept = ""
    st.session_state.lyrics = ""
    st.session_state.final_prompt = ""
    st.session_state.error_count = 0

# --- UI HEADER ---
st.title("🎵 Suno Lyric Engineer (Hybrid)")

# Display enhanced brain status with model counts
gemini_count = len([m for m in engine.available_models if m.provider == "gemini"])
groq_count = len([m for m in engine.available_models if m.provider == "groq"])

if groq_count > 0:
    brain_status = f"🧠 **Dual Brain Active:** Gemini ({gemini_count} models) + Groq ({groq_count} models)"
else:
    brain_status = f"🧠 **Single Brain:** Gemini Only ({gemini_count} models)"

st.caption(brain_status)

# --- INPUTS WITH VALIDATION ---
try:
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            topic = st.text_input("Song Topic", placeholder="e.g., Cyberpunk Rain in Kolkata", max_chars=100)
            # Validate topic
            if topic and len(topic.strip()) < 3:
                st.warning("⚠️ Topic too short. Please provide a descriptive topic.")
        with col2:
            language = st.selectbox("Language", ["Hindi", "Haryanvi", "Punjabi", "Urdu", "Bengali"])
        with col3:
            preset_choice = st.selectbox("Artist Style Preset", list(artist_presets.keys()))
            if preset_choice == "Custom":
                vocal_style = st.text_input("Custom Vocal Style", placeholder="e.g., Male Gritty Folk Rap", max_chars=150)
            else:
                vocal_style = artist_presets.get(preset_choice, "")
                if vocal_style:
                    st.caption(f"🎤 **Style:** {vocal_style}")
except Exception as e:
    logger.error(f"Input UI generation failed: {e}")
    st.error("❌ UI Error occurred. Please refresh the page.")
    st.stop()

st.divider()

# --- STEP 1: CONCEPT (ARTIST BRAIN) ---
col_c1, col_c2 = st.columns([1, 3])
with col_c1:
    st.markdown("### Step 1: Concept")
    if st.button("🚀 Generate Concept", type="primary", disabled=(st.session_state.step > 1)):
        if not topic:
            st.warning("⚠️ Please enter a song topic to continue.")
        else:
            with col_c2:
                try:
                    # Increment error counter for rate limiting
                    if st.session_state.error_count > 5:
                        st.error("❌ Too many errors occurred. Please refresh the page and try again.")
                        st.stop()
                    
                    # Validate inputs before API call
                    if not topic or len(topic.strip()) < 3:
                        st.error("❌ Please provide a valid topic (at least 3 characters).")
                        st.stop()
                    
                    # Memory usage check
                    if psutil:
                        try:
                            memory_percent = psutil.virtual_memory().percent
                            if memory_percent > 90:
                                st.warning("⚠️ System memory is low. This may affect performance.")
                        except Exception:
                            pass  # Ignore memory check errors
                    
                    sys = f"ROLE: Creative Director. GOAL: Viral {language} song concept. OUTPUT: Title, Mood, Scene, Structure."
                    usr = f"Topic: {topic[:100]}. Language: {language}. Create a compelling, unique song concept that will resonate with audiences."
                    
                    # USE ENHANCED GENERATION WITH FALLBACK
                    with st.spinner("🎨 Creating concept..."):
                        res, model = engine.generate_art(usr, sys)
                    
                    # Validate response
                    if not res or len(res.strip()) < 50:
                        raise Exception("Generated content is too short or empty")
                    
                    st.session_state.concept = res[:2000]  # Limit length
                    st.session_state.step = 2
                    st.session_state.error_count = 0  # Reset on success
                    st.success(f"✅ Concept generated successfully using {model}")
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    st.session_state.error_count += 1
                    st.error(f"❌ Failed to generate concept: {error_msg}")
                    logger.error(f"Concept generation failed: {error_msg}")
                    
                    # Auto-refresh models if it seems like a model availability issue
                    if "no working models" in error_msg.lower() or "model" in error_msg.lower():
                        with st.spinner("🔄 Auto-refreshing models..."):
                            if engine.refresh_models():
                                st.info("🔄 Models refreshed. Please try again.")
                            else:
                                st.error("❌ Model refresh failed. Please check your API keys.")
                    
                    # Suggest recovery actions
                    with st.expander("🔧 Troubleshooting", expanded=False):
                        st.write("- Try a different topic")
                        st.write("- Check your internet connection")
                        st.write("- Refresh the page if errors persist")
                        st.write("- Try again in a few minutes")

if st.session_state.step >= 2:
    with col_c2:
        st.info(f"**Generated Concept:**\n\n{st.session_state.concept}")

# --- STEP 2: LYRICS (ARTIST BRAIN) ---
if st.session_state.step >= 2:
    st.divider()
    col_l1, col_l2 = st.columns([1, 3])
    with col_l1:
        st.markdown("### Step 2: Lyrics")
        if st.button("✍️ Write Lyrics", type="primary", disabled=(st.session_state.step > 2)):
            with col_l2:
                try:
                    # Enhanced lyrics generation prompt
                    sys = f"""
ROLE: You are a Legendary Poet & Songwriter (Think: Gulzar meets Ed Sheeran).
GOAL: Write a timeless, "Genius-Level" song that makes people cry or feel deep emotion.

THE "MASTERPIECE" FORMULA:

1. 🧠 SHOW, DON'T TELL (CRITICAL):
   - ❌ Bad: "I am very sad and lonely."
   - ✅ Good: "Chai ka cup thanda ho gaya, tera intezaar karte karte." (The tea went cold waiting for you.)
   - Use SENSORY details: Smell of rain, sound of a train, cold hands, flickering streetlights.

2. 🌊 THE "WAVE" STRUCTURE:
   - Verse 1: Intimate, small details (Setting the scene).
   - Pre-Chorus: The realization (The feeling builds).
   - Chorus: The Universal Truth (The line everyone sings).
   - Verse 2: The twist or deeper memory.
   - Bridge: THE BREAKING POINT (Raw emotion).

3. 🗣️ LANGUAGE BLEND (Classy Hinglish):
   - Mix {language} with *Poetic* English.
   - Use English for concepts like "Silence," "Destiny," "Midnight," "Void."
   - Avoid cheap slang (No 'Baby', 'Flex', 'Swag').

4. 🎵 RHYTHM & INTERNAL RHYME:
   - Don't just rhyme the end of sentences. Rhyme the *middle*.
   - Example: "Jo *raaste* thay *waaste*, wo kho gaye *aahista*."

OUTPUT: Raw lyrics only. No explanations.
"""
                    usr = f"Concept: {st.session_state.concept}\n\nWrite compelling, authentic hit lyrics that capture the essence of this concept. Make it memorable and emotionally resonant."
                    
                    # USE ENHANCED GENERATION WITH FALLBACK
                    res, model = engine.generate_art(usr, sys)
                    st.session_state.lyrics = res
                    st.session_state.step = 3
                    st.success(f"✅ Lyrics generated successfully using {model}")
                    st.rerun()
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Failed to generate lyrics: {error_msg}")
                    logger.error(f"Lyrics generation failed: {error_msg}")
                    
                    # Auto-refresh models if it seems like a model availability issue
                    if "no working models" in error_msg.lower() or "model" in error_msg.lower():
                        with st.spinner("🔄 Auto-refreshing models..."):
                            if engine.refresh_models():
                                st.info("🔄 Models refreshed. Please try again.")
                            else:
                                st.error("❌ Model refresh failed. Please check your API keys.")

    if st.session_state.step >= 3:
        with col_l2:
             st.session_state.lyrics = st.text_area(
                 "Edit Lyrics (you can modify the generated lyrics if needed)", 
                 st.session_state.lyrics, 
                 height=300,
                 help="Feel free to edit the generated lyrics to better match your vision"
             )

# --- STEP 3: OPTIMIZATION (LOGIC BRAIN) ---
if st.session_state.step >= 3:
    st.divider()
    col_o1, col_o2 = st.columns([1, 3])
    with col_o1:
        st.markdown("### Step 3: Engineer")
        if st.button("⚡ Optimize for Suno", type="primary"):
            with col_o2:
                try:
                    # Enhanced Suno optimization prompt
                    sys = f"""
ROLE: Suno v4.5 Audio Director (Specializing in Emotional Dynamics).
TASK: Engineer the lyrics into a Suno prompt that creates a "Vocal Performance," not just a song.

---
### PART 1: STYLE STRING (The Atmosphere)
Create a rich, texture-heavy style string.
*Formula:* [Emotion] + [Instrument] + [Vocal Texture] + [Pace]
*Examples:*
- "Heartbreak Ballad, Solo Piano, Cello Swell, Raw Male Vocals, Slow Tempo"
- "Cinematic Orchestral, Indian Flute, Deep Bass, Ethereal Female Voice, 80 BPM"

---
### PART 2: PHONETICS (The "Native" Sound)
**SAFETY RULE: DO NOT OVER-ENGINEER.**
Respell ONLY words that American AI mispronounces. If a word is phonetic in English (e.g., 'Kal', 'Bus', 'Hum'), KEEP IT ORIGINAL.

*Targeted Respellings Only:*
1. 'Dil' -> 'Dill' (Prevents 'Dial')
2. 'Hai' -> 'High' (Prevents 'Hay')
3. 'Kyun' -> 'Kyoon'
4. 'Tujhe' -> 'Too-jay'
5. 'Main' -> 'May'
6. 'Zindagi' -> 'Zind-a-gee'

*English Words:*
- KEEP standard spelling (e.g., 'Love', 'Destiny', 'Vibe'). DO NOT phonetically respell English.

---
### PART 3: EMOTIONAL TAGS (The "Soul")
You MUST use these tags to direct the singer:
- [Intimate Whisper] -> For Verse 1 (Draws listener in).
- [Voice Crack] -> For the saddest line.
- [Power Belt] or [High Note] -> For the climax of the Chorus.
- [Silence] -> Before the final line.
- [Sob] or [Breath] -> For raw texture.

OUTPUT FORMAT:
### STYLE
[Insert Style String]

### LYRICS
[Intro]
(Atmospheric instrumental)

[Verse 1]
[Intimate Whisper]
(Lyrics with selective phonetics...)

[Pre-Chorus]
[Rising Intensity]
(Lyrics...)

[Chorus]
[Soaring Vocals]
(Lyrics...)

[Bridge]
[Stripped Back]
[Voice Crack]
(Lyrics...)

[Outro]
[Fade to Silence]
(Lyrics...)
"""
                    usr = f"""
                    OPTIMIZE THESE LYRICS FOR SUNO AI:
                    
                    Lyrics:
                    {st.session_state.lyrics}
                    
                    Vocal Style: {vocal_style}
                    Language: {language}
                    
                    Create a production-ready Suno prompt that will generate high-quality audio.
                    """
                    
                    # USE ENHANCED LOGIC GENERATION (prefers Groq but falls back)
                    res, model = engine.generate_logic(usr, sys)
                    st.session_state.final_prompt = res
                    st.session_state.step = 4
                    st.success(f"✅ Suno optimization completed using {model}")
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Failed to optimize for Suno: {error_msg}")
                    logger.error(f"Suno optimization failed: {error_msg}")
                    
                    # Auto-refresh models if it seems like a model availability issue
                    if "no working models" in error_msg.lower() or "model" in error_msg.lower():
                        with st.spinner("🔄 Auto-refreshing models..."):
                            if engine.refresh_models():
                                st.info("🔄 Models refreshed. Please try again.")
                            else:
                                st.error("❌ Model refresh failed. Please check your API keys.")

    if st.session_state.final_prompt:
        with col_o2:
            st.success("✅ **Production Ready!** Copy the optimized prompt below:")
            
            # Primary method: Text area for reliable manual copying
            st.text_area(
                "📋 **Optimized Suno Prompt** (Select All + Ctrl+C to copy):",
                st.session_state.final_prompt,
                height=250,
                help="Click in the text area, press Ctrl+A to select all, then Ctrl+C to copy"
            )
            
            # Secondary method: Code block for alternative copying
            st.code(st.session_state.final_prompt, language="text")
            
            # Nice-to-have: JavaScript copy button (may not work in all environments)
            st.markdown("**Quick Copy Button** *(may require clipboard permissions)*:")
            # Safely escape the prompt for JavaScript template literal
            safe_prompt = st.session_state.final_prompt.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
            copy_js = f"""
            <script>
            function copyToClipboard() {{
                const text = `{safe_prompt}`;
                
                if (navigator.clipboard && window.isSecureContext) {{
                    navigator.clipboard.writeText(text).then(function() {{
                        document.getElementById('copy-status').innerHTML = '✅ Copied!';
                        document.getElementById('copy-status').style.color = 'green';
                        setTimeout(function() {{
                            document.getElementById('copy-status').innerHTML = '📋 Quick Copy';
                            document.getElementById('copy-status').style.color = 'black';
                        }}, 2000);
                    }}, function(err) {{
                        document.getElementById('copy-status').innerHTML = '❌ Use manual copy above';
                        document.getElementById('copy-status').style.color = 'red';
                    }});
                }} else {{
                    document.getElementById('copy-status').innerHTML = '❌ Use manual copy above';
                    document.getElementById('copy-status').style.color = 'red';
                }}
            }}
            </script>
            
            <button onclick="copyToClipboard()" style="
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.25rem;
                cursor: pointer;
                font-weight: bold;
                margin: 5px 0;
                font-size: 12px;
            " onmouseover="this.style.backgroundColor='#45a049'" onmouseout="this.style.backgroundColor='#4CAF50'">
                <span id="copy-status">📋 Quick Copy</span>
            </button>
            """
            
            st.components.v1.html(copy_js, height=60)
            
            # Additional helpful information
            with st.expander("🎯 **How to use this in Suno AI**", expanded=False):
                st.markdown("""
                **Steps to create your song:**
                1. Go to [Suno AI](https://suno.ai)
                2. Click "Create" 
                3. Paste the optimized prompt above into the lyrics field
                4. Select or let Suno auto-generate the music style
                5. Click "Generate" and wait for your song!
                
                **Pro Tips:**
                - The prompt is already optimized for best results
                - Vocal style and language cues are included
                - Tags will help structure your song properly
                """)
            
            # Reset option
            if st.button("🔄 Start New Song", type="secondary"):
                for key in ["step", "concept", "lyrics", "final_prompt"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()