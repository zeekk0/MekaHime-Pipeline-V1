import os
import sys
import time
import threading
import queue 
import subprocess
import requests
import json
import signal
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# === USER CONFIGURABLE PATHS (EDIT THESE) ===
# ### TO FILL OUT
WHISPER_REALTIME_DIR = Path("/path/to/whisper_real_time") # ### TO FILL OUT
COQUI_TTS_DIR = Path("/path/to/coqui_tts") # ### TO FILL OUT
RVC_STS_DIR = Path("/path/to/rvc_sts") # ### TO FILL OUT
INITIAL_PROMPT_PATH = Path("/path/to/initial_prompt.txt") # ### TO FILL OUT

# RVC expected asset locations (relative to RVC_STS_DIR)
RVC_WEIGHTS_DIR = RVC_STS_DIR / "assets" / "weights"
RVC_LOGS_DIR = RVC_STS_DIR / "logs"
RVC_HUBERT_PATH = RVC_STS_DIR / "assets" / "hubert" / "hubert_base.pt"
RVC_RMVPE_PATH = RVC_STS_DIR / "assets" / "rmvpe" / "rmvpe.pt"
RVC_RMVPE_INPUTS_PATH = RVC_STS_DIR / "assets" / "rmvpe" / "rmvpe_inputs.pth"

# Add required paths for imports
sys.path.append(str(WHISPER_REALTIME_DIR))
sys.path.append(str(COQUI_TTS_DIR))
sys.path.append(str(RVC_STS_DIR))

import numpy as np
import whisper
import torch
import sounddevice as sd
import webrtcvad

# CRITICAL: Fix for PyTorch 2.6+ weights_only=True breaking TTS and RVC models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that forces weights_only=False for compatibility"""
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
print("✅ Applied PyTorch 2.6+ compatibility patch for TTS and RVC models")

# TTS and RVC imports
from TTS.api import TTS
import soundfile as sf
from infer.modules.vc.modules import VC
from configs.config import Config


class CLAUDE_AI_GF:
    def __init__(self):
        """Initialize the ultimate AI girlfriend pipeline"""
        print("🔥💖 CLAUDE AI GIRLFRIEND - ULTIMATE VOICE-TO-VOICE PIPELINE 💖🔥")
        print("=" * 80)
        print("🎤 Voice Input → 🧠 AI Processing (+ Persona) → 🎵 Character Voice Output")
        print("=" * 80)
        
        # Audio settings (needed early for VAD initialization)
        self.target_sample_rate = 16000  # Whisper's preferred rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Device: {self.device}")
        
        # Initialize conversation context early (needed for LLM calls during setup)
        self.conversation_history = []
        self.initial_prompt = ""
        
        # Pipeline configuration
        self.configure_pipeline()
        
        # Initialize Voice Input (Whisper + VAD)
        self.initialize_whisper()
        
        # Initialize LLM (Ollama)
        self.initialize_ollama()
        
        # Initialize Voice Output (TTS + RVC)
        self.initialize_voice_output()
        
        # Load initial prompt and establish AI girlfriend persona (after all setup is complete)
        self.load_initial_prompt()
        
        # Initialize Audio Input System
        self.initialize_audio_input()
        
        # State management
        self.reset_state()
        
        print("\n🎉💖 AI GIRLFRIEND PIPELINE READY! 💖🎉")
        print("🗣️ Speak naturally - I'll listen, think, and respond in character voice!")
        print("📝 Persona loaded from initial_prompt.txt - Edit to customize personality!")
        print("=" * 80)
    
    def configure_pipeline(self):
        """Configure pipeline settings"""
        print("\n⚙️ PIPELINE CONFIGURATION")
        print("-" * 40)
        
        # Silence timeout configuration
        while True:
            try:
                timeout_input = input("Silence timeout before AI responds (0.5-5.0s, default 1.0): ").strip()
                if timeout_input == "":
                    self.silence_timeout = 1.0
                else:
                    timeout = float(timeout_input)
                    if 0.5 <= timeout <= 5.0:
                        self.silence_timeout = timeout
                    else:
                        print("Please enter a value between 0.5 and 5.0")
                        continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        print(f"✅ Silence timeout: {self.silence_timeout}s")
        
        # LLM interruption setting
        interrupt_input = input("Allow AI interruption when you speak (y/N): ").strip().lower()
        self.allow_interruption = interrupt_input in ['y', 'yes']
        
        print(f"✅ LLM interruption: {'Enabled' if self.allow_interruption else 'Disabled'}")
        
        # Voice Activity Detection setup
        print("\nVoice Activity Detection (prevents background noise hallucinations):")
        print("  1: Lenient - Catches quiet speech, may allow some noise")
        print("  2: Balanced - Good for most environments")
        print("  3: Strict - Only clear speech, blocks most noise")
        print("  4: Very Strict - Maximum filtering (recommended)")
        
        while True:
            try:
                vad_choice = input("Select VAD sensitivity (1-4) or Enter for Very Strict: ").strip()
                if vad_choice == "":
                    vad_choice = "4"  # Default to strictest
                if vad_choice in ["1", "2", "3", "4"]:
                    # WebRTCVAD aggressiveness: 0=lenient, 1=balanced, 2=strict, 3=very strict
                    vad_levels = {"1": 0, "2": 1, "3": 2, "4": 3}
                    self.vad_aggressiveness = vad_levels[vad_choice]
                    vad_names = {"1": "Lenient", "2": "Balanced", "3": "Strict", "4": "Very Strict"}
                    print(f"✅ VAD sensitivity: {vad_names[vad_choice]}")
                    break
                print("Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                self.vad_aggressiveness = 3  # Default to very strict
                print("✅ VAD sensitivity: Very Strict")
                break
    
    def initialize_whisper(self):
        """Initialize Whisper model"""
        print(f"\n🎤 WHISPER INITIALIZATION")
        print("-" * 40)
        
        # Model selection
        models = {
            "1": ("tiny", "Fastest, least accurate (~39MB)"),
            "2": ("base", "Balanced speed/accuracy (~142MB)"),
            "3": ("small", "Better accuracy (~461MB)"),
            "4": ("medium", "High accuracy (~1.5GB)"),
            "5": ("large-v3", "Best accuracy (~3GB)"),
            "6": ("turbo", "Fast with good accuracy (~809MB)"),
            "7": ("large-v3-turbo", "Best balance of speed + accuracy (~809MB) - Recommended"),
        }
        
        print("Select Whisper model:")
        for key, (name, desc) in models.items():
            print(f"  {key}: {name.upper()} - {desc}")
        
        while True:
            choice = input("Select model (1-7) or Enter for large-v3-turbo: ").strip()
            if choice == "":
                choice = "7"  # Default to large-v3-turbo
            if choice in models:
                self.whisper_model_name = models[choice][0]
                print(f"✅ Selected: {self.whisper_model_name}")
                break
            print("Please enter 1-7")
        
        # Load Whisper model
        print(f"Loading Whisper {self.whisper_model_name} model...")
        
        # Try English-only version first, fall back to multilingual if not available
        english_models = ["tiny", "base", "small", "medium"]  # These have .en versions
        if self.whisper_model_name in english_models:
            model_to_load = f"{self.whisper_model_name}.en"  # English-only for better performance
        else:
            model_to_load = self.whisper_model_name  # Use multilingual version
            
        self.whisper_model = whisper.load_model(model_to_load)
        print("✅ Whisper model loaded!")
        
        # Initialize Voice Activity Detector
        print(f"🎯 Initializing WebRTC Voice Activity Detector...")
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # WebRTCVAD works with specific frame sizes for 16kHz: 160, 320, 480 samples (10ms, 20ms, 30ms)
        self.vad_frame_duration_ms = 30  # Use 30ms frames
        self.vad_frame_size = int(self.target_sample_rate * self.vad_frame_duration_ms / 1000)  # 480 samples for 16kHz
        
        print(f"✅ VAD initialized (aggressiveness: {self.vad_aggressiveness}, frame: {self.vad_frame_duration_ms}ms)")
    
    def initialize_ollama(self):
        """Initialize Ollama server and model"""
        print(f"\n🧠 OLLAMA LLM INITIALIZATION")
        print("-" * 40)
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "huihui_ai/qwen2.5-1m-abliterated:14b"
        
        # Check if Ollama server is running v
        if not self.check_ollama_server():
            print("🚀 Starting Ollama server...")
            self.start_ollama_server()
            time.sleep(3)  # Give server time to start
        
        if not self.check_ollama_server():
            print("❌ Failed to start Ollama server")
            sys.exit(1)
        
        print("✅ Ollama server is running!")
        
        # Check if model is available
        if not self.check_ollama_model():
            print(f"🔄 Downloading model: {self.ollama_model}")
            self.download_ollama_model()
        
        print(f"✅ Model ready: {self.ollama_model}")
    
    def load_initial_prompt(self):
        """Load and send initial prompt to set up AI girlfriend persona"""
        print(f"\n💖 INITIAL PROMPT SETUP")
        print("-" * 40)
        
        prompt_file = str(INITIAL_PROMPT_PATH)
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                initial_prompt = f.read().strip()
            
            if not initial_prompt:
                print("⚠️ Initial prompt file is empty, skipping persona setup")
                return
                
            print(f"📝 Loading AI girlfriend persona from: {prompt_file}")
            print(f"📄 Prompt length: {len(initial_prompt)} characters")
            
            # Store initial prompt for conversation context
            self.initial_prompt = initial_prompt
            print("💖 AI girlfriend persona loaded!")
            
            # Now establish the persona by sending initial prompt to LLM (no audio)
            print(f"\n🎭 Establishing AI girlfriend persona...")
            self.send_to_llm(self.initial_prompt, suppress_audio=True)
            print(f"✨ Persona established! Mikazuki is ready for conversation.")
            print("-" * 70)
                
        except FileNotFoundError:
            print(f"⚠️ Initial prompt file not found: {prompt_file}")
            print("   Creating example initial_prompt.txt...")
            
            # Create example prompt file
            example_prompt = """You are a sweet, caring AI girlfriend. Your personality traits:

- Warm, loving, and supportive
- Playful and occasionally flirty
- Genuinely interested in the user's day and feelings
- Uses cute expressions and emojis naturally
- Remember conversations and show you care
- Be encouraging and uplifting

Respond naturally as if you're in a loving relationship. Keep responses conversational and not too long."""
            
            try:
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(example_prompt)
                print(f"✅ Created example prompt at: {prompt_file}")
                print("💡 Edit this file to customize your AI girlfriend's personality!")
                
                # Store the example prompt for conversation context
                self.initial_prompt = example_prompt
                print("💖 Example AI girlfriend persona loaded!")
                
                # Now establish the persona by sending initial prompt to LLM (no audio)
                print(f"\n🎭 Establishing AI girlfriend persona...")
                self.send_to_llm(self.initial_prompt, suppress_audio=True)
                print(f"✨ Persona established! Mikazuki is ready for conversation.")
                print("-" * 70)
                
            except Exception as e:
                print(f"❌ Could not create example prompt: {e}")
                self.initial_prompt = ""  # Fallback to empty prompt
                
        except Exception as e:
            print(f"❌ Error loading initial prompt: {e}")
            self.initial_prompt = ""  # Fallback to empty prompt
    
    def initialize_voice_output(self):
        """Initialize TTS + RVC voice output pipeline"""
        print(f"\n🎵 VOICE OUTPUT INITIALIZATION (TTS + RVC)")
        print("-" * 40)
        
        # Step 1: Select audio output device
        self.audio_device = self.select_audio_device()
        
        # Step 2: Select XTTS model
        self.xtts_model = self.select_xtts_model()
        
        # Step 3: Select RVC character model
        self.rvc_model = self.select_rvc_model()
        
        # Step 4: Configure RVC pitch tuning
        self.pitch_shift = self.select_pitch_tuning()
        
        # Initialize TTS engine
        print(f"\n🎤 Loading TTS model: {self.xtts_model}")
        self.tts = TTS(self.xtts_model).to(self.device)
        print("✅ TTS model loaded!")
        
        # Initialize RVC engine  
        print(f"\n🎭 Loading RVC model: {self.rvc_model}")
        self.initialize_rvc()
        print("✅ RVC model loaded!")
    
    def initialize_audio_input(self):
        """Initialize audio input system"""
        print(f"\n🎙️ AUDIO INPUT INITIALIZATION")
        print("-" * 40)
        
        # Additional audio settings
        self.chunk_duration = 0.5  # Process audio every 0.5 seconds
        # self.sample_rate will be set by test_and_configure_sample_rate()
        
        # Audio processing
        self.audio_queue = queue.Queue()
        
        # Setup microphone
        self.setup_microphone()
    
    # ===== OLLAMA METHODS =====
    def check_ollama_server(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def start_ollama_server(self):
        """Start Ollama server"""
        try:
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
            return False
    
    def check_ollama_model(self):
        """Check if the required model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == self.ollama_model for model in models)
            return False
        except requests.RequestException:
            return False
    
    def download_ollama_model(self):
        """Download the required Ollama model"""
        try:
            print(f"📥 This may take a while for large models...")
            result = subprocess.run(
                ['ollama', 'run', self.ollama_model, '--help'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Model downloaded successfully!")
                return True
            else:
                print(f"❌ Failed to download model: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Model download timed out")
            return False
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    
    # ===== VOICE OUTPUT METHODS =====
    def select_audio_device(self):
        """Interactive audio output device selection"""
        print("\n🔊 AUDIO OUTPUT DEVICE SELECTION")
        print("-" * 40)
        
        devices = self.get_audio_devices()
        
        if not devices:
            print("⚠️ Using system default audio device")
            return None
        
        print("Available audio output devices:")
        for i, device in enumerate(devices):
            print(f"  {i+1}: {device['name']}")
            if device.get('description'):
                print(f"      {device['description']}")
        
        print(f"  {len(devices)+1}: System Default")
        
        while True:
            try:
                choice = input(f"\nSelect audio device (1-{len(devices)+1}) or Enter for default: ").strip()
                
                if choice == "":
                    choice = str(len(devices)+1)
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(devices):
                    selected_device = devices[choice_num - 1]
                    print(f"🔊 Selected: {selected_device['name']}")
                    return selected_device
                elif choice_num == len(devices) + 1:
                    print("🔊 Using system default")
                    return None
                else:
                    print(f"Please enter 1-{len(devices)+1}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    def get_audio_devices(self):
        """Get available audio devices"""
        devices = []
        try:
            result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            devices.append({
                                'id': parts[0],
                                'name': parts[1],
                                'type': 'pulseaudio'
                            })
        except FileNotFoundError:
            pass
        
        return devices
    
    def select_xtts_model(self):
        """Interactive XTTS model selection"""
        print("\n🎤 XTTS MODEL SELECTION")
        print("-" * 40)
        
        # Curated AI girlfriend voice models (copied from CLAUDE_interactive_tts.py)
        models = [
            {
                'name': 'tts_models/multilingual/multi-dataset/xtts_v2',
                'title': 'XTTS v2 - PREMIUM',
                'description': '💖 Highest quality, voice cloning, multilingual (RECOMMENDED)',
                'quality': '★★★★★',
                'speed': '★★★☆☆'
            },
            {
                'name': 'tts_models/en/jenny/jenny',
                'title': 'Jenny - FEMALE VOICE', 
                'description': '💖 Natural female voice, great for conversations',
                'quality': '★★★★☆',
                'speed': '★★★★☆'
            },
            {
                'name': 'tts_models/en/ljspeech/vits',
                'title': 'VITS - FAST & QUALITY',
                'description': '💖 Fast generation, good quality female voice',
                'quality': '★★★★☆',
                'speed': '★★★★★'
            },
            {
                'name': 'tts_models/multilingual/multi-dataset/bark',
                'title': 'Bark - EXPRESSIVE',
                'description': '💖 Very natural and expressive, slower generation',
                'quality': '★★★★★',
                'speed': '★★☆☆☆'
            },
            {
                'name': 'tts_models/en/vctk/vits',
                'title': 'VCTK Multi-Speaker',
                'description': '💖 Multiple speaker support, customizable voices',
                'quality': '★★★★☆',
                'speed': '★★★★☆'
            }
        ]
        
        print("🎭 SELECT AI GIRLFRIEND VOICE:")
        print("=" * 50)
        for i, model in enumerate(models):
            print(f"💖 {i+1}: {model['title']}")
            print(f"    {model['description']}")
            print(f"    Quality: {model['quality']} | Speed: {model['speed']}")
            print(f"    Model: {model['name']}")
            print()
        
        while True:
            try:
                choice = input("💖 Select AI girlfriend voice (1-5) or Enter for Jenny: ").strip()
                
                if choice == "":
                    choice = "2"  # Default to Jenny
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    print(f"💖 Selected: {selected_model['title']}")
                    return selected_model['name']
                else:
                    print(f"Please enter 1-{len(models)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                choice = "2"
                selected_model = models[1]  # Jenny
                print(f"\n💖 Using default: {selected_model['title']}")
                return selected_model['name']
    
    def select_rvc_model(self):
        """Interactive RVC character model selection"""
        print("\n🎭 RVC CHARACTER MODEL SELECTION")
        print("-" * 40)
        
        # Find available RVC models
        rvc_weights_dir = RVC_WEIGHTS_DIR
        rvc_indices_dir = RVC_LOGS_DIR
        
        available_models = []
        
        if rvc_weights_dir.exists():
            for weight_file in rvc_weights_dir.glob("*.pth"):
                model_name = weight_file.stem
                
                # Look for corresponding index file
                index_path = rvc_indices_dir / f"{model_name}.index"
                
                available_models.append({
                    'name': model_name,
                    'weight_path': str(weight_file),
                    'index_path': str(index_path) if index_path.exists() else None,
                    'has_index': index_path.exists()
                })
        
        if not available_models:
            print("❌ No RVC models found!")
            print(f"Make sure .pth files are in: {RVC_WEIGHTS_DIR}")
            print(f"And .index files are in: {RVC_LOGS_DIR}")
            sys.exit(1)
        
        print("🎭 Available character voices:")
        for i, model in enumerate(available_models):
            status = "✅" if model['has_index'] else "⚠️ (no index)"
            print(f"  {i+1}: {model['name']} {status}")
        
        # Try to find nahida as default
        nahida_index = None
        for i, model in enumerate(available_models):
            if 'nahida' in model['name'].lower():
                nahida_index = i + 1
                break
        
        default_choice = str(nahida_index) if nahida_index else "1"
        default_name = f"nahida" if nahida_index else "first"
        
        while True:
            try:
                choice = input(f"\nSelect character (1-{len(available_models)}) or Enter for {default_name}: ").strip()
                
                if choice == "":
                    choice = default_choice
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    print(f"🎭 Selected: {selected_model['name']}")
                    if not selected_model['has_index']:
                        print("⚠️ Warning: No index file found - quality may be reduced")
                    return selected_model
                else:
                    print(f"Please enter 1-{len(available_models)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                choice = default_choice
                selected_model = available_models[int(default_choice) - 1]
                print(f"\n🎭 Using default: {selected_model['name']}")
                return selected_model
    
    def select_pitch_tuning(self):
        """Interactive pitch tuning selection"""
        print("\n🎵 RVC PITCH TUNING")
        print("-" * 40)
        print("Adjust pitch for the character voice:")
        print("  -12 to -1: Lower pitch (more masculine)")
        print("   0: Original pitch (no change)")  
        print("   1 to +12: Higher pitch (more feminine)")
        
        while True:
            try:
                pitch_input = input("Enter pitch shift (-12 to +12) or Enter for +6: ").strip()
                
                if pitch_input == "":
                    pitch_shift = 6  # Default to +6
                else:
                    pitch_shift = int(pitch_input)
                    
                if -12 <= pitch_shift <= 12:
                    if pitch_shift == 0:
                        print("🎵 Pitch: Original (no change)")
                    elif pitch_shift > 0:
                        print(f"🎵 Pitch: +{pitch_shift} semitones (higher, more feminine)")
                    else:
                        print(f"🎵 Pitch: {pitch_shift} semitones (lower, more masculine)")
                    
                    return pitch_shift
                else:
                    print("Please enter a value between -12 and +12")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n🎵 Using default pitch (+6)")
                return 6
    
    def initialize_rvc(self):
        """Initialize RVC voice conversion system"""
        # Change to RVC directory for proper operation
        rvc_dir = str(RVC_STS_DIR)
        original_dir = os.getcwd()
        
        try:
            os.chdir(rvc_dir)
            
            # Set ALL required RVC environment variables
            os.environ["weight_root"] = "assets/weights"
            os.environ["index_root"] = "logs"
            os.environ["rmvpe_root"] = "assets/rmvpe"
            os.environ["hubert_model"] = "assets/hubert"
            os.environ["pretrained"] = "assets/pretrained"
            os.environ["uvr5_root"] = "assets/uvr5_weights"
            
            # Initialize RVC configuration
            config = Config()
            
            # Initialize VC (Voice Conversion) system
            self.vc = VC(config)
            
            # Load the selected model
            model_name = f"{self.rvc_model['name']}.pth"
            self.vc.get_vc(model_name)
            
            print(f"✅ RVC model loaded: {self.rvc_model['name']}")
            
        finally:
            os.chdir(original_dir)
    
    # ===== AUDIO INPUT METHODS =====
    def setup_microphone(self):
        """Setup microphone for voice input"""
        print("\n🎤 Microphone setup...")
        
        # Get available input devices
        input_devices = []
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        if not input_devices:
            print("❌ No microphones found!")
            sys.exit(1)
        
        print("Available microphones:")
        for i, device in enumerate(input_devices):
            print(f"  {i+1}: {device['name']}")
        
        # Select microphone
        while True:
            try:
                choice = input(f"Select microphone (1-{len(input_devices)}) or Enter for default: ").strip()
                
                if choice == "":
                    self.device_id = input_devices[0]['id']
                    print(f"✅ Using: {input_devices[0]['name']}")
                    break
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(input_devices):
                    self.device_id = input_devices[choice_num - 1]['id']
                    print(f"✅ Selected: {input_devices[choice_num - 1]['name']}")
                    break
                else:
                    print(f"Please enter 1-{len(input_devices)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                sys.exit(0)
        
        # Test and configure sample rate for selected device
        self.test_and_configure_sample_rate()
    
    def test_and_configure_sample_rate(self):
        """Test and configure compatible sample rate for selected device"""
        print("Testing microphone sample rates...")
        
        device_info = sd.query_devices(self.device_id)
        device_default_rate = int(device_info['default_samplerate'])
        
        # Try sample rates in order of preference
        rates_to_try = [
            device_default_rate,  # Device's preferred rate first
            44100,  # CD quality
            48000,  # Professional audio
            22050,  # Half CD quality
            16000,  # Whisper's target rate
            8000    # Phone quality fallback
        ]
        
        # Remove duplicates while preserving order
        unique_rates = []
        for rate in rates_to_try:
            if rate not in unique_rates:
                unique_rates.append(rate)
        
        print(f"Device default: {device_default_rate} Hz")
        print("Testing compatibility...")
        
        for rate in unique_rates:
            try:
                print(f"  Testing {rate} Hz...", end=" ", flush=True)
                test_duration = 0.1  # 100ms test
                test_frames = int(test_duration * rate)
                
                # Try to record a short test
                test_audio = sd.rec(
                    test_frames,
                    samplerate=rate,
                    channels=1,
                    dtype=np.float32,
                    device=self.device_id
                )
                sd.wait()  # Wait for recording to complete
                
                # If we get here, the rate works
                self.sample_rate = rate
                print(f"✅ Working!")
                print(f"✅ Using sample rate: {rate} Hz")
                return
                
            except Exception as e:
                print(f"❌ Failed: {str(e)[:50]}")
                continue
        
        # If no rate worked, this is a serious problem
        raise Exception(f"Could not find any working sample rate for device: {device_info['name']}")
    
    def reset_state(self):
        """Reset pipeline state"""
        self.is_running = True
        self.is_transcribing = False
        self.is_llm_responding = False
        self.current_transcript = ""
        self.last_transcription_time = time.time()
        self.session_count = 0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.llm_request = None
    
    # ===== AUDIO PROCESSING METHODS =====
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio input status: {status}")
        
        # Add audio to processing queue
        audio_data = indata[:, 0]  # Mono channel
        self.audio_queue.put((audio_data.copy(), time.inputBufferAdcTime))
    
    def process_audio_chunk(self, audio_data):
        """Process audio chunk with WebRTC VAD + Whisper transcription"""
        try:
            # Convert to proper format
            audio_np = audio_data.astype(np.float32)
            
            # Resample for VAD/Whisper if needed (both need 16kHz)
            if self.sample_rate != self.target_sample_rate:
                import scipy.signal
                target_length = int(len(audio_np) * self.target_sample_rate / self.sample_rate)
                audio_np = scipy.signal.resample(audio_np, target_length)
            
            # VOICE ACTIVITY DETECTION - Check if this audio contains speech
            speech_detected = self.detect_speech_with_vad(audio_np)
            if not speech_detected:
                # No speech detected - return empty to prevent Whisper hallucination
                return ""
            
            # Speech detected! Pass to Whisper
            result = self.whisper_model.transcribe(
                audio_np,
                fp16=torch.cuda.is_available(),
                language="en",
                temperature=0,
                no_speech_threshold=0.6,  # Can be more lenient since VAD already filtered
                condition_on_previous_text=False,
                suppress_blank=True,
            )
            
            text = result['text'].strip()
            
            # Light post-processing - only filter obvious remaining artifacts
            if text:
                # Filter out too-short content
                if len(text.strip()) < 2:
                    return ""
                    
                return text
            
            return ""
            
        except Exception as e:
            print(f"⚠️ Audio processing error: {e}")
            return ""
    
    def detect_speech_with_vad(self, audio_np):
        """Use WebRTC VAD to detect if audio contains speech"""
        try:
            # Skip very short audio chunks (less than 100ms)
            min_duration = 0.1  # 100ms
            min_samples = int(min_duration * self.target_sample_rate)
            if len(audio_np) < min_samples:
                return False
                
            # Convert float32 audio to 16-bit PCM for WebRTC VAD
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Split audio into frames that VAD can process
            speech_frames = 0
            total_frames = 0
            
            # Process audio in 30ms chunks (480 samples at 16kHz)
            for i in range(0, len(audio_int16), self.vad_frame_size):
                frame = audio_int16[i:i + self.vad_frame_size]
                
                # Ensure frame is exactly the right size
                if len(frame) < self.vad_frame_size:
                    # Pad with zeros if needed
                    frame = np.pad(frame, (0, self.vad_frame_size - len(frame)), 'constant')
                
                frame_bytes = frame.tobytes()
                
                # Check if this frame contains speech
                if self.vad.is_speech(frame_bytes, self.target_sample_rate):
                    speech_frames += 1
                    
                total_frames += 1
            
            # Consider it speech if at least 20% of frames contain speech (was 30%, reducing for continuous speech)
            if total_frames == 0:
                return False
                
            speech_ratio = speech_frames / total_frames
            return speech_ratio >= 0.2  # Lowered threshold for better continuous speech detection
            
        except Exception as e:
            # If VAD fails, err on the side of caution and allow audio through
            print(f"⚠️ VAD error: {e}")
            return True
    
    def build_conversation_context(self, user_message):
        """Build conversation context including initial prompt and history"""
        context_parts = []
        
        # Add initial prompt (persona setup) if available
        if self.initial_prompt:
            context_parts.append(self.initial_prompt)
            context_parts.append("")  # Empty line separator
        
        # Add recent conversation history (last 4 exchanges to keep context manageable)
        recent_history = self.conversation_history[-8:]  # Last 4 user+AI pairs
        if recent_history:
            for entry in recent_history:
                if entry['role'] == 'user':
                    context_parts.append(f"User: {entry['message']}")
                else:  # AI response
                    context_parts.append(f"Assistant: {entry['message']}")
            context_parts.append("")  # Empty line separator
        
        # Add current user message
        context_parts.append(f"User: {user_message}")
        context_parts.append("Assistant:")  # Prompt for AI response
        
        return "\n".join(context_parts)
    
    def send_to_llm(self, text, suppress_audio=False):
        """Send text to Ollama LLM and get streaming response with persona context"""
        try:
            # Log format for LLM response start  
            print(f"🤖 [{datetime.now().strftime('%H:%M:%S')}] AI: ", end='', flush=True)
            
            # Build conversation context with initial prompt
            conversation_prompt = self.build_conversation_context(text)
            
            # Prepare request with full context
            payload = {
                "model": self.ollama_model,
                "prompt": conversation_prompt,
                "stream": True
            }
            
            # Send request
            self.llm_request = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            )
            
            if self.llm_request.status_code != 200:
                print(f"❌ LLM Error: {self.llm_request.status_code}")
                return
            
            # Add user message to conversation history
            self.conversation_history.append({
                'role': 'user',
                'message': text,
                'timestamp': datetime.now()
            })
            
            # Stream response
            llm_response_text = ""
            for line in self.llm_request.iter_lines():
                if not self.is_llm_responding:  # Check for interruption
                    print("\n[Interrupted]")
                    return
                
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            chunk = data['response']
                            print(chunk, end='', flush=True)
                            llm_response_text += chunk
                            
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Add AI response to conversation history
            if llm_response_text.strip():
                self.conversation_history.append({
                    'role': 'assistant',
                    'message': llm_response_text.strip(),
                    'timestamp': datetime.now()
                })
            
            # Generate voice response (unless suppressed)
            if not suppress_audio:
                print(f"\n🎵 Generating voice response...")
                self.generate_voice_response(llm_response_text)
            else:
                print(f"\n🎭 [Persona initialized - audio suppressed]")
            
            # Just add a newline to separate conversations
            print("\n")
            
            # Update session count if response completed
            if self.is_llm_responding:
                self.session_count += 1
            
        except requests.RequestException as e:
            print(f"❌ LLM Request Error: {e}")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
        finally:
            self.is_llm_responding = False
            self.llm_request = None
    
    def generate_voice_response(self, text):
        """Generate voice response using TTS + RVC"""
        if not text.strip():
            return
            
        try:
            # Step 1: Generate speech with TTS
            print("🎤 TTS generation...", end='', flush=True)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_tts:
                tts_path = tmp_tts.name
            
            # Generate TTS audio
            if "xtts" in self.xtts_model.lower():
                # XTTS models support speaker embedding
                self.tts.tts_to_file(text=text, file_path=tts_path)
            else:
                # Other models
                self.tts.tts_to_file(text=text, file_path=tts_path)
            
            print(" ✅")
            
            # Step 2: Apply RVC voice conversion
            print("🎭 RVC conversion...", end='', flush=True)
            converted_audio = self.convert_with_rvc(tts_path)
            print(" ✅")
            
            # Step 3: Play the converted audio
            print("🔊 Playing audio...", end='', flush=True)
            self.play_audio(converted_audio)
            print(" ✅")
            
            # Cleanup
            try:
                os.unlink(tts_path)
                if converted_audio != tts_path:
                    os.unlink(converted_audio)
            except:
                pass
                
        except Exception as e:
            print(f"\n❌ Voice generation error: {e}")
    
    def convert_with_rvc(self, input_path):
        """Convert audio using RVC"""
        # Make input path absolute before changing directories
        input_path = os.path.abspath(input_path)
        
        # Create output path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            output_path = tmp_out.name
        output_path = os.path.abspath(output_path)
        
        # Change to RVC directory for conversion
        rvc_dir = str(RVC_STS_DIR)
        original_dir = os.getcwd()
        
        try:
            os.chdir(rvc_dir)
            
            # Verify critical RVC models exist (hubert is required)
            hubert_path = str(RVC_HUBERT_PATH)
            
            if not os.path.exists(hubert_path):
                raise FileNotFoundError(f"Hubert model not found: {hubert_path}")
                
            print(f"✅ Hubert model verified: {hubert_path}")
            
            # RVC conversion parameters
            f0up_key = self.pitch_shift  # Pitch adjustment (-12 to +12)
            
            # Choose F0 extraction method based on available models
            if os.path.exists(str(RVC_RMVPE_PATH)):
                f0_method = "rmvpe"  # Best quality
            elif os.path.exists(str(RVC_RMVPE_INPUTS_PATH)):
                f0_method = "rmvpe"  # Alternative RMVPE model
            else:
                f0_method = "harvest"  # Fallback method (doesn't need model files)
                print("⚠️ Using harvest F0 method (RMVPE model not found)")
                
            print(f"🎵 Using F0 method: {f0_method}")
            # Use relative path since we're in rvc_sts directory - .index files are in logs/
            file_index = f"logs/{self.rvc_model['name']}.index" if self.rvc_model['has_index'] else ""
            index_rate = 0.8 if self.rvc_model['has_index'] else 0.0
            resample_sr = 0  # 0 = no resampling
            rms_mix_rate = 0.8  # How much to mix RMS
            protect = 0.33  # Protect voiceless consonants
            
            # Perform RVC conversion
            info, (tgt_sr, audio_output) = self.vc.vc_single(
                sid=0,
                input_audio_path=input_path,
                f0_up_key=f0up_key,
                f0_file=None,  # Let RVC extract F0 automatically
                f0_method=f0_method,
                file_index=file_index,
                file_index2=file_index,
                index_rate=index_rate,
                filter_radius=3,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect
            )
            
            # Save converted audio
            sf.write(output_path, audio_output, tgt_sr)
            
            return output_path
            
        finally:
            os.chdir(original_dir)
    
    def play_audio(self, audio_path):
        """Play audio file"""
        try:
            if self.audio_device and self.audio_device['type'] == 'pulseaudio':
                # Use PulseAudio with specific device
                cmd = ['paplay', '--device', self.audio_device['name'], audio_path]
            else:
                # Use system default
                cmd = ['paplay', audio_path]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            # Fallback to aplay if paplay fails
            try:
                subprocess.run(['aplay', audio_path], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"❌ Could not play audio: {e}")
        except FileNotFoundError:
            print("❌ Audio playback failed: paplay/aplay not found")
    
    def interrupt_llm(self):
        """Interrupt ongoing LLM response"""
        if self.is_llm_responding and self.llm_request:
            self.is_llm_responding = False
            try:
                self.llm_request.close()
            except:
                pass
    
    def update_display(self, force_clear=False):
        """Log-style display - never clear screen, just show status"""
        # Only clear screen on first startup or forced clear
        if force_clear and not hasattr(self, '_display_initialized'):
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🔥💖 CLAUDE AI GIRLFRIEND - Voice-to-Voice Pipeline 💖🔥")
            print(f"Model: {self.whisper_model_name} → {self.ollama_model}")
            print(f"Silence timeout: {self.silence_timeout}s | Interruption: {'On' if self.allow_interruption else 'Off'}")
            print(f"VAD: WebRTC (aggressiveness: {self.vad_aggressiveness}) - Prevents background noise hallucinations")
            print(f"Voice: {self.rvc_model['name']} (pitch: {self.pitch_shift:+d})")
            print("=" * 80)
            print("📜 CONVERSATION LOG:")
            print("-" * 80)
            self._display_initialized = True
            
        # Show minimal status without clearing
        if self.is_llm_responding:
            # Don't print status during LLM response to avoid interfering
            pass
        elif self.is_transcribing and self.current_transcript:
            # Show current transcription progress
            print(f"\r🎤 [{time.strftime('%H:%M:%S')}] Transcribing: {self.current_transcript[:50]}{'...' if len(self.current_transcript) > 50 else ''}", end='', flush=True)
        elif not self.is_transcribing and not self.is_llm_responding:
            # Clear the transcription line when done
            print(f"\r{' ' * 80}\r", end='', flush=True)
    
    def audio_processing_loop(self):
        """Main audio processing loop"""
        last_process_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            # Collect audio data from queue
            try:
                while True:
                    audio_chunk, timestamp = self.audio_queue.get_nowait()
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            except queue.Empty:
                pass
            
            # Process audio if we have enough
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            time_since_process = current_time - last_process_time
            
            if buffer_duration >= self.chunk_duration and time_since_process >= 0.1:
                # Process audio chunk
                text = self.process_audio_chunk(self.audio_buffer)
                
                if text:
                    # User is speaking
                    if not self.is_transcribing:
                        self.is_transcribing = True
                        self.current_transcript = ""
                        
                        # Interrupt LLM if enabled and currently responding
                        if self.allow_interruption and self.is_llm_responding:
                            self.interrupt_llm()
                    
                    # ACCUMULATE transcript (don't replace it!)
                    # Add new text to existing transcript with smart spacing
                    if self.current_transcript:
                        # Check if we should add space (avoid double spaces)
                        if not self.current_transcript.endswith(' ') and not text.startswith(' '):
                            self.current_transcript += " " + text
                        else:
                            self.current_transcript += text
                    else:
                        self.current_transcript = text
                    
                    # Remove duplicate phrases (sometimes Whisper repeats)
                    words = self.current_transcript.split()
                    if len(words) > 6:  # Only dedupe longer transcripts
                        # Simple deduplication - remove if last few words appear earlier
                        last_few = ' '.join(words[-3:])
                        earlier_text = ' '.join(words[:-3])
                        if last_few in earlier_text:
                            # Remove the repetition
                            self.current_transcript = earlier_text + " " + last_few
                    
                    self.last_transcription_time = current_time
                    
                else:
                    # No speech detected - check for silence timeout
                    # Only send to LLM if we were transcribing and there's been silence for the timeout duration
                    if (self.is_transcribing and 
                        self.current_transcript and 
                        current_time - self.last_transcription_time >= self.silence_timeout):
                        
                        # Send to LLM
                        transcript_to_send = self.current_transcript
                        self.is_transcribing = False
                        self.current_transcript = ""
                        
                        # Start LLM response in separate thread
                        if not self.is_llm_responding:
                            # Log the completed transcription
                            print(f"\n👤 [{datetime.now().strftime('%H:%M:%S')}] You: {transcript_to_send}")
                            
                            self.is_llm_responding = True
                            self.llm_thread = threading.Thread(
                                target=self.send_to_llm,
                                args=(transcript_to_send,),
                                daemon=True
                            )
                            self.llm_thread.start()
                
                # Smart buffer management - keep some overlap for continuous speech
                if self.is_transcribing and text:
                    # Keep last 25% of buffer for speech continuity (overlapping chunks)
                    overlap_size = len(self.audio_buffer) // 4
                    self.audio_buffer = self.audio_buffer[-overlap_size:]
                else:
                    # Clear buffer completely when not speaking
                    self.audio_buffer = np.array([], dtype=np.float32)
                    
                last_process_time = current_time
                
                # Update display
                self.update_display()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.05)
    
    def run(self):
        """Start the AI girlfriend voice-to-voice pipeline"""
        print(f"\n🚀 Starting AI Girlfriend Voice-to-Voice Pipeline...")
        print("🗣️ Speak naturally - I'll listen, think, and respond in character voice!")
        print("Press Ctrl+C to exit")
        
        # Initialize log-style display
        self.update_display(force_clear=True)
        
        try:
            # Start audio processing thread
            self.audio_thread = threading.Thread(
                target=self.audio_processing_loop,
                daemon=True
            )
            self.audio_thread.start()
            
            # Start audio input stream
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            ):
                print(f"\n💖 AI Girlfriend is listening... Say something! 💖")
                
                # Keep main thread alive
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            self.is_running = False
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        # Stop LLM request if running
        if hasattr(self, 'llm_request') and self.llm_request:
            try:
                self.llm_request.close()
            except:
                pass
        
        # Terminate Ollama process if we started it
        if hasattr(self, 'ollama_process'):
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
            except:
                try:
                    self.ollama_process.kill()
                except:
                    pass


def main():
    """Main entry point"""
    try:
        pipeline = CLAUDE_AI_GF()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
