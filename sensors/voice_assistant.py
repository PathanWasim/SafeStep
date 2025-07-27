"""
Voice Assistant Module
=====================

Speech recognition and text-to-speech with error handling and fallback mechanisms.
"""

import time
import threading
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Advanced voice assistant with NLP capabilities and error handling"""
    
    def __init__(self, enable_speech: bool = True, enable_tts: bool = True):
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.enabled = False
        self.speech_enabled = False
        self.tts_enabled = False
        
        # Voice commands mapping
        self.commands = {
            'navigate': self._handle_navigation_command,
            'where': self._handle_location_query,
            'what': self._handle_object_query,
            'help': self._handle_help_command,
            'stop': self._handle_stop_command,
            'emergency': self._handle_emergency_command
        }
        
        self.listening = False
        self.last_command_time = time.time()
        
        if enable_speech:
            self._init_speech_recognition()
        
        if enable_tts:
            self._init_text_to_speech()
    
    def _init_speech_recognition(self):
        """Initialize speech recognition with error handling"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.speech_enabled = True
            logger.info("Speech recognition initialized successfully")
            
        except ImportError as e:
            logger.warning(f"SpeechRecognition not available: {e}")
            self.speech_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            self.speech_enabled = False
    
    def _init_text_to_speech(self):
        """Initialize text-to-speech with error handling"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            self.tts_enabled = True
            logger.info("Text-to-speech initialized successfully")
            
        except ImportError as e:
            logger.warning(f"pyttsx3 not available: {e}")
            self.tts_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech: {e}")
            self.tts_enabled = False
    
    def start_listening(self):
        """Start continuous voice recognition"""
        if not self.speech_enabled:
            logger.warning("Speech recognition not available")
            return
        
        self.listening = True
        thread = threading.Thread(target=self._listen_continuously)
        thread.daemon = True
        thread.start()
        logger.info("Voice recognition started")
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
        logger.info("Voice recognition stopped")
    
    def speak(self, text: str, priority: str = "normal"):
        """Text-to-speech with priority handling"""
        if not self.tts_enabled:
            logger.warning(f"TTS not available, cannot speak: {text}")
            return
        
        try:
            if priority == "emergency":
                # Clear queue for emergency messages
                self.tts_engine.stop()
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
    
    def _listen_continuously(self):
        """Continuous listening loop"""
        if not self.speech_enabled or self.recognizer is None or self.microphone is None:
            return
        
        try:
            import speech_recognition as sr
            
            with self.microphone as source:
                # Calibrate once
                self.recognizer.adjust_for_ambient_noise(source)
                
                while self.listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            logger.info(f"Voice command: {text}")
                            self._process_command(text)
                        except sr.UnknownValueError:
                            pass
                        except sr.RequestError as e:
                            logger.error(f"Speech recognition error: {e}")
                    except sr.WaitTimeoutError:
                        pass
                    except Exception as e:
                        logger.error(f"Voice assistant error: {e}")
                        time.sleep(1)
                        
        except Exception as e:
            logger.error(f"Continuous listening failed: {e}")
    
    def _process_command(self, text: str):
        """Process voice command"""
        for command, handler in self.commands.items():
            if command in text:
                handler(text)
                break
    
    def _handle_navigation_command(self, text: str):
        """Handle navigation commands"""
        if "bathroom" in text or "restroom" in text:
            self.speak("Navigating to nearest restroom")
        elif "exit" in text or "door" in text:
            self.speak("Searching for nearest exit")
        elif "elevator" in text:
            self.speak("Locating elevator")
        else:
            self.speak("Please specify your destination")
    
    def _handle_location_query(self, text: str):
        """Handle location queries"""
        self.speak("You are currently in the main corridor")
    
    def _handle_object_query(self, text: str):
        """Handle object identification queries"""
        self.speak("I can see several objects around you. Please be more specific")
    
    def _handle_help_command(self, text: str):
        """Handle help requests"""
        self.speak("I'm here to help you navigate. You can ask me to find exits, restrooms, or describe your surroundings")
    
    def _handle_stop_command(self, text: str):
        """Handle stop commands"""
        self.speak("Navigation stopped")
    
    def _handle_emergency_command(self, text: str):
        """Handle emergency commands"""
        self.speak("Emergency mode activated. Finding nearest exit", priority="emergency")
    
    def is_speech_enabled(self) -> bool:
        """Check if speech recognition is enabled"""
        return self.speech_enabled
    
    def is_tts_enabled(self) -> bool:
        """Check if text-to-speech is enabled"""
        return self.tts_enabled
    
    def is_enabled(self) -> bool:
        """Check if voice assistant is enabled"""
        return self.speech_enabled or self.tts_enabled 