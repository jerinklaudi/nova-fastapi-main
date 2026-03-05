import os
import tempfile
from typing import List, Dict, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)

class AudioFeedbackService:
    """Service for generating audio feedback from detection results."""
    
    @staticmethod
    def generate_object_description(detections: List[Dict]) -> str:
        """Generate a textual description of detected objects."""
        if not detections:
            return "No objects detected."
        
        # Group detections by label
        label_counts = {}
        for detection in detections:
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            if label not in label_counts:
                label_counts[label] = []
            label_counts[label].append(confidence)
        
        # Generate description
        descriptions = []
        for label, confidences in label_counts.items():
            count = len(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            
            if count == 1:
                descriptions.append(f"{label} with {avg_confidence:.0%} confidence")
            else:
                descriptions.append(f"{count} {label}s with {avg_confidence:.0%} average confidence")
        
        if len(descriptions) == 1:
            return f"I see a {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"I see a {descriptions[0]} and a {descriptions[1]}."
        else:
            return f"I see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
    
    @staticmethod
    def generate_face_description(faces: List[Dict]) -> str:
        """Generate a textual description of detected faces."""
        if not faces:
            return "No faces detected."
        
        descriptions = []
        for i, face in enumerate(faces):
            confidence = face.get('confidence', 0.0)
            person_id = face.get('person_id')
            
            if person_id:
                descriptions.append(f"person {person_id} with {confidence:.0%} confidence")
            else:
                descriptions.append(f"unknown person with {confidence:.0%} confidence")
        
        if len(descriptions) == 1:
            return f"I see {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"I see {descriptions[0]} and {descriptions[1]}."
        else:
            return f"I see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
    
    @staticmethod
    def generate_combined_description(objects: List[Dict], faces: List[Dict]) -> str:
        """Generate a combined description of objects and faces."""
        object_desc = AudioFeedbackService.generate_object_description(objects)
        face_desc = AudioFeedbackService.generate_face_description(faces)
        
        if "No objects detected" in object_desc and "No faces detected" in face_desc:
            return "No objects or faces detected."
        elif "No objects detected" in object_desc:
            return face_desc
        elif "No faces detected" in face_desc:
            return object_desc
        else:
            return f"{object_desc} {face_desc}"
    
    @staticmethod
    def text_to_speech(text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Convert text to speech using system TTS."""
        try:
            # Try different TTS engines based on platform
            import platform
            
            if platform.system() == "Windows":
                return AudioFeedbackService._windows_tts(text, output_path)
            elif platform.system() == "Darwin":  # macOS
                return AudioFeedbackService._macos_tts(text, output_path)
            else:  # Linux
                return AudioFeedbackService._linux_tts(text, output_path)
                
        except Exception as e:
            logger.error(f"TTS conversion failed: {str(e)}")
            return None
    
    @staticmethod
    def _windows_tts(text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Windows TTS using PowerShell."""
        try:
            import subprocess
            
            if not output_path:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # PowerShell command for TTS
            powershell_cmd = f'''
            $speak = New-Object -ComObject SAPI.SpVoice
            $stream = New-Object SAPI.SpFileStream
            $stream.Open("{output_path}", 3, $false)
            $speak.AudioOutputStream = $stream
            $speak.Speak("{text}")
            $stream.Close()
            '''
            
            subprocess.run(['powershell', '-Command', powershell_cmd], 
                         check=True, capture_output=True)
            
            logger.info(f"Windows TTS completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Windows TTS failed: {str(e)}")
            return None
    
    @staticmethod
    def _macos_tts(text: str, output_path: Optional[str] = None) -> Optional[str]:
        """macOS TTS using say command."""
        try:
            import subprocess
            
            if not output_path:
                output_path = tempfile.mktemp(suffix='.aiff')
            
            # macOS say command
            subprocess.run(['say', '-o', output_path, text], 
                         check=True, capture_output=True)
            
            logger.info(f"macOS TTS completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"macOS TTS failed: {str(e)}")
            return None
    
    @staticmethod
    def _linux_tts(text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Linux TTS using espeak or festival."""
        try:
            import subprocess
            
            if not output_path:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Try espeak first
            try:
                subprocess.run(['espeak', '-w', output_path, text], 
                             check=True, capture_output=True)
                logger.info(f"Linux espeak TTS completed: {output_path}")
                return output_path
            except subprocess.CalledProcessError:
                # Fall back to festival
                festival_cmd = f'(echo "{text}" | text2wave -o {output_path})'
                subprocess.run(['bash', '-c', festival_cmd], 
                             check=True, capture_output=True)
                logger.info(f"Linux festival TTS completed: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Linux TTS failed: {str(e)}")
            return None
    
    @staticmethod
    def cleanup_audio_file(file_path: str) -> None:
        """Clean up temporary audio file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up audio file: {str(e)}")