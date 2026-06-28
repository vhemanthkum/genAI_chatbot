from langdetect import detect_langs
import logging

# Dictionary mapping common ISO 639-1 language codes to English names
LANG_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'nl': 'Dutch',
    'ko': 'Korean',
    'tr': 'Turkish',
    'te': 'Telugu'
}

def detect_language(text: str) -> dict:
    """
    Detects the primary language of the text.
    Handles potential code-switching by returning a primary label but acknowledging mixtures.
    """
    if not text or not text.strip():
        return {"code": "en", "name": "English", "confidence": 1.0, "is_mixed": False}
        
    try:
        # Get top predictions
        langs = detect_langs(text)
        primary = langs[0]
        
        primary_code = primary.lang
        primary_conf = primary.prob
        
        is_mixed = False
        if len(langs) > 1 and langs[1].prob > 0.2:
            is_mixed = True
            
        return {
            "code": primary_code,
            "name": LANG_MAP.get(primary_code, primary_code.upper()),
            "confidence": primary_conf,
            "is_mixed": is_mixed
        }
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return {"code": "unknown", "name": "Unknown", "confidence": 0.0, "is_mixed": False}
