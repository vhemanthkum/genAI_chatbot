import spacy

class MedicalNER:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            self.nlp = None

    def extract_entities(self, text):
        """
        Extracts medical-like entities from the text.
        With en_core_web_sm, we'll look for general entities like ORG, PRODUCT, EVENT, or simply noun chunks.
        For a specialized model like en_core_sci_sm, it would extract 'DISEASE', 'CHEMICAL', etc.
        """
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        # We can extract standard named entities
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
            
        # We can also extract noun chunks as a fallback for medical terms
        # (e.g. "type 2 diabetes", "severe headache")
        if not entities:
            for chunk in doc.noun_chunks:
                # Filter out pronouns and basic nouns if we want, but keeping it simple
                if chunk.root.pos_ == "NOUN":
                    entities.append({"text": chunk.text, "label": "MEDICAL_TERM_CANDIDATE"})
                    
        # Remove duplicates
        unique_entities = []
        seen = set()
        for e in entities:
            if e['text'].lower() not in seen:
                unique_entities.append(e)
                seen.add(e['text'].lower())
                
        return unique_entities

if __name__ == "__main__":
    ner = MedicalNER()
    test_text = "What are the common treatments for Glaucoma and Diabetes?"
    print(f"Extracting entities from: '{test_text}'")
    print(ner.extract_entities(test_text))
