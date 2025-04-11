import re

class TextProcessor:
    @staticmethod
    def clean_text(text):
        """Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles."""
        text = re.sub(r'\s+', ' ', text)  # Suppression des espaces multiples
        text = re.sub(r'[^\w\s\.,;:!?]', '', text)  # Suppression des caractères spéciaux
        return text.strip()
