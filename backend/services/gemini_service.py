import google.generativeai as genai
from utils.config import settings

class GeminiService:
    def __init__(self):
        # Configure Gemini with Flash model
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Use Gemini 2.5 Flash model
        self.model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config={
                "temperature": settings.TEMPERATURE,
                "max_output_tokens": settings.MAX_TOKENS,
            }
        )
        
        # Use latest embedding model (compatible with Flash)
        self.embedding_model = settings.GEMINI_EMBEDDING_MODEL
        
        print(f"✅ Initialized Gemini Service")
        print(f"   Model: {settings.GEMINI_MODEL}")
        print(f"   Embedding Model: {self.embedding_model}")
        print(f"   Embedding Dimension: {settings.EMBEDDING_DIMENSION}")

    async def generate_content(self, prompt: str):
        """Generate content using Gemini 2.5 Flash"""
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini generation error: {e}")
            return f"Error generating response: {str(e)}"

    async def get_embedding(self, text: str):
        """Get embedding using Gemini's embedding model"""
        try:
            # For Gemini 2.5 Flash, use text-embedding-004 or 005
            response = await self.model.embed_content_async(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            print(f"❌ Gemini embedding error: {e}")
            # Return zero vector as fallback
            return [0] * settings.EMBEDDING_DIMENSION

    async def translate_to_urdu(self, text: str):
        """Translate text to Urdu using Gemini Flash"""
        try:
            prompt = f"""Translate the following English text to Urdu. 
            Keep technical terms in English if there's no common Urdu equivalent.
            
            Text: {text}
            
            Urdu Translation:"""
            
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text  # Return original if translation fails

gemini_service = GeminiService()