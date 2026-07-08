import logging
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import config

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.enabled = config.is_azure_configured()
        if not self.enabled:
            logger.warning("Azure OpenAI is not fully configured. Embeddings and chat will be disabled.")
            self.client = None
            return

        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        self.embedding_deployment = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.chat_deployment = config.AZURE_OPENAI_CHAT_DEPLOYMENT

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a single text chunk."""
        if not self.enabled or not self.client:
            raise ValueError("Azure OpenAI client is not configured. Please set environment variables in .env.")
        
        # Clean text to ensure there are no weird characters causing issues
        text = text.replace("\r", " ").replace("\n", " ").strip()
        if not text:
            # Return empty embedding (dummy vector) of correct size (3072 for text-embedding-3-large)
            return [0.0] * 3072

        max_retries = 5
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.embedding_deployment
                )
                return response.data[0].embedding
            except Exception as e:
                logger.warning(f"Error calling Azure OpenAI Embedding (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2.0
        return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embedding vectors for a list of text chunks in batches."""
        if not self.enabled or not self.client:
            raise ValueError("Azure OpenAI client is not configured. Please set environment variables in .env.")
        
        if not texts:
            return []

        cleaned_texts = [t.replace("\r", " ").replace("\n", " ").strip() for t in texts]
        # Remove empty items or replace with space to avoid API issues
        cleaned_texts = [t if t else " " for t in cleaned_texts]

        # Azure OpenAI allows batching, but let's send in chunk sizes of 16 to avoid payload size errors
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i+batch_size]
            max_retries = 5
            backoff = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_deployment
                    )
                    # Sort by index to maintain correct order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    batch_embeddings = [item.embedding for item in sorted_data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    logger.warning(f"Error calling Azure OpenAI Embedding Batch (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(backoff)
                    backoff *= 2.0
                    
        return all_embeddings

    def generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate an answer using context retrieved from vector search."""
        if not self.enabled or not self.client:
            return "Azure OpenAI is not configured. Please add your credentials to the .env file."
        
        if not self.chat_deployment:
            return "Error: AZURE_OPENAI_CHAT_DEPLOYMENT is not specified in the .env file. Please specify a chat model deployment (e.g. gpt-4o)."

        # Format context text
        context_str = ""
        for idx, ctx in enumerate(contexts):
            source = ctx.get("metadata", {}).get("document_name", "Unknown Source")
            pages = ctx.get("metadata", {}).get("pages", [])
            page_info = f"Pages: {', '.join(map(str, pages))}" if pages else ""
            context_str += f"[{idx+1}] Source: {source} ({page_info})\nContent: {ctx.get('text', '')}\n\n"

        system_prompt = (
            "You are an intelligent, helpful RAG (Retrieval-Augmented Generation) assistant. "
            "You are answering questions based on the retrieved PDF document sections provided below.\n\n"
            "Here are the rules you must follow:\n"
            "1. Answer the question using ONLY the facts mentioned in the provided context.\n"
            "2. If the context does not contain enough information to answer the question, state clearly that you cannot find the answer in the provided documents.\n"
            "3. When you make a statement based on a context item, cite its source number (e.g., [1], [2]) at the end of the sentence or paragraph.\n"
            "4. Be professional, direct, and detailed in your explanation based on the text.\n\n"
            f"--- START RETRIEVED CONTEXT ---\n{context_str}--- END RETRIEVED CONTEXT ---"
        )

        user_prompt = f"Question: {question}"

        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return f"Error generating answer: {str(e)}"
