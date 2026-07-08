import json
import logging
from typing import List, Dict, Any
from ai_service import AIService

logger = logging.getLogger("AgenticChunker")

class AgenticChunker:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    def _call_llm_for_grouping(self, blocks: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Sends a sequence of text blocks to the LLM and asks it to group them
        based on topic shifts and semantic cohesion.
        """
        if not self.ai_service.enabled or not self.ai_service.chat_deployment:
            # Fallback if AI service is not configured
            return [[i] for i in range(len(blocks))]

        # Format blocks for the prompt
        numbered_blocks = ""
        for idx, block in enumerate(blocks):
            numbered_blocks += f"Block [{idx}]: (Page {block['page']}) {block['text'][:400]}\n---\n"

        system_prompt = (
            "You are an expert document structuring agent. Your task is to analyze a sequence of text blocks "
            "and decide how to group them into semantically cohesive, topic-aligned chunks. "
            "Each group must cover a single topic or a continuous line of reasoning. "
            "You must output ONLY valid JSON in the specified format."
        )

        user_prompt = (
            f"Here is a sequence of text blocks extracted from a document:\n\n{numbered_blocks}\n"
            f"Your task is to group consecutive block numbers that belong to the same topic.\n"
            f"Return a JSON object with a 'groups' key containing a list of lists of integers. "
            f"Every block index from 0 to {len(blocks) - 1} must be included in exactly one group in sequential order. "
            f"Do not skip any indexes. For example, if indices 0 and 1 are the same topic, "
            f"and 2 is a new topic, return:\n"
            f'{{"groups": [[0, 1], [2]]}}\n\n'
            f"Response JSON:"
        )

        try:
            # Direct API call to Azure OpenAI
            response = self.ai_service.client.chat.completions.create(
                model=self.ai_service.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            groups = data.get("groups", [])
            
            # Validate groups: must contain integers, cover all indices
            if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
                raise ValueError("Invalid group structure returned by LLM")
                
            all_indices = [idx for group in groups for idx in group]
            if len(all_indices) != len(blocks) or set(all_indices) != set(range(len(blocks))):
                logger.warning("LLM grouping missed some blocks or indices. Using default fallbacks.")
                return [[i] for i in range(len(blocks))]
                
            return groups

        except Exception as e:
            logger.warning(f"Error during agentic chunk grouping call: {e}. Falling back to default split.")
            # Default fallback: each block is its own group
            return [[i] for i in range(len(blocks))]

    def chunk_document(self, doc_data: Dict[str, Any], target_chunk_size: int = 1200) -> List[Dict[str, Any]]:
        """
        Chunks the document using agentic semantic grouping.
        """
        document_name = doc_data.get("document_name", "unknown.pdf")
        text_blocks = []

        # 1. Gather all text elements with their page mapping
        for page in doc_data.get("pages", []):
            page_no = page.get("page_number", 1)
            for item in page.get("items", []):
                item_type = item.get("type", "")
                item_text = item.get("text", "")
                image_text = item.get("image_text", "")

                full_item_text = item_text
                if image_text and image_text != item_text:
                    full_item_text = f"{item_text} [OCR: {image_text}]"

                if item_type in ("page_header", "page_footer") or not full_item_text.strip():
                    continue

                text_blocks.append({
                    "text": full_item_text.strip(),
                    "page": page_no
                })

        if not text_blocks:
            return []

        # 2. Segment blocks using a sliding window for agentic grouping
        # We send blocks in windows (e.g. 15 blocks at a time) to the LLM for semantic group assessment
        window_size = 15
        final_chunks = []
        
        logger.info(f"Agentic chunking started for '{document_name}'. Processing {len(text_blocks)} blocks...")

        i = 0
        while i < len(text_blocks):
            window = text_blocks[i : i + window_size]
            logger.info(f"Agentic chunker grouping blocks {i} to {i + len(window) - 1}...")
            
            groups = self._call_llm_for_grouping(window)
            
            # 3. Create semantic chunks from the LLM grouping results
            for group in groups:
                chunk_blocks = [window[idx] for idx in group]
                chunk_text = " ".join([b["text"] for b in chunk_blocks]).strip()
                pages = sorted(list(set([b["page"] for b in chunk_blocks])))
                
                if not chunk_text:
                    continue
                
                # If a merged chunk is excessively large, split it by characters
                if len(chunk_text) > target_chunk_size * 1.5:
                    sub_chunks = self._split_large_text(chunk_text, target_chunk_size, 200)
                    for sub_txt in sub_chunks:
                        final_chunks.append({
                            "text": sub_txt,
                            "document_name": document_name,
                            "pages": pages,
                            "chunk_index": len(final_chunks),
                            "chunking_method": "agentic_split"
                        })
                else:
                    final_chunks.append({
                        "text": chunk_text,
                        "document_name": document_name,
                        "pages": pages,
                        "chunk_index": len(final_chunks),
                        "chunking_method": "agentic"
                    })
                    
            i += window_size

        logger.info(f"Agentic chunking completed: generated {len(final_chunks)} semantic chunks.")
        return final_chunks

    def _split_large_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            step = max(1, chunk_size - chunk_overlap)
            start += step
        return chunks
