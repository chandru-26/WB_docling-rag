import os
import json
import logging
from typing import List, Dict, Any, Set
from glob import glob
from tqdm import tqdm

import config
from parser import PDFExtractor
from ai_service import AIService
from vector_db import VectorDB
from agentic_chunker import AgenticChunker


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IngestionPipeline")

class IngestionPipeline:
    def __init__(self, ai_service=None, vector_db=None):
        self.ai_service = ai_service or AIService()
        self.vector_db = vector_db or VectorDB()
        self.input_dir = config.INPUT_DIR
        self.output_dir = config.OUTPUT_DIR
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def split_large_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a large block of text into smaller overlapping chunks."""
        if not text:
            return []
        
        # Simple character-based sliding window chunking
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            
            # Avoid infinite loop if overlap >= size
            step = max(1, chunk_size - chunk_overlap)
            start += step
            
        return chunks

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks the document sections while mapping back to source pages and headers.
        """
        document_name = doc_data.get("document_name", "unknown.pdf")
        text_blocks = []

        # Gather all text items from all pages
        for page in doc_data.get("pages", []):
            page_no = page.get("page_number", 1)
            for item in page.get("items", []):
                item_type = item.get("type", "")
                item_text = item.get("text", "")
                image_text = item.get("image_text", "")

                # Combine OCR text and base text
                full_item_text = item_text
                if image_text and image_text != item_text:
                    full_item_text = f"{item_text} [OCR: {image_text}]"

                # Skip header/footer and empty values to prevent clutter
                if item_type in ("page_header", "page_footer") or not full_item_text.strip():
                    continue

                text_blocks.append({
                    "text": full_item_text.strip(),
                    "page": page_no
                })

        if not text_blocks:
            return []

        chunks = []
        current_blocks = []
        current_len = 0

        def save_current_chunk(blocks):
            if not blocks:
                return
            chunk_text = " ".join([b["text"] for b in blocks]).strip()
            # Extract unique page numbers in sorted order
            pages = sorted(list(set([b["page"] for b in blocks])))
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "document_name": document_name,
                    "pages": pages,
                    "chunk_index": len(chunks)
                })

        for block in text_blocks:
            block_text = block["text"]
            block_page = block["page"]

            # If a single item is longer than chunk size, split it
            if len(block_text) > self.chunk_size:
                # Flush the current buffer first
                if current_blocks:
                    save_current_chunk(current_blocks)
                    current_blocks = []
                    current_len = 0

                # Split the large block
                sub_texts = self.split_large_text(block_text, self.chunk_size, self.chunk_overlap)
                for sub_txt in sub_texts:
                    chunks.append({
                        "text": sub_txt,
                        "document_name": document_name,
                        "pages": [block_page],
                        "chunk_index": len(chunks)
                    })
            else:
                # If adding this block exceeds limit, save current chunk and start a new one with overlap
                if current_len + len(block_text) > self.chunk_size and current_blocks:
                    save_current_chunk(current_blocks)
                    
                    # Backtrack to satisfy overlap requirements
                    overlap_blocks = []
                    overlap_len = 0
                    for b in reversed(current_blocks):
                        if overlap_len + len(b["text"]) <= self.chunk_overlap:
                            overlap_blocks.insert(0, b)
                            overlap_len += len(b["text"])
                        else:
                            break
                    current_blocks = overlap_blocks
                    current_len = overlap_len

                current_blocks.append(block)
                current_len += len(block_text) + 1  # Add 1 for the spacing

        # Save any trailing blocks
        if current_blocks:
            save_current_chunk(current_blocks)

        return chunks

    def process_single_pdf(self, pdf_path: str) -> bool:
        """Processes a single PDF file: parse, chunk, embed, and upload."""
        pdf_basename = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_basename)[0]
        json_cache_path = os.path.join(self.output_dir, f"{pdf_name_no_ext}.json")

        try:
            doc_data = None

            # 1. Parse PDF or load cached extraction
            if os.path.exists(json_cache_path):
                logger.info(f"Loading cached text extraction for '{pdf_basename}'...")
                try:
                    with open(json_cache_path, "r", encoding="utf-8") as f:
                        doc_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read cached JSON for {pdf_basename}: {e}. Reparsing...")

            if not doc_data:
                logger.info(f"Running Docling parser on '{pdf_basename}'...")
                extractor = PDFExtractor(pdf_path, self.output_dir)
                doc_data = extractor.extract_pdf()

            # 2. Chunk text
            logger.info(f"Chunking '{pdf_basename}' using strategy '{config.CHUNKING_STRATEGY}'...")
            if config.CHUNKING_STRATEGY == "agentic":
                chunker = AgenticChunker(self.ai_service)
                chunks = chunker.chunk_document(doc_data, target_chunk_size=self.chunk_size)
            else:
                chunks = self.chunk_document(doc_data)
            logger.info(f"Generated {len(chunks)} chunks for '{pdf_basename}'.")

            if not chunks:
                logger.warning(f"No text extracted or chunked for '{pdf_basename}'. Skipping vector storage.")
                return True

            # 3. Create embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks using Azure OpenAI...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.ai_service.get_embeddings_batch(chunk_texts)

            # 4. Store in Qdrant
            logger.info(f"Storing chunks in Qdrant...")
            self.vector_db.upsert_chunks(chunks, embeddings)
            logger.info(f"Successfully finished processing '{pdf_basename}'.")
            return True

        except Exception as e:
            logger.error(f"Error processing PDF '{pdf_basename}': {e}", exc_info=True)
            return False

    def run(self):
        """Runs the ingestion pipeline for all PDF files in the input directory."""
        if not self.ai_service.enabled:
            logger.error("Pipeline cannot run because Azure OpenAI is not configured in .env.")
            print("\n[ERROR] Azure OpenAI is not configured. Please fill in your details in the .env file.\n")
            return

        # Find all PDF files in input directory
        pdf_files = glob(os.path.join(self.input_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in '{self.input_dir}' directory.")
            print(f"\n[INFO] Place your PDF files in the '{self.input_dir}' folder and run this script again.\n")
            return

        logger.info(f"Found {len(pdf_files)} PDF(s) in '{self.input_dir}'.")

        # Get already ingested docs
        ingested_docs = self.vector_db.get_ingested_documents()
        logger.info(f"Found {len(ingested_docs)} document(s) already indexed in Qdrant.")

        # Filter out already ingested docs
        to_process = []
        for path in pdf_files:
            # We compare with base name of PDF
            pdf_basename = os.path.basename(path)
            if pdf_basename in ingested_docs:
                logger.info(f"Skipping '{pdf_basename}' (already indexed).")
            else:
                to_process.append(path)

        if not to_process:
            logger.info("All found PDFs are already indexed. Nothing to do!")
            print("\n[SUCCESS] All PDF documents are up-to-date in the database.\n")
            return

        logger.info(f"Starting ingestion for {len(to_process)} document(s)...")
        success_count = 0
        
        for idx, pdf_path in enumerate(to_process):
            print(f"\n[{idx+1}/{len(to_process)}] Processing: {os.path.basename(pdf_path)}")
            success = self.process_single_pdf(pdf_path)
            if success:
                success_count += 1

        print(f"\n[SUMMARY] Ingestion complete. Successfully indexed {success_count}/{len(to_process)} document(s).")
        stats = self.vector_db.get_db_stats()
        print(f"Total documents now in Qdrant: {stats['total_documents']}")
        print(f"Total vector chunks: {stats['total_chunks']}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
