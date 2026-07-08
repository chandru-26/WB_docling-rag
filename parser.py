import io
import os
import json

import fitz
import easyocr
import numpy as np
from PIL import Image

from docling.document_converter import DocumentConverter


class PDFExtractor:

    def __init__(self, pdf_path, output_dir):

        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, "extracted_images")

        os.makedirs(self.images_dir, exist_ok=True)

        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------

    def _get_item_page_numbers(self, item):

        page_numbers = set()

        prov = getattr(item, "prov", None)

        if isinstance(prov, list):

            for entry in prov:

                # support both dict-style provenance and ProvenanceItem objects
                if isinstance(entry, dict) and entry.get("page_no") is not None:
                    page_numbers.add(entry["page_no"])
                else:
                    # try object-like access (docling ProvenanceItem)
                    page_no = getattr(entry, "page_no", None)
                    if page_no is not None:
                        page_numbers.add(page_no)

        if not page_numbers and hasattr(item, "page_no"):

            page_numbers.add(getattr(item, "page_no"))

        return sorted(page_numbers) if page_numbers else [1]

    def _get_page_rect(self, page, bbox):

        page_height = page.rect.height

        x0 = bbox["l"]

        x1 = bbox["r"]

        y0 = bbox["t"]

        y1 = bbox["b"]

        coord_origin = bbox.get("coord_origin")

        if coord_origin is not None and str(coord_origin).upper().endswith("BOTTOMLEFT"):

            y0 = page_height - y0

            y1 = page_height - y1

        return fitz.Rect(x0, y0, x1, y1)

    def _ocr_image_region(self, pdf_document, page_no, bbox):

        try:
            page = pdf_document.load_page(page_no - 1)

            # Normalize bbox values whether passed as dict or object
            try:
                l = float(bbox["l"])
                r = float(bbox["r"])
                t = float(bbox["t"])
                b = float(bbox["b"])
                coord_origin = bbox.get("coord_origin") if isinstance(bbox, dict) else getattr(bbox, 'coord_origin', None)
            except Exception:
                bb = bbox
                l = float(getattr(bb, "l", bb[0] if isinstance(bb, (list, tuple)) else 0.0))
                r = float(getattr(bb, "r", bb[2] if isinstance(bb, (list, tuple)) else 0.0))
                t = float(getattr(bb, "t", bb[1] if isinstance(bb, (list, tuple)) else 0.0))
                b = float(getattr(bb, "b", bb[3] if isinstance(bb, (list, tuple)) else 0.0))
                coord_origin = getattr(bb, 'coord_origin', None)

            w = max(1.0, abs(r - l))
            h = max(1.0, abs(t - b))

            # Try multiple scales and slightly different padding if the region is small
            scales = [3, 4, 6]
            pad_multipliers = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]

            final_text = ""
            for scale, pads in zip(scales, pad_multipliers):
                pad_x = max(8.0, w * pads[0])
                pad_y = max(10.0, h * pads[1])

                expanded = {
                    "l": max(0.0, l - pad_x),
                    "r": r + pad_x,
                    "t": t + pad_y,
                    "b": max(0.0, b - pad_y),
                    "coord_origin": coord_origin,
                }

                rect = self._get_page_rect(page, expanded)

                if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                    print(f"[OCR] skipping empty rect on page {page_no}: {rect}")
                    continue

                try:
                    matrix = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(clip=rect, matrix=matrix, alpha=False)
                    png_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                except Exception as e:
                    print(f"[OCR] render failed (scale={scale}): {e}")
                    continue

                # Save cropped image for debugging
                try:
                    fname = f"{os.path.splitext(os.path.basename(self.pdf_path))[0]}_p{page_no}_{int(rect.x0)}_{int(rect.y0)}_s{scale}.png"
                    img_path = os.path.join(self.images_dir, fname)
                    image.save(img_path)
                    print(f"[OCR] saved cropped image to {img_path}")
                except Exception as e:
                    print(f"[OCR] failed saving cropped image: {e}")

                try:
                    results = self.ocr_reader.readtext(np.array(image))
                except Exception as e:
                    print(f"[OCR] readtext failed (scale={scale}): {e}")
                    results = []

                if not results:
                    print(f"[OCR] no results for page {page_no} rect {rect} (scale={scale})")
                    continue

                texts = []
                confidences = []
                for res in results:
                    # easyocr returns [bbox, text, prob]
                    if isinstance(res, (list, tuple)) and len(res) >= 2:
                        if len(res) == 3:
                            _, txt, prob = res
                        else:
                            # sometimes returns [text, prob]
                            txt = res[1]
                            prob = res[2] if len(res) > 2 else None
                    else:
                        continue
                    texts.append(str(txt).strip())
                    try:
                        confidences.append(float(prob))
                    except Exception:
                        pass

                merged = " ".join(t for t in texts if t).strip()
                avg_conf = sum(confidences) / len(confidences) if confidences else None

                print(f"[OCR] page {page_no} (scale={scale}) -> {len(texts)} segments, avg_conf={avg_conf}")

                if merged:
                    final_text = merged
                    break

            # If still empty, as a last resort run OCR on the entire page at high res
            if not final_text:
                try:
                    full_mat = fitz.Matrix(2, 2)
                    full_pix = page.get_pixmap(matrix=full_mat, alpha=False)
                    full_png = full_pix.tobytes("png")
                    full_image = Image.open(io.BytesIO(full_png)).convert("RGB")
                    try:
                        full_results = self.ocr_reader.readtext(np.array(full_image))
                    except Exception as e:
                        print(f"[OCR] full page readtext failed: {e}")
                        full_results = []

                    if full_results:
                        full_texts = [r[1] for r in full_results if isinstance(r, (list, tuple)) and len(r) > 1]
                        final_text = " ".join(t for t in full_texts if t).strip()
                        if final_text:
                            print(f"[OCR] page {page_no} whole-page fallback -> {len(full_texts)} segments")
                except Exception as e:
                    print(f"[OCR] whole-page fallback failed: {e}")

            return final_text

        except Exception as e:
            print(f"[OCR] unexpected error on page {page_no}: {e}")
            return ""

    def _extract_items_by_page(self, document, pdf_document):

        page_items = {

            page_no: [] for page_no in sorted(document.pages.keys())

        }

        item_sources = [

            ("texts", "text"),

            ("tables", "table"),

            ("pictures", "picture"),

            ("key_value_items", "key_value"),

            ("form_items", "form_item"),

            ("field_items", "field_item"),

            ("groups", "group"),

        ]

        for source_name, default_type in item_sources:

            for item in getattr(document, source_name, []):

                page_numbers = self._get_item_page_numbers(item)

                item_type = getattr(item.label, "value", None) or str(

                    getattr(item, "label", default_type)

                )

                text = getattr(item, "text", None) or getattr(item, "orig", None)

                image_text = ""

                if source_name == "pictures":

                    prov = getattr(item, "prov", None)

                    bbox = None

                    if isinstance(prov, list):



                        for entry in prov:

                            # dict-style provenance
                            if isinstance(entry, dict) and entry.get("bbox") is not None:
                                bbox = entry["bbox"]
                                page_from_prov = entry.get("page_no") or entry.get("page")
                                break
                            else:
                                # ProvenanceItem object
                                page_from_prov = getattr(entry, "page_no", None)
                                bb = getattr(entry, "bbox", None)
                                if bb is not None:
                                    # convert bounding object to plain dict
                                    try:
                                        bbox = {
                                            "l": float(bb.l),
                                            "t": float(bb.t),
                                            "r": float(bb.r),
                                            "b": float(bb.b),
                                            "coord_origin": getattr(bb, "coord_origin", None),
                                        }
                                    except Exception:
                                        bbox = bb
                                    break

                    if bbox is not None:

                        image_text = self._ocr_image_region(

                            pdf_document, page_numbers[0], bbox

                        )
                    else:
                        # no bbox discovered; image_text remains empty
                        image_text = ""

                    if not text and image_text:

                        text = image_text

                    if text is None:

                        captions = getattr(item, "captions", None)

                        if captions:

                            text = "; ".join(

                                str(c.caption) if hasattr(c, "caption") else str(c)

                                for c in captions

                            )

                if text is None:

                    text = default_type

                for page_no in page_numbers:

                    item_record = {

                        "type": item_type,

                        "text": text,

                    }

                    if source_name == "pictures":
                        # always include image_text key (may be empty)
                        item_record["image_text"] = image_text

                    page_items.setdefault(page_no, []).append(item_record)

        return page_items

    def extract_pdf(self):

        self.ocr_reader = easyocr.Reader(["en"], gpu=False)

        converter = DocumentConverter()

        result = converter.convert(self.pdf_path)

        document = result.document

        pdf_document = fitz.open(self.pdf_path)

        page_items = self._extract_items_by_page(document, pdf_document)

        output = {

            "document_name": os.path.basename(self.pdf_path),

            "pages": []

        }

        for page_no, page in sorted(document.pages.items()):

            page_data = {

                "page_number": page_no,

                "items": page_items.get(page_no, []),

            }

            output["pages"].append(page_data)

        json_path = os.path.join(

            self.output_dir,

            "document.json"

        )

        with open(

            json_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                output,

                f,

                indent=4,

                ensure_ascii=False

            )

        print("Saved:", json_path)

        return output


if __name__ == "__main__":

    extractor = PDFExtractor(

        "input/sample.pdf",

        "output"

    )

    extractor.extract_pdf()