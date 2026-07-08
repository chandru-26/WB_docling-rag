import io
import fitz
import easyocr
import numpy as np
from PIL import Image
from docling.document_converter import DocumentConverter

print('opening converter')
converter = DocumentConverter()
res = converter.convert('input/sample.pdf')
doc = res.document

print('found pictures', len(doc.pictures))
if not doc.pictures:
    print('no pictures')
    raise SystemExit(0)

p = doc.pictures[0]
prov = p.prov[0]
print('prov type', type(prov))
print('page_no', prov.page_no)
bb = prov.bbox
print('bbox raw', bb)

bbox = {'l': float(bb.l), 't': float(bb.t), 'r': float(bb.r), 'b': float(bb.b), 'coord_origin': getattr(bb, 'coord_origin', None)}
print('bbox', bbox)

pdf = fitz.open('input/sample.pdf')
page = pdf.load_page(prov.page_no - 1)
# expand
l = bbox['l']; r = bbox['r']; t = bbox['t']; b = bbox['b']
w = max(1.0, r - l); h = max(1.0, t - b)
pad_x = max(5.0, w * 1.0)
pad_y = max(8.0, h * 2.0)
expanded = {'l': max(0.0, l - pad_x), 'r': r + pad_x, 't': t + pad_y, 'b': max(0.0, b - pad_y), 'coord_origin': bbox['coord_origin']}
print('expanded', expanded)

# get rect
page_height = page.rect.height
x0 = expanded['l']; x1 = expanded['r']; y0 = expanded['t']; y1 = expanded['b']
if expanded['coord_origin'] is not None and str(expanded['coord_origin']).upper().endswith('BOTTOMLEFT'):
    y0 = page_height - y0
    y1 = page_height - y1
rect = fitz.Rect(x0, y0, x1, y1)
print('rect', rect, rect.width, rect.height)

mat = fitz.Matrix(3, 3)
pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
print('pix size', pix.width, pix.height)

png = pix.tobytes('png')
print('png bytes', len(png))
image = Image.open(io.BytesIO(png)).convert('RGB')
arr = np.array(image)
print('arr shape', arr.shape)

reader = easyocr.Reader(['en'], gpu=False)
print('running ocr...')
res = reader.readtext(arr)
print('ocr result', res)
if res:
    print('joined', ' '.join([r[1] for r in res if len(r) > 1]))

# save image for inspection
image.save('output/extracted_images/test_crop.png')
print('saved test_crop.png')
