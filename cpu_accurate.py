"""
rapidocr_gpu_converter.py
---------------------------
RapidOCR with DirectML GPU acceleration.
- Uses RapidOCR wrapper (correct decoder, proper char dict) which now automatically
  picks up DmlExecutionProvider since onnxruntime-directml is the only runtime installed.
- Uses a background thread to overlap Poppler PDF rendering (CPU) with GPU inference,
  eliminating most of the idle gap between GPU spikes.
"""
import threading
import queue
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_PDF        = r"path/to/your/input.pdf"
OUTPUT_DIR       = r"path/to/your/output_folder"
POPPLER_BIN_PATH = r"C:\path\to\poppler\bin"
DPI              = 150
CHUNK_SIZE       = 5
# =================================================

_SENTINEL = None  # signals the render thread is done

def ensure_paths_exist():
    input_path = Path(INPUT_PDF)
    output_dir_path = Path(OUTPUT_DIR)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_filename = input_path.stem + "_rapidocr_dml.md"
    return output_dir_path / output_filename

def render_worker(total_pages, result_queue):
    """
    Background thread: renders chunks of PDF pages to numpy arrays using Poppler (CPU).
    Puts (page_number, numpy_array) tuples onto the queue for the GPU to consume.
    """
    for i in range(1, total_pages + 1, CHUNK_SIZE):
        images = convert_from_path(
            INPUT_PDF,
            first_page=i,
            last_page=min(i + CHUNK_SIZE - 1, total_pages),
            poppler_path=POPPLER_BIN_PATH,
            dpi=DPI
        )
        for idx, image in enumerate(images):
            img_np = np.array(image.convert('RGB'))
            result_queue.put((i + idx, img_np))
            image.close()
    result_queue.put(_SENTINEL)  # signal done

def convert_pdf_chunked():
    try:
        final_output_file = ensure_paths_exist()
        print(f"Target Output: {final_output_file}")

        print("Initializing RapidOCR (DmlExecutionProvider active)...")
        engine = RapidOCR()

        print("Counting pages...")
        info = pdfinfo_from_path(INPUT_PDF, poppler_path=POPPLER_BIN_PATH)
        total_pages = info["Pages"]
        print(f"Total Pages: {total_pages}")

        # Start the Poppler render thread immediately
        img_queue = queue.Queue(maxsize=10)  # buffer up to 10 pre-rendered pages in RAM
        render_thread = threading.Thread(
            target=render_worker,
            args=(total_pages, img_queue),
            daemon=True
        )
        render_thread.start()

        with open(final_output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {final_output_file.stem}\n\n")

        processed = 0
        with open(final_output_file, 'a', encoding='utf-8') as f:
            with tqdm(total=total_pages, desc="OCR Progress") as pbar:
                while True:
                    item = img_queue.get()
                    if item is _SENTINEL:
                        break

                    page_num, img_np = item

                    # Resize image to fit within VRAM budget of MX110 (2GB).
                    # Detection network input is the bottleneck; capping at 1200px on
                    # longest side keeps intermediate tensors under ~300MB VRAM.
                    max_side = 1200
                    h, w = img_np.shape[:2]
                    if max(h, w) > max_side:
                        scale = max_side / max(h, w)
                        import cv2
                        img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)))

                    # GPU inference via RapidOCR (DmlExecutionProvider)
                    result, _ = engine(img_np)

                    f.write(f"\n\n--- Page {page_num} ---\n\n")
                    if result:
                        for line in result:
                            # line = [bounding_box, text, confidence]
                            _, text, conf = line
                            if text and text.strip():
                                f.write(text.strip() + "\n")
                    else:
                        f.write("<!-- No text detected -->\n")

                    processed += 1
                    pbar.update(1)

        render_thread.join()
        print(f"\nDone! {processed}/{total_pages} pages saved to:\n{final_output_file}")

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    convert_pdf_chunked()
