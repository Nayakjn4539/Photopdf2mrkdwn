"""
batch_pdf_ocr.py
-----------------
Batch OCR converter using RapidOCR on NVIDIA MX110 (DirectML).

Stability guarantees for low-RAM (4GB) / low-VRAM (2GB) hardware:
- NVIDIA-only: hard fails with diagnostics if NVIDIA unavailable
- Single-page rendering: Poppler renders 1 page at a time = minimal RAM spike
- VRAM recycling: engine is fully destroyed and rebuilt every ENGINE_RECYCLE_EVERY
  pages, flushing DirectML's internal VRAM allocator pool (which hoards memory)
- Memory guard: checks free RAM before each page; waits if critically low
- Graceful error reporting: prints last successful page number on any crash
"""
import cv2
import gc
import time
import sys
import numpy as np
import psutil
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

# ── Enforce NVIDIA GPU (device_id=1) — hard fail if unavailable ──────────────
def _require_nvidia_dml(model_path: str) -> int:
    NVIDIA_DEVICE_ID = 1
    try:
        sess = ort.InferenceSession(
            model_path,
            providers=[('DmlExecutionProvider', {'device_id': NVIDIA_DEVICE_ID}),
                       'CPUExecutionProvider']
        )
        active = sess.get_providers()
        del sess
        gc.collect()
        if 'DmlExecutionProvider' in active:
            print(f"[GPU] NVIDIA confirmed on DirectML device_id={NVIDIA_DEVICE_ID} ✓")
            return NVIDIA_DEVICE_ID
        else:
            print("\n" + "="*60)
            print("ERROR: NVIDIA GPU (device_id=1) is NOT available right now.")
            print("="*60)
            print("Reason: DirectML reported 887A0004 = DXGI_ERROR_UNSUPPORTED")
            print("        This usually means:")
            print("  1. The NVIDIA GPU is powered off by Windows Optimus/battery saver.")
            print("  2. The GPU driver was reset or updated since last run.")
            print("  3. Another process is holding the D3D12 device exclusively.")
            print("\nFixes to try:")
            print("  - Open NVIDIA Control Panel → Manage 3D Settings →")
            print("    Set 'Preferred graphics processor' to 'High-performance NVIDIA'.")
            print("  - Switch from Battery Saver mode to Balanced or Performance.")
            print("  - Reboot once and try again.")
            print("  - Check Device Manager to verify MX110 is not disabled.")
            print("="*60 + "\n")
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print("\n" + "="*60)
        print(f"ERROR: Failed to initialise NVIDIA GPU (device_id=1): {e}")
        print("="*60 + "\n")
        raise SystemExit(1)

import rapidocr_onnxruntime.utils as _rutils
from pathlib import Path as _Path
_cls_model = str(_Path(_rutils.__file__).parent / 'models' / 'ch_ppocr_mobile_v2.0_cls_infer.onnx')
_DML_DEVICE_ID = _require_nvidia_dml(_cls_model)

def _dml_init(self, config):
    sess_opt = SessionOptions()
    sess_opt.log_severity_level = 4
    sess_opt.enable_cpu_mem_arena = False
    sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    _rutils.OrtInferSession._verify_model(config['model_path'])

    providers = [
        ('DmlExecutionProvider', {'device_id': _DML_DEVICE_ID}),
        'CPUExecutionProvider'
    ]
    self.session = InferenceSession(
        config['model_path'], sess_options=sess_opt, providers=providers
    )
    model_name = config['model_path'].split('\\')[-1]
    print(f"  [{self.session.get_providers()[0]}:device={_DML_DEVICE_ID}] {model_name}")

_rutils.OrtInferSession.__init__ = _dml_init
# ─────────────────────────────────────────────────────────────────────────────

from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
from pathlib import Path
from tqdm import tqdm

# ======================== CONFIGURATION ========================
INPUT_FOLDER     = r"path/to/your/pdf_source"
OUTPUT_FOLDER    = r"path/to/your/output_folder"
POPPLER_BIN_PATH = r"C:\path\to\poppler\bin"

DPI              = 150
MAX_SIDE         = 900   # VRAM guard — keeps det model tensors under ~300MB

# Recycle (destroy + rebuild) the ONNX engine every N pages.
# DirectML's internal allocator pools VRAM and never voluntarily releases it.
# Recycling at this interval forces a full VRAM flush.
ENGINE_RECYCLE_EVERY = 10

# Minimum free RAM in MB before pausing to let the OS recover memory.
MIN_FREE_RAM_MB  = 400

# Set True to reprocess PDFs that already have an output file.
FORCE_RECONVERT  = True
# ===============================================================


def make_engine() -> RapidOCR:
    """Create a fresh RapidOCR engine with DirectML on NVIDIA."""
    return RapidOCR()


def destroy_engine(engine: RapidOCR):
    """Fully destroy an engine and flush VRAM."""
    del engine
    gc.collect()
    time.sleep(0.5)   # brief pause lets DirectML finish releasing D3D12 resources


def resize_for_det(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def check_memory(page_num: int):
    """
    Block until free RAM is above MIN_FREE_RAM_MB.
    Prints a warning and waits 3s if memory is tight.
    """
    while True:
        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        if free_mb >= MIN_FREE_RAM_MB:
            return
        tqdm.write(
            f"  [MEM GUARD] Page {page_num}: only {free_mb:.0f}MB RAM free. "
            f"Waiting for OS to recover memory..."
        )
        gc.collect()
        time.sleep(3)


def format_page_text(result) -> str:
    """
    Groups bounding boxes on the same horizontal line and joins them with spaces.
    """
    if not result:
        return ""
    items = []
    for detection in result:
        box, text, conf = detection
        if not text or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min  = min(xs)
        y_min  = min(ys)
        y_max  = max(ys)
        cy     = (y_min + y_max) / 2
        height = y_max - y_min
        items.append((x_min, cy, height, text.strip()))

    if not items:
        return ""

    items.sort(key=lambda x: x[1])
    lines = []
    current_line = [items[0]]
    for item in items[1:]:
        if abs(item[1] - current_line[-1][1]) < current_line[-1][2] * 0.6:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    output_lines = []
    for line_group in lines:
        line_group.sort(key=lambda x: x[0])
        output_lines.append("  ".join(item[3] for item in line_group))

    return "\n".join(output_lines)


def convert_pdf(pdf_path: Path, out_path: Path):
    """Convert a single PDF, recycling the engine every ENGINE_RECYCLE_EVERY pages."""
    try:
        info = pdfinfo_from_path(str(pdf_path), poppler_path=POPPLER_BIN_PATH)
        total_pages = info["Pages"]
    except Exception as e:
        print(f"  Could not read page count: {e}")
        return False

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path.stem}\n\n")

    errors      = 0
    last_ok     = 0
    engine      = make_engine()
    pages_since_recycle = 0

    with open(out_path, 'a', encoding='utf-8') as f:
        with tqdm(total=total_pages, desc="  Pages", unit="pg", leave=False) as pbar:
            for page_num in range(1, total_pages + 1):

                # ── Memory guard ─────────────────────────────────────────────
                check_memory(page_num)

                # ── VRAM recycle ─────────────────────────────────────────────
                if pages_since_recycle >= ENGINE_RECYCLE_EVERY:
                    tqdm.write(
                        f"  [VRAM FLUSH] Recycling engine at page {page_num} "
                        f"(every {ENGINE_RECYCLE_EVERY} pages)"
                    )
                    destroy_engine(engine)
                    engine = make_engine()
                    pages_since_recycle = 0

                try:
                    # Render exactly ONE page — minimum RAM impact
                    images = convert_from_path(
                        str(pdf_path),
                        first_page=page_num,
                        last_page=page_num,
                        poppler_path=POPPLER_BIN_PATH,
                        dpi=DPI
                    )
                    # Immediately convert and discard PIL image
                    img_np = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                    images[0].close()
                    del images
                    gc.collect()

                    img_np = resize_for_det(img_np)
                    result, _ = engine(img_np)

                    # Free the input array before writing (keeps peak RAM flat)
                    del img_np
                    gc.collect()

                    page_text = format_page_text(result)
                    f.write(f"\n\n--- Page {page_num} ---\n\n")
                    f.write(page_text if page_text else "<!-- No text detected -->")
                    f.flush()   # write to disk immediately — don't buffer

                    last_ok = page_num
                    pages_since_recycle += 1

                except Exception as e:
                    errors += 1
                    msg = (
                        f"\n{'='*55}\n"
                        f"INFERENCE ERROR at page {page_num}\n"
                        f"Last successful page : {last_ok}\n"
                        f"PDF                  : {pdf_path.name}\n"
                        f"Reason               : {e}\n"
                        f"Resume tip: Set start page to {last_ok + 1}\n"
                        f"{'='*55}"
                    )
                    tqdm.write(msg)
                    f.write(f"\n\n--- Page {page_num} ---\n\n<!-- ERROR: {e} -->\n")
                    f.flush()

                pbar.update(1)

    destroy_engine(engine)
    return errors == 0


def main():
    input_dir  = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)

    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        sys.exit(0)

    print(f"\nFound {len(pdf_files)} PDF(s) in '{input_dir}'")
    for i, p in enumerate(pdf_files, 1):
        print(f"  {i}. {p.name}")
    print(f"\nEngine recycles every {ENGINE_RECYCLE_EVERY} pages to flush VRAM.")
    print(f"Memory guard pauses if free RAM < {MIN_FREE_RAM_MB}MB.\n")

    done = skipped = with_errors = 0

    for pdf_path in tqdm(pdf_files, desc="PDFs", unit="pdf"):
        out_path = output_dir / (pdf_path.stem + "_ocr.md")

        if out_path.exists() and not FORCE_RECONVERT:
            tqdm.write(f"[SKIP] {pdf_path.name} → already converted")
            skipped += 1
            continue

        tqdm.write(f"\n[START] {pdf_path.name}")
        success = convert_pdf(pdf_path, out_path)

        if success:
            tqdm.write(f"[DONE]  → {out_path.name}")
            done += 1
        else:
            tqdm.write(f"[WARN]  → {out_path.name} (some pages errored)")
            with_errors += 1

    print(f"\n{'='*50}")
    print(f"Batch complete!")
    print(f"  Converted    : {done}")
    print(f"  Skipped      : {skipped}")
    print(f"  With errors  : {with_errors}")
    print(f"  Output dir   : {output_dir}")


if __name__ == "__main__":
    main()
