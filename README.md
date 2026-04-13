# Photopdf2mrkdwn

This repository provides scripts to convert image-based (photo/scanned) PDFs into formatted Markdown text. It includes optimized scripts for both GPU-equipped and CPU-only systems, ensuring high accuracy and performance across different hardware configurations.

---

## Contents

1. [Scripts Overview](#scripts-overview)
    - [batch_pdf_ocr.py (Dedicated GPU)](#batch_pdf_ocrpy-dedicated-gpu)
    - [cpu_accurate.py (CPU Only / No Dedicated GPU)](#cpu_accuratepy-cpu-only--no-dedicated-gpu)
2. [How to Run](#how-to-run)
3. [Configuration Guide (Paths & Environment)](#configuration-guide-paths--environment)
4. [Technical Details: Monkey-Patching & GPU Logic](#technical-details-monkey-patching--gpu-logic)
5. [Common Issues & Fixes](#common-issues--fixes)

---

## Scripts Overview

### batch_pdf_ocr.py (Dedicated GPU)

This script is designed for systems equipped with a **dedicated NVIDIA GPU**. It is optimized for high-speed batch processing of multiple PDF files.

- **Purpose**: Batch conversion of folders containing PDFs using GPU acceleration.
- **Hardware Enforced**: Specifically targets the dedicated GPU (`device_id=1`) for maximum performance.
- **Stability**: Includes advanced memory management (RAM/VRAM) features such as a Memory Guard and periodic engine recycling to prevent crashes on systems with limited resources.

### cpu_accurate.py (CPU Only / No Dedicated GPU)

This script is intended for systems **without a dedicated or external GPU**. It is optimized to provide the highest possible accuracy while running entirely on the CPU.

- **Purpose**: Accurate single-file conversion for users without a dedicated graphics card.
- **Efficiency**: Designed to maintain low resource spikes (rarely hitting eGPU or iGPU heavily) while ensuring the OCR remains precise.
- **Threading**: Uses background threading to overlap PDF rendering with OCR processing, keeping the engine busy without overloading the system.

---

## How to Run

1.  **Open a Terminal**: Open PowerShell or your preferred command line tool.
2.  **Navigate to the Project**: Use `cd` to enter your project folder.
3.  **Run the Script**:
    - For GPU users: `python batch_pdf_ocr.py`
    - For CPU users: `python cpu_accurate.py`

---

## Configuration Guide (Paths & Environment)

### Setting Up File Paths

You must set the correct file paths inside each script before running them. Open the script file in a text editor and look for the `CONFIGURATION` section at the top.

**Critical Rule**: Always use **raw strings** in Python when writing Windows paths. Add an `r` before the opening quote.
- **Correct**: `r"C:\Folder\PDF_Source"`
- **Incorrect**: `"C:\Folder\PDF_Source"`

**Variables to Update:**
- `INPUT_FOLDER` or `INPUT_PDF`: The location of your scanned PDFs.
- `OUTPUT_FOLDER` or `OUTPUT_DIR`: The destination for your Markdown files.
- `POPPLER_BIN_PATH`: The path to the `bin` directory of your Poppler installation.

### Environment Requirements
Ensure you have the following libraries installed:
```powershell
pip install rapidocr-onnxruntime onnxruntime-directml opencv-python pdf2image tqdm psutil numpy
```

---

## Technical Details: Monkey-Patching & GPU Logic

### How the Monkey-Patching Works
The **RapidOCR** library's internal code is designed to only check for **CUDA** (NVIDIA's proprietary toolkit). If it doesn't find a full CUDA installation, it defaults to the CPU.

To bypass this without forcing you to edit the library's files, the scripts perform a **Monkey-Patch** at startup:
1.  **Intercepting Initialization**: The scripts target the `OrtInferSession.__init__` method inside the `rapidocr_onnxruntime` library.
2.  **Injecting DirectML**: Instead of the default behavior, the scripts inject a custom initialization function that explicitly adds `DmlExecutionProvider` (Windows DirectX 12) to the list of available providers.
3.  **Forcing the Device**: It passes the specific `device_id` (e.g., `1` for your dedicated NVIDIA GPU) directly into the engine, overriding the library's CPU-only default.

---

### How will the user know it's working?
The scripts provide **explicit visual confirmation** in the terminal immediately upon startup. A user will know the monkey-patch and GPU are active if they see the following lines:

```text
[DmlExecutionProvider:device=1] ch_PP-OCRv3_det_infer.onnx
[DmlExecutionProvider:device=1] ch_ppocr_mobile_v2.0_cls_infer.onnx
[DmlExecutionProvider:device=1] ch_PP-OCRv3_rec_infer.onnx
```
This confirms that every internal model (Detection, Classification, and Recognition) has been successfully "forced" onto the GPU via the monkey-patch.

If the user sees `CPUExecutionProvider` instead of `DmlExecutionProvider`, it means the patch failed or the GPU was unavailable, and the script (in the case of `batch_pdf_ocr.py`) will halt to prevent slow CPU processing.

---

## Common Issues & Fixes

### 1. "ModuleNotFoundError: No module named 'frontend'"
This is a common conflict with the `fitz` library. You must uninstall the dummy package and install the correct one:
```powershell
pip uninstall fitz
pip install PyMuPDF
```

### 2. ONNX Runtime Conflicts
If the script ignores your GPU, you may have both the CPU and GPU versions of the runtime installed. Force-reinstall the DirectML version:
```powershell
pip uninstall onnxruntime
pip install onnxruntime-directml --force-reinstall
```

---

## FOR AGENT

### Instructions for AI Agent
If you are an AI agent tasked with running these scripts, please follow these mandates:
1. **Hardware Check**: Prioritize `batch_pdf_ocr.py` if a dedicated NVIDIA GPU is available (typically `device_id=1`). Use `cpu_accurate.py` for CPU-only systems or if the dedicated GPU is unavailable.
2. **Path Configuration**: Always use raw strings (`r"..."`) when updating paths in the `CONFIGURATION` section of the scripts to ensure Windows compatibility.
3. **Validation**: Before running, ensure that `onnxruntime-directml` is the only ONNX runtime installed to avoid provider conflicts.
4. **Monitoring**: Watch for the `[DmlExecutionProvider:device=1]` logs to confirm GPU activation. If `CPUExecutionProvider` appears on a GPU-intended run, stop and diagnose immediately.

### Creating and Updating `.agentskills`
To help the agent remember these project-specific workflows and technical nuances, run the following command in your terminal to create a `.agentskills` file:

```powershell
Set-Content -Path ".agentskills" -Value @"
# Photopdf2mrkdwn Task Skills
- Workflow: PDF to Markdown OCR conversion.
- GPU Enforcement: Use batch_pdf_ocr.py with device_id=1 for dedicated NVIDIA cards.
- CPU Fallback: Use cpu_accurate.py for high-precision OCR on systems without a dedicated GPU.
- Path Format: Always use Python raw strings (r"C:\path") for configuration.
- Verification: Confirm DmlExecutionProvider:device=1 appears in terminal logs at startup.
- Memory Guard: Respect ENGINE_RECYCLE_EVERY and MIN_FREE_RAM_MB settings for system stability.
"@
```

To update your current session's memory or a new agent's context with these skills, you can use the following instruction:
> "Read the .agentskills file and the README.md technical details to understand the specialized GPU monkey-patching and path configuration required for this project."

