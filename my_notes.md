## Remaining todo
Step 1 — Fill in Google Search API credentials
Open Google_Search_API_Wrapper/.env and replace the placeholders:


api_key=YOUR_ACTUAL_GOOGLE_API_KEY
cx=YOUR_ACTUAL_SEARCH_ENGINE_ID
To get these:

Go to console.cloud.google.com, create a project, enable Custom Search JSON API, generate an API key
Go to programmablesearchengine.google.com to get a cx (Search Engine ID)

## Additional installments
- pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    - verify: python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"


## To speed up the downloading
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`\



## Limitations
### GPU Limitations
GPU: NVIDIA GeForce GTX 1060 6GB
CUDA Version: 12.9 (driver version 576.52)
Currently using: 1458MiB / 6144MiB (only 4.6GB free)
This is actually a critical limitation for running this project. The models involved are:

Medichat-Llama3-8B: ~16GB VRAM in float16
HuatuoGPT-Vision-7B: ~14GB VRAM
VideoLLaMA2-7B: ~14GB VRAM
Qwen-Audio-Chat: ~8GB VRAM

None of these will fit in 6GB VRAM—even with 4-bit quantization, a 7B model typically needs ~4-5GB, but I'm loading in bfloat16 which requires 14-16GB for these model sizes. The GTX 1060 6GB is fundamentally undersized for this workload, and I need to reconsider the approach.
With quantization I could fit a 7B model in about 4-5GB, but the current setup doesn't use it, the GPU is older Pascal architecture with limited bfloat16 support, and I only have ~4.5GB free after existing allocations. I should either add 4-bit or 8-bit quantization using bitsandbytes, reinstall PyTorch with proper CUDA support, or fall back to CPU inference if speed isn't critical.
For the thesis work, the priority is getting quantization working to squeeze the model into VRAM, ensuring CUDA 12.1 is properly installed since it's compatible with the driver version available. I need to reconsider the hardware constraints here—the HuatuoGPT-Vision model requires 14GB VRAM with bfloat16, which is well beyond what the GTX 1060 can handle, and the GPU doesn't even support bfloat16 efficiently anyway. For text-only work with 4-bit quantization, the Medichat model might squeeze into 6GB, but the vision, audio, and video models are essentially impossible on this hardware without major compromises.

You have a GTX 1060 6GB with CUDA 12.9. Two problems to fix:

PyTorch installed as CPU-only — needs the CUDA build
All 4 model files hardcode device="cuda" with no fallback, and the 8B model won't fit in 6GB VRAM unquantized
Critical VRAM reality: Medichat-Llama3-8B in float16 = ~16GB VRAM. Your GPU has 6GB. You need 4-bit quantization (~4–5GB) to fit it. Let me fix all the model files:




