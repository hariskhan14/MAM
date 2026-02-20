## Thesis Areas
1. Blockchain-Backed Audit Trails for Multi-Agent Medical AI: Study the design of an immutable logging system for collaborative diagnostics. For example, develop a model where every agent’s input, decision, and retrieved knowledge is recorded on a permissioned blockchain. Research could formalize what events to log and how to structure smart contracts to manage log integrity and access. This ensures traceability and non-repudiation of agent actions. The thesis might analyze trade-offs (e.g. latency, storage) and propose consensus mechanisms suited to healthcare settings. Such work directly addresses auditability and regulatory compliance – making the system HIPAA-audit-ready – while exploring how a distributed ledger can safeguard patient data and diagnostic provenance.


2. Causal/Counterfactual Reasoning in Modular Diagnostic Agents: Develop a formal framework that augments MAM’s dialogue with causal inference. For example, introduce an agent or module dedicated to generating counterfactual scenarios (e.g. “If symptom X were absent, how would the diagnosis change?”) and integrate its reasoning into the group consensus. The thesis could build on counterfactual diagnosis literature to define algorithms enabling LLM-based agents to perform intervention-style reasoning. It might also explore knowledge representation (e.g. causal graphs of disease-symptom links) that agents can query. Incorporating blockchain here could ensure data provenance: e.g., critical reasoning steps or clinical data references used in causal analysis could be timestamped on-chain, enhancing trust in the diagnostic trail (combining trust in data with trust in diagnosis).

4. Dynamic Agent Onboarding via Smart Contracts: Investigate a modular architecture that allows new specialist agents to join the MAM network at runtime. The thesis could define protocols (possibly mediated by smart contracts) for vetting and integrating new agent roles. For example, when a new AI module for a novel medical specialty is developed, it could register its interface and capabilities on a blockchain-based registry. Other agents would then discover and interact with it through defined API schemas. This research would be largely theoretical, addressing questions of interoperability, security (ensuring untrusted agents can be sandboxed), and role hierarchies. Blockchain adds value by enabling decentralized certification of agents – guaranteeing that an added module meets regulatory standards before it participates in diagnosis.

## Remaining Todos
1. Step 1 — Fill in Google Search API credentials
2. Open Google_Search_API_Wrapper/.env and replace the placeholders:
    - api_key=YOUR_ACTUAL_GOOGLE_API_KEY
    - cx=YOUR_ACTUAL_SEARCH_ENGINE_ID
    - To get these:
        - Go to console.cloud.google.com, create a project, enable Custom Search JSON API, generate an API key
        - Go to programmablesearchengine.google.com to get a cx (Search Engine ID)

## Additional installments
- pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
- pip install bitsandbytes
- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
    - True
    - NVIDIA GeForce GTX 1060 6GB


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

The image/audio/video models are not quantized in the original code — they load in bfloat16/float16 which needs 14–16GB. For your thesis research, you have two practical paths:

Text-only pipeline — works now after the fixes above. Fully functional for text-based medical QA research.
Multimodal on a cloud GPU — Google Colab Pro (A100, 40GB) or RunPod/Vast.ai for image/video/audio experiments. This is the typical approach for research with limited local hardware.


## Last Error:

python -c "from model.language_model import MedicalAssistant; MedicalAssistant()"
C:\Users\PATHAN\anaconda3\envs\mam\lib\site-packages\huggingface_hub\file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "E:\Haris\Git\MAM\model\language_model.py", line 12, in __init__
    self.model = AutoModelForCausalLM.from_pretrained(
  File "C:\Users\PATHAN\anaconda3\envs\mam\lib\site-packages\transformers\models\auto\auto_factory.py", line 566, in from_pretrained
    return model_class.from_pretrained(
  File "C:\Users\PATHAN\anaconda3\envs\mam\lib\site-packages\transformers\modeling_utils.py", line 3790, in from_pretrained
    raise ValueError(
ValueError:
                        Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
                        the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
                        these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom
                        `device_map` to `from_pretrained`. Check
                        https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                        for more details.