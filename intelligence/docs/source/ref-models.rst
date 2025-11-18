##########################
 List of available models
##########################

This document provides an overview of the available models, their providers, and
corresponding aliases. Note that all models are also supported via remote inference.

********
 Models
********

========================================== ============= ================ ======== ========== ============= ============================================================================== ========================================================================= ==============================================================================
**Model**                                  **Publisher** **Model Family** **Size** **Quant.** **Precision** **Web (On-Device)**                                                            **Node.JS (On-Device)**                                                   **MLX-Swift (On-Device)**
========================================== ============= ================ ======== ========== ============= ============================================================================== ========================================================================= ==============================================================================
``meta/llama3.2-1b/instruct-q4``           Meta          Llama 3.2        1B       Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC>`__        `✔ <https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-q4f16>`__ `✔ <https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit>`__
``meta/llama3.2-1b/instruct-fp16``         Meta          Llama 3.2        1B       None       Float 16      `✔ <https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC>`__          `✔ <https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct>`__       –
``meta/llama3.2-1b/instruct-bf16``         Meta          Llama 3.2        1B       None       BFloat 16     –                                                                              –                                                                         `✔ <https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16>`__
``meta/llama3.2-3b/instruct``              Meta          Llama 3.2        3B       None       Float 16      –                                                                              –                                                                         `✔ <https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct>`__
``meta/llama3.2-3b/instruct-q4``           Meta          Llama 3.2        3B       Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC>`__        –                                                                         `✔ <https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit>`__
``meta/llama3.1-8b/instruct-q4``           Meta          Llama 3.1        8B       Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC>`__        –                                                                         `✔ <https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit>`__
``huggingface/smollm2-135m/instruct-fp16`` HuggingFace   SmolLM2          135M     None       Float 16      `✔ <https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC>`__          `✔ <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>`__        `✔ <https://huggingface.co/mlx-community/SmolLM-135M-Instruct-fp16>`__
``huggingface/smollm2-360m/instruct-q4``   HuggingFace   SmolLM2          360M     Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC>`__        `✔ <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>`__        `✔ <https://huggingface.co/mlx-community/SmolLM-360M-Instruct-4bit>`__
``huggingface/smollm2-360m/instruct-fp16`` HuggingFace   SmolLM2          360M     None       Float 16      `✔ <https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q0f16-MLC>`__          `✔ <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>`__        `✔ <https://huggingface.co/mlx-community/SmolLM-360M-Instruct-fp16>`__
``huggingface/smollm2-1.7b/instruct-q4``   HuggingFace   SmolLM2          1.7B     Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC>`__        `✔ <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct>`__        `✔ <https://huggingface.co/mlx-community/SmolLM-1.7B-Instruct-4bit>`__
``deepseek/r1-distill-llama-8b/q4``        DeepSeek      R1               8B       Q4         Float 16      `✔ <https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC>`__ –                                                                         `✔ <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit>`__
========================================== ============= ================ ======== ========== ============= ============================================================================== ========================================================================= ==============================================================================

*********
 Aliases
*********

The following table lists all aliases for each canonical model:

========================================== ================================
**Canonical Model**                        **Aliases**
========================================== ================================
``meta/llama3.2-1b/instruct-q4``           ``meta/llama3.2-1b``
``meta/llama3.2-3b/instruct-q4``           ``meta/llama3.2-3b``
``meta/llama3.1-8b/instruct-q4``           ``meta/llama3.1-8b``
``huggingface/smollm2-135m/instruct-fp16`` ``huggingface/smollm2-135m``
``huggingface/smollm2-360m/instruct-q4``   ``huggingface/smollm2-360m``
``huggingface/smollm2-1.7b/instruct-q4``   ``huggingface/smollm2-1.7b``
``deepseek/r1-distill-llama-8b/q4``        ``deepseek/r1-distill-llama-8b``
========================================== ================================
