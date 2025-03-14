List of available models
========================

This document provides an overview of the available models, their providers, and
corresponding aliases. Note that all models are also supported via remote inference.

Models
------

=================================== ============= ================ ======== ================ ============= =================================================================== ===============================================================
**Model**                           **Publisher** **Model Family** **Size** **Quantization** **Precision** **Web support (On-Device)**                                         **Node.JS support (On-Device)**
=================================== ============= ================ ======== ================ ============= =================================================================== ===============================================================
``meta/llama3.2-1b/instruct-fp16``  Meta          Llama 3.2        1B       None             Float 16      `Llama-3.2-1B-Instruct-q0f16-MLC                                    `onnx-community/Llama-3.2-1B-Instruct
                                                                                                           <https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC>`_   <https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct>`_
``meta/llama3.2-3b/instruct-q4``    Meta          Llama 3.2        3B       Q4               Float 16      `Llama-3.2-3B-Instruct-q4f16_1-MLC                                  –
                                                                                                           <https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC>`_
``meta/llama3.1-8b/instruct-q4``    Meta          Llama 3.1        8B       Q4               Float 16      `Llama-3.1-8B-Instruct-q4f16_1-MLC                                  –
                                                                                                           <https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC>`_
``deepseek/r1-distill-llama-8b/q4`` DeepSeek      R1               8B       Q4               Float 16      `Llama-3.1-8B-Instruct-q4f16_1-MLC                                  –
                                                                                                           <https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC>`_
=================================== ============= ================ ======== ================ ============= =================================================================== ===============================================================

Aliases
-------

The following table lists all aliases for each canonical model:

=================================== ================================================
**Canonical Model**                 **Aliases**
=================================== ================================================
``meta/llama3.2-1b/instruct-fp16``  ``meta/llama3.2-1b``, ``meta/llama3.2-1b/fp16``,
                                    ``meta/llama3.2-1b/instruct``
``meta/llama3.2-3b/instruct-q4``    ``meta/llama3.2-3b/q4``,
                                    ``meta/llama3.2-3b/instruct``
``meta/llama3.1-8b/instruct-q4``    ``meta/llama3.1-8b``, ``meta/llama3.1-8b/q4``,
                                    ``meta/llama3.1-8b/instruct``
``deepseek/r1-distill-llama-8b/q4`` ``deepseek/r1``, ``deepseek/r1-distill``,
                                    ``deepseek/r1-distill-llama``,
                                    ``deepseek/r1-distill-llama-8b``
=================================== ================================================
