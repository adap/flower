# Federated Instruction Tuning
---
model:
  name: "mistralai/Mistral-7B-v0.3"
  quantization: 8 # 8 or 4 if you want to do quantization with BitsAndBytes
  gradient_checkpointing: True
  lora:
    peft_lora_r: 32
    peft_lora_alpha: 64

train:
  num_rounds: null
  save_every_round: 5
  learning_rate_max: 5e-5
  learning_rate_min: 1e-6
  seq_length: 512
  training_arguments:
    output_dir: null # to be set by hydra
    learning_rate: null # to be set by the client
    per_device_train_batch_size: 16
    gradient_accumulation_steps: 1
    logging_steps: 10
    num_train_epochs: 3
    max_steps: 10
    report_to: null
    save_steps: 1000
    save_total_limit: 10
    gradient_checkpointing: True
    lr_scheduler_type: "constant"

strategy:
  _target_: flwr.server.strategy.FedAvg
  fraction_fit: $fraction_fit
  fraction_evaluate: 0.0 # no client evaluation
