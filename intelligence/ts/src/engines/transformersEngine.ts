// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import {
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  Tensor,
  TextGenerationPipeline,
  TextStreamer,
  pipeline,
} from '@huggingface/transformers';

import type { ProgressInfo, TextGenerationConfig } from '@huggingface/transformers';
import { FailureCode, Message, Result, Progress, ChatResponseResult } from '../typing';

import { BaseEngine } from './engine';

const stoppingCriteria = new InterruptableStoppingCriteria();
const choice = 0;

export class TransformersEngine extends BaseEngine {
  private generationPipelines: Record<string, TextGenerationPipeline> = {};

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    maxCompletionTokens?: number,
    stream?: boolean,
    onStreamEvent?: (event: { chunk: string }) => void
  ): Promise<ChatResponseResult> {
    try {
      if (!(model in this.generationPipelines)) {
        let options = {};
        const modelElems = model.split('|');
        const modelId = modelElems[0];
        if (modelElems.length > 1) {
          options = {
            dtype: modelElems[1],
          };
        }
        this.generationPipelines.model = await pipeline('text-generation', modelId, options);
      }
      const tokenizer = this.generationPipelines.model.tokenizer;
      const modelInstance = this.generationPipelines.model.model;

      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
      }) as {
        input_ids: Tensor | number[] | number[][];
        attention_mask: Tensor | number[] | number[][];
        token_type_ids?: Tensor | number[] | number[][] | undefined;
      };

      let streamer = undefined;
      if (stream && onStreamEvent) {
        streamer = new TextStreamer(tokenizer, {
          skip_prompt: true,
          callback_function: (output: string) => {
            let formattedOutput = output;
            for (const str of tokenizer.special_tokens as string[]) {
              formattedOutput = formattedOutput.replace(str, '');
            }
            onStreamEvent({ chunk: formattedOutput });
          },
        });
      }

      stoppingCriteria.reset();
      const stoppingCriteriaList = new StoppingCriteriaList();
      stoppingCriteriaList.push(stoppingCriteria);
      const { past_key_values: _, sequences } = (await modelInstance.generate({
        ...inputs,
        generation_config: {
          do_sample: false,
          max_new_tokens: maxCompletionTokens ?? 1024,
          temperature: temperature ?? 1,
          return_dict_in_generate: true,
        } as TextGenerationConfig,
        stopping_criteria: stoppingCriteriaList,
        ...(streamer && { streamer }),
      })) as { past_key_values: object; sequences: Tensor };

      const decoded = tokenizer.batch_decode(sequences, {
        skip_special_tokens: true,
      });

      let promptLengths: number[] | undefined;
      const inputIds = inputs.input_ids as Tensor;
      const inputDim = inputIds.dims.at(-1);
      if (typeof inputDim === 'number' && inputDim > 0) {
        promptLengths = tokenizer
          .batch_decode(inputIds, { skip_special_tokens: true })
          .map((x) => x.length);
      }

      if (promptLengths) {
        for (let i = 0; i < decoded.length; ++i) {
          decoded[i] = decoded[i].slice(promptLengths[i]);
        }
      }

      return {
        ok: true,
        message: {
          role: 'assistant',
          content: decoded[choice],
        },
      };
    } catch (error) {
      return {
        ok: false,
        failure: {
          code: FailureCode.LocalEngineChatError,
          description: `TransformersEngine failed with: ${String(error)}`,
        },
      };
    }
  }

  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    try {
      if (!(model in this.generationPipelines)) {
        this.generationPipelines.model = await pipeline('text-generation', model, {
          dtype: 'q4',
          progress_callback: (progressInfo: ProgressInfo) => {
            let percentage = 0;
            let total = 0;
            let loaded = 0;
            let description = progressInfo.status as string;
            if (progressInfo.status == 'progress') {
              percentage = progressInfo.progress;
              total = progressInfo.total;
              loaded = progressInfo.loaded;
              description = progressInfo.file;
            } else if (progressInfo.status === 'done') {
              percentage = 100;
              description = progressInfo.status;
            }
            callback({
              totalBytes: total,
              loadedBytes: loaded,
              percentage,
              description,
            });
          },
        });
      }
      return { ok: true, value: undefined };
    } catch (error) {
      return {
        ok: false,
        failure: { code: FailureCode.LocalEngineFetchError, description: String(error) },
      };
    }
  }
}
