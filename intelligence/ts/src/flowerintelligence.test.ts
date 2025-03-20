// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import { vi } from 'vitest';

vi.mock('./constants', () => ({
  DEFAULT_MODEL: 'meta/llama3.2-1b/instruct-fp16',
  REMOTE_URL: process.env.FI_DEV_REMOTE_URL,
  VERSION: '0.1.7',
  SDK: 'TS',
}));

import emojiRegex from 'emoji-regex';
import { describe, expect, it, beforeEach, assert } from 'vitest';
import { FlowerIntelligence } from './flowerintelligence';
import { RemoteEngine } from './engines/remoteEngine';
import { TransformersEngine } from './engines/transformersEngine';
import { FailureCode } from './typing';

describe('FlowerIntelligence', () => {
  let fi: FlowerIntelligence;

  beforeEach(() => {
    fi = new FlowerIntelligence();
    fi.remoteHandoff = true;
    fi.apiKey = process.env.FI_API_KEY ?? '';
  });

  describe('getEngine', () => {
    it('should return a remote engine when forceRemote is true', async () => {
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, false);
      expect(getEngineRes.ok).toBe(true);
      if (getEngineRes.ok) {
        expect(getEngineRes.value).toBeInstanceOf(RemoteEngine);
      }
    });

    it('should return a local engine when the model can run locally', async () => {
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', false, false);
      expect(getEngineRes.ok).toBe(true);
      if (getEngineRes.ok) {
        expect(getEngineRes.value).toBeInstanceOf(TransformersEngine);
      }
    });

    it('should throw an error if the model cannot run locally or remotely', async () => {
      fi.remoteHandoff = false;
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, false);
      expect(getEngineRes.ok).toBe(false);
      if (!getEngineRes.ok) {
        expect(Math.floor(getEngineRes.failure.code / 100) * 100).toBe(FailureCode.ConfigError);
      }
    });

    it('should throw an error if forceLocal and forceRemote are both true', async () => {
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, true);
      expect(getEngineRes.ok).toBe(false);
      if (!getEngineRes.ok) {
        expect(getEngineRes.failure.description).toBe(
          'The `forceLocal` and `forceRemote` options cannot be true at the same time.'
        );
      }
    });

    // it('should throw an error if the model ID is invalid', async () => {
    //   const getEngineRes = await fi['getEngine']('invalid-model', false, false);
    //   expect(getEngineRes.ok).toBe(false);
    //   if (!getEngineRes.ok) {
    //     expect(getEngineRes.failure.code).toBe(FailureCode.UnknownModelError);
    //   }
    // });
  });

  describe('chooseLocalEngine', () => {
    it('should return a local engine for a valid provider', async () => {
      const chooseEngineRes = await fi['chooseLocalEngine']('meta/llama3.2-1b/instruct-fp16');
      expect(chooseEngineRes.ok).toBe(true);
      if (chooseEngineRes.ok) {
        expect(chooseEngineRes.value).toBeInstanceOf(TransformersEngine);
      }
    });
  });

  describe('Chat', () => {
    it('generates some text', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content:
              'Can you write a single sentence email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
      });
      expect(data).not.toBeNull();
    });
    it('generates some text from encrypted request', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content:
              'Can you write a single sentence email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
        encrypt: true,
      });
      expect(data).not.toBeNull();
    });
    it('generates some reduced text', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content: 'Can you write an email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
        maxCompletionTokens: 5,
      });
      if (!data.ok) {
        assert.fail(data.failure.description);
      } else {
        expect(data.message.content).not.toBeNull();
        if (data.message.content) {
          expect(data.message.content.trim().split(/\s+/)).length.lessThanOrEqual(5);
        }
      }
    });
    it('generates some text with custom system prompt', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that writes emails full of emojies',
          },
          {
            role: 'user',
            content:
              'Can you write a single sentence email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
      });
      const regex = emojiRegex();

      if (!data.ok) {
        assert.fail(data.failure.description);
      } else {
        expect(data.message.content).not.toBeNull();
        if (data.message.content) {
          expect(regex.test(data.message.content)).toBe(true);
        }
      }
    });
  });
});
