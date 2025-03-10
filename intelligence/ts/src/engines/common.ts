// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import { REMOTE_URL, SDK, VERSION } from '../constants';
import { FailureCode, Result } from '../typing';

interface ModelResponse {
  is_supported: boolean;
  engine_model: string;
  model: string;
}

export async function checkSupport(model: string, engine: string): Promise<Result<string>> {
  try {
    const response = await fetch(REMOTE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        engine_name: engine,
        sdk_version: VERSION,
        sdk: SDK,
      }),
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.RemoteError,
          description: `Remote call failed: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as ModelResponse;

    if (data.is_supported) {
      return { ok: true, value: data.engine_model };
    } else {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `Model '${model}' is not supported on the webllm engine.`,
        },
      };
    }
  } catch (error) {
    return {
      ok: false,
      failure: {
        code: FailureCode.ConnectionError,
        description: `Error calling remote endpoint: ${String(error)}`,
      },
    };
  }
}
