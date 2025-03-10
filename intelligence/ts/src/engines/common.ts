// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import { REMOTE_URL, VERSION } from '../constants';
import { FailureCode, Result } from '../typing';

export async function checkSupport(model: string, engine: string): Promise<Result<string>> {
  try {
    const response = await fetch(REMOTE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        engine_name: engine,
        version: VERSION,
        sdk: 'typescript',
      }),
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `Remote call failed: ${response.statusText}`,
        },
      };
    }

    const data = await response.json();

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
        code: FailureCode.UnsupportedModelError,
        description: `Error calling remote endpoint: ${error}`,
      },
    };
  }
}
