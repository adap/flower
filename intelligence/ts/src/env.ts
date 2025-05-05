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

import { FailureCode, Result } from './typing';

/* eslint-disable @typescript-eslint/no-unnecessary-condition */
export const isNode: boolean = typeof process !== 'undefined' && process.versions?.node != null;
/* eslint-enable @typescript-eslint/no-unnecessary-condition */

export async function getAvailableRAM(): Promise<Result<number>> {
  if (typeof window !== 'undefined' && 'navigator' in window && 'deviceMemory' in navigator) {
    // deviceMemory is in GB, it can only be one of 0.25, 0.5, 1, 2, 4, 8
    const totalRAM = navigator.deviceMemory as number;
    // Assume 50% is available, convert GB to MB
    return { ok: true, value: Math.floor((totalRAM * 1024) / 2) };
  }

  if (isNode) {
    const os = await import('os');
    // Convert bytes to MB
    return { ok: true, value: Math.floor(os.freemem() / (1024 * 1024)) };
  }

  return {
    ok: false,
    failure: {
      code: FailureCode.LocalError,
      description: 'Unsupported environment: Cannot determine available RAM.',
    },
  };
}
