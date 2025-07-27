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

import { FailureCode, Result } from '../../typing';
import { getTimestamp } from './cryptoUtils';

/** Handles server communication for key exchange */
export class NetworkService {
  private serverUrl: string;
  private apiKey: string;
  private serverPublicKey: string | null = null;
  private serverPublicKeyExpiresAt: number | null = null;
  private clientPublicKeyExpiresAt: number | null = null;

  constructor(serverUrl: string, apiKey: string) {
    this.serverUrl = serverUrl;
    this.apiKey = apiKey;
  }

  private isServerKeyExpired(): boolean {
    if (!this.serverPublicKeyExpiresAt) return true;
    // Conversion from milliseconds to microseconds
    return Date.now() >= this.serverPublicKeyExpiresAt * 1000;
  }

  isClientKeyExpired(): boolean {
    if (!this.clientPublicKeyExpiresAt) return true;
    // Conversion from milliseconds to microseconds
    return Date.now() >= this.clientPublicKeyExpiresAt * 1000;
  }

  async submitClientPublicKey(clientPublicKey: string): Promise<Result<string>> {
    const response = await fetch(`${this.serverUrl}/v1/encryption/public-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({ public_key_base64: clientPublicKey }),
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: `Failed to send public key: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as { expires_at: string; encryption_id: string };
    this.clientPublicKeyExpiresAt = getTimestamp(data.expires_at);
    return { ok: true, value: data.encryption_id };
  }

  async getServerPublicKey(): Promise<Result<string>> {
    if (this.isServerKeyExpired() || !this.serverPublicKey) await this.fetchNewServerPublicKey();
    if (!this.serverPublicKey) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: 'Public key is not set.' },
      };
    }
    return { ok: true, value: this.serverPublicKey };
  }

  private async fetchNewServerPublicKey(): Promise<Result<void>> {
    const response = await fetch(`${this.serverUrl}/v1/encryption/server-public-key`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: `Failed to fetch server public key: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as { public_key_base64: string; expires_at: string };
    this.serverPublicKey = data.public_key_base64;
    this.serverPublicKeyExpiresAt = getTimestamp(data.expires_at);
    return { ok: true, value: undefined };
  }
}
