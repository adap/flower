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
