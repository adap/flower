const AES_KEY_LENGTH = 32;
const SIGN_ALG = 'HMAC';

export async function hkdf(
  hash: string,
  ikm: ArrayBuffer, // Input Keying Material (shared secret)
  info: Uint8Array, // Contextual information (e.g., 'ecdh key exchange')
  length: number
): Promise<ArrayBuffer> {
  const salt = new Uint8Array(AES_KEY_LENGTH); // All-zero salt
  const saltKey = await crypto.subtle.importKey('raw', salt, { name: SIGN_ALG, hash }, false, [
    'sign',
  ]);
  const prk = await crypto.subtle.sign(SIGN_ALG, saltKey, ikm);

  const prkKey = await crypto.subtle.importKey('raw', prk, { name: SIGN_ALG, hash }, false, [
    'sign',
  ]);
  const hashLength = AES_KEY_LENGTH;
  const numBlocks = Math.ceil(length / hashLength);

  let previousBlock = new Uint8Array(0);
  const output = new Uint8Array(length);
  let offset = 0;

  for (let i = 0; i < numBlocks; i++) {
    const input = new Uint8Array([...previousBlock, ...info, i + 1]);
    previousBlock = new Uint8Array(await crypto.subtle.sign(SIGN_ALG, prkKey, input));

    output.set(previousBlock.slice(0, Math.min(hashLength, length - offset)), offset);
    offset += hashLength;
  }

  return output.buffer;
}

/**
 * Convert date formatted as "2025-03-06T13:19:47.353034" to numerical timestamp
 * (in this example, 1741267187353)
 */
export function getTimestamp(dateString: string) {
  return new Date(dateString.slice(0, 23) + 'Z').valueOf();
}
