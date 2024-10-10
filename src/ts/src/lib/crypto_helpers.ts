import { ec as EC } from "elliptic";
import * as crypto from "crypto";

const ec = new EC("p256");

// Convert public key to bytes
export function publicKeyToBytes(key: EC.KeyPair): Buffer {
  return Buffer.from(key.getPublic("array"));
}

// Convert bytes back to a public key
export function bytesToPublicKey(bytes: Buffer): EC.KeyPair {
  return ec.keyFromPublic(bytes);
}

// Generate shared key between private and public keys
export function generateSharedKey(privateKey: EC.KeyPair, publicKey: EC.KeyPair): Buffer {
  return Buffer.from(privateKey.derive(publicKey.getPublic()).toArray());
}

// Compute HMAC using shared key and data
export function computeHMAC(key: Buffer, message: Buffer): Buffer {
  return crypto.createHmac("sha256", key).update(message).digest();
}
