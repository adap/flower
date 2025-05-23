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
// ==============================================================================

package ai.flower.intelligence

import io.ktor.client.HttpClient
import java.security.*
import java.security.spec.ECGenParameterSpec
import java.security.spec.X509EncodedKeySpec
import java.util.*
import javax.crypto.Cipher
import javax.crypto.KeyAgreement
import javax.crypto.Mac
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.runBlocking
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant

class CryptographyHandler(
  private val baseURL: String = Constants.BASE_URL,
  private val apiKey: String,
  private val client: HttpClient,
) {
  private val keyPair: KeyPair
  private var sharedSecret: SecretKey
  private var encryptionId: String
  private var encryptionIdExpiresAt: Instant
  private var serverPublicKeyExpiresAt: Instant

  init {
    keyPair = generateECKeyPair()
    val submitResponse = runBlocking { submitClientPublicKey(keyPair.public) }
    encryptionId = submitResponse.encryptionId
    encryptionIdExpiresAt = submitResponse.expiresAt
    val serverResponse = runBlocking { getServerPublicKey() }
    serverPublicKeyExpiresAt = serverResponse.expiresAt
    sharedSecret =
      deriveSharedSecret(
        keyPair.private,
        Base64.getDecoder().decode(serverResponse.publicKeyEncoded),
      )
  }

  private fun generateECKeyPair(): KeyPair {
    val keyGen = KeyPairGenerator.getInstance("EC")
    keyGen.initialize(ECGenParameterSpec("secp384r1"))
    return keyGen.generateKeyPair()
  }

  private suspend fun submitClientPublicKey(publicKey: PublicKey): SubmitClientPublicKeyResponse {
    val publicKeyEncoded = Base64.getEncoder().encodeToString(publicKey.encoded)
    val payload = mapOf("public_key_base64" to publicKeyEncoded)
    return NetworkService.postElement(
      client = client,
      element = payload,
      authorization = "Bearer $apiKey",
      url = "$baseURL${Constants.ENCRYPTION_PUBLIC_KEY_PATH}",
    )
  }

  private suspend fun getServerPublicKey(): GetServerPublicKeyResponse {
    return NetworkService.getElement(
      client = client,
      url = "$baseURL${Constants.ENCRYPTION_SERVER_PUBLIC_KEY_PATH}",
      authorization = "Bearer $apiKey",
    )
  }

  private fun deriveSharedSecret(privateKey: PrivateKey, publicKeyBytes: ByteArray): SecretKey {
    val keyFactory = KeyFactory.getInstance("EC")
    val pubKeySpec = X509EncodedKeySpec(publicKeyBytes)
    val publicKey = keyFactory.generatePublic(pubKeySpec)

    val keyAgreement = KeyAgreement.getInstance("ECDH")
    keyAgreement.init(privateKey)
    keyAgreement.doPhase(publicKey, true)
    val secret = keyAgreement.generateSecret()
    val derivedKey = Hkdf.deriveKey(ikm = secret, salt = ByteArray(32), outputLength = 32)
    return SecretKeySpec(derivedKey, "AES")
  }

  suspend fun encryptMessage(message: String): String {
    refreshSharedSecretIfNeeded()
    val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    val iv = ByteArray(12).also { SecureRandom().nextBytes(it) }
    val gcmSpec = GCMParameterSpec(128, iv)
    cipher.init(Cipher.ENCRYPT_MODE, sharedSecret, gcmSpec)
    val ciphertext = cipher.doFinal(message.toByteArray(Charsets.UTF_8))
    val combined = iv + ciphertext
    return Base64.getEncoder().encodeToString(combined)
  }

  suspend fun decryptMessage(encryptedMessage: String): String {
    refreshSharedSecretIfNeeded()
    val decoded = Base64.getDecoder().decode(encryptedMessage)
    val iv = decoded.copyOfRange(0, 12)
    val cipherText = decoded.copyOfRange(12, decoded.size)
    val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    val gcmSpec = GCMParameterSpec(128, iv)
    cipher.init(Cipher.DECRYPT_MODE, sharedSecret, gcmSpec)
    val plainBytes = cipher.doFinal(cipherText)
    return String(plainBytes, Charsets.UTF_8)
  }

  private suspend fun refreshSharedSecretIfNeeded() {
    if (serverPublicKeyExpiresAt < Clock.System.now()) {
      val serverResponse = getServerPublicKey()
      serverPublicKeyExpiresAt = serverResponse.expiresAt
      sharedSecret =
        deriveSharedSecret(
          keyPair.private,
          Base64.getDecoder().decode(serverResponse.publicKeyEncoded),
        )
    }
  }

  suspend fun refreshEncryptionIdIfNeeded() {
    if (encryptionIdExpiresAt < Clock.System.now()) {
      val submitResponse = submitClientPublicKey(keyPair.public)
      encryptionId = submitResponse.encryptionId
      encryptionIdExpiresAt = submitResponse.expiresAt
    }
  }
}

object Hkdf {
  private const val HMAC_ALGORITHM = "HmacSHA256"
  private const val HASH_LEN = 32

  fun deriveKey(
    ikm: ByteArray,
    salt: ByteArray = ByteArray(HASH_LEN),
    info: ByteArray = ByteArray(0),
    outputLength: Int = 32,
  ): ByteArray {
    // Extract
    val prk = hmacSha256(salt, ikm)

    // Expand
    val n = (outputLength + HASH_LEN - 1) / HASH_LEN
    val result = ByteArray(outputLength)
    var t = ByteArray(0)
    var offset = 0

    for (i in 1..n) {
      val macInput = t + info + byteArrayOf(i.toByte())
      t = hmacSha256(prk, macInput)
      val toCopy = minOf(HASH_LEN, outputLength - offset)
      System.arraycopy(t, 0, result, offset, toCopy)
      offset += toCopy
    }

    return result
  }

  private fun hmacSha256(key: ByteArray, data: ByteArray): ByteArray {
    val mac = Mac.getInstance(HMAC_ALGORITHM)
    mac.init(SecretKeySpec(key, HMAC_ALGORITHM))
    return mac.doFinal(data)
  }
}
