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

import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.security.*
import java.security.spec.*
import java.time.Instant
import java.util.*
import javax.crypto.*
import javax.crypto.interfaces.DHPublicKey
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

class CryptographyHandler(
  private val serverUrl: String,
  private val apiKey: String
) {
  private val keyPair: KeyPair
  private var sharedSecret: SecretKey
  private var encryptionId: String
  private var encryptionIdExpiresAt: Instant
  private var serverPublicKeyExpiresAt: Instant

  init {
    keyPair = generateECKeyPair()
    val submitResponse = runBlocking {
      submitClientPublicKey(keyPair.public)
    }
    encryptionId = submitResponse.encryptionId
    encryptionIdExpiresAt = submitResponse.expiresAt
    val serverResponse = runBlocking {
      getServerPublicKey()
    }
    serverPublicKeyExpiresAt = serverResponse.expiresAt
    sharedSecret = deriveSharedSecret(keyPair.private, serverResponse.publicKeyEncoded)
  }

  private fun generateECKeyPair(): KeyPair {
    val keyGen = KeyPairGenerator.getInstance("EC")
    keyGen.initialize(ECGenParameterSpec("secp384r1"))
    return keyGen.generateKeyPair()
  }

  private suspend fun submitClientPublicKey(publicKey: PublicKey): SubmitClientPublicKeyResponse =
    withContext(Dispatchers.IO) {
      val publicKeyEncoded = Base64.getEncoder().encodeToString(publicKey.encoded)
      val payload = JSONObject().put("public_key_base64", publicKeyEncoded)
      val request = HttpRequest.newBuilder()
        .uri(URI.create("$serverUrl/encryption/public-key"))
        .header("Authorization", "Bearer $apiKey")
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(payload.toString()))
        .build()
      val response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString())
      if (response.statusCode() != 200) {
        throw Exception("Failed to submit client public key")
      }
      val json = JSONObject(response.body())
      SubmitClientPublicKeyResponse(
        Instant.parse(json.getString("expires_at")),
        json.getString("encryption_id")
      )
    }

  private suspend fun getServerPublicKey(): GetServerPublicKeyResponse =
    withContext(Dispatchers.IO) {
      val request = HttpRequest.newBuilder()
        .uri(URI.create("$serverUrl/encryption/server-public-key"))
        .header("Authorization", "Bearer $apiKey")
        .header("Content-Type", "application/json")
        .GET()
        .build()
      val response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString())
      if (response.statusCode() != 200) {
        throw Exception("Failed to get server public key")
      }
      val json = JSONObject(response.body())
      val pubKeyBase64 = json.getString("public_key_base64")
      val pubKeyBytes = Base64.getDecoder().decode(pubKeyBase64)
      GetServerPublicKeyResponse(pubKeyBytes, Instant.parse(json.getString("expires_at")))
    }

  private fun deriveSharedSecret(privateKey: PrivateKey, publicKeyBytes: ByteArray): SecretKey {
    val keyFactory = KeyFactory.getInstance("EC")
    val pubKeySpec = X509EncodedKeySpec(publicKeyBytes)
    val publicKey = keyFactory.generatePublic(pubKeySpec)

    val keyAgreement = KeyAgreement.getInstance("ECDH")
    keyAgreement.init(privateKey)
    keyAgreement.doPhase(publicKey, true)
    val secret = keyAgreement.generateSecret()

    // Derive a 256-bit AES key from the shared secret
    return SecretKeySpec(secret.copyOf(32), "AES")
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
    if (serverPublicKeyExpiresAt.isBefore(Instant.now())) {
      val serverResponse = getServerPublicKey()
      serverPublicKeyExpiresAt = serverResponse.expiresAt
      sharedSecret = deriveSharedSecret(keyPair.private, serverResponse.publicKeyEncoded)
    }
  }

  suspend fun refreshEncryptionIdIfNeeded() {
    if (encryptionIdExpiresAt.isBefore(Instant.now())) {
      val submitResponse = submitClientPublicKey(keyPair.public)
      encryptionId = submitResponse.encryptionId
      encryptionIdExpiresAt = submitResponse.expiresAt
    }
  }
}
