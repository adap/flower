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

import Crypto
import Foundation

class CryptographyHandler {
  private let serverUrl: String
  private let apiKey: String
  private let privateKey: P384.KeyAgreement.PrivateKey
  private let publicKey: P384.KeyAgreement.PublicKey
  private var sharedSecret: SymmetricKey
  private var encryptionIdExpiresAt: Date
  private var serverPublicKeyExpiresAt: Date
  public var encryptionId: String

  public init(serverUrl: String, apiKey: String) async throws {
    self.serverUrl = serverUrl
    self.apiKey = apiKey
    self.privateKey = P384.KeyAgreement.PrivateKey()
    self.publicKey = privateKey.publicKey
    let submitClientPublicKeyResponse = try await CryptographyHandler.submitClientPublicKey(
      self.publicKey,
      serverURL: serverUrl,
      apiKey: apiKey
    )
    self.encryptionId = submitClientPublicKeyResponse.encryptionId
    self.encryptionIdExpiresAt = submitClientPublicKeyResponse.expiresAt
    let getServerPublicKeyResponse = try await CryptographyHandler.getServerPublicKey(
      serverURL: serverUrl,
      apiKey: apiKey
    )
    self.serverPublicKeyExpiresAt = getServerPublicKeyResponse.expiresAt
    self.sharedSecret = try CryptographyHandler.deriveSharedSecret(
      privateKey: self.privateKey,
      publicKeyData: getServerPublicKeyResponse.publicKeyBase64
    )
  }

  private static func submitClientPublicKey(
    _ publicKey: P384.KeyAgreement.PublicKey,
    serverURL: String,
    apiKey: String
  ) async throws -> SubmitClientPublicKeyResponse {
    let publicKeyBase64 = publicKey.derRepresentation.base64EncodedString()
    let payload = ["public_key_base64": publicKeyBase64]

    guard let url = URL(string: "\(serverURL)/encryption/public-key") else {
      throw URLError(.badURL)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])

    let (data, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
      throw Failure(
        code: .encryptionError,
        message: "Failed to submit client public key"
      )
    }

    let parsedResponse = try parseJson(
      from: data,
      as: SubmitClientPublicKeyResponse.self
    )
    return parsedResponse
  }

  private static func getServerPublicKey(
    serverURL: String,
    apiKey: String
  ) async throws -> GetServerPublicKeyResponse {
    guard let url = URL(string: "\(serverURL)/encryption/server-public-key") else {
      throw Failure(
        code: .encryptionError,
        message: "Failed to fetch server public key URL"
      )
    }
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")

    let (data, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
      throw Failure(
        code: .encryptionError,
        message: "Failed to fetch server public key"
      )
    }
    let parsedResponse = try parseJson(
      from: data,
      as: GetServerPublicKeyResponse.self
    )
    return parsedResponse
  }

  private static func parseJson<T: Codable>(
    from data: Data,
    as type: T.Type
  ) throws -> T {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try decoder.decode(T.self, from: data)
  }

  private static func deriveSharedSecret(
    privateKey: P384.KeyAgreement.PrivateKey,
    publicKeyData: Data
  ) throws -> SymmetricKey {
    let publicKey = try P384.KeyAgreement.PublicKey(derRepresentation: publicKeyData)
    let sharedSecret = try privateKey.sharedSecretFromKeyAgreement(with: publicKey)

    let derivedKey = sharedSecret.hkdfDerivedSymmetricKey(
      using: hashAlgorithm,
      salt: Data(),
      sharedInfo: hkdfInfo,
      outputByteCount: aesKeyLength
    )

    return derivedKey
  }

  func encryptMessage(_ message: String) async throws -> String {
    try await refreshSharedSecretIfNeeded()
    let messageData = message.data(using: .utf8)!
    let nonce = AES.GCM.Nonce()
    let sealedBox = try AES.GCM.seal(messageData, using: sharedSecret, nonce: nonce)

    guard let encryptedData = sealedBox.combined else {
      throw Failure(
        code: .encryptionError,
        message: "Encryption failed"
      )
    }
    return encryptedData.base64EncodedString()
  }

  func decryptMessage(_ encryptedMessage: String) async throws -> String {
    try await refreshSharedSecretIfNeeded()
    guard let encryptedData = Data(base64Encoded: encryptedMessage) else {
      throw Failure(
        code: .encryptionError,
        message: "Invalid base64 string"
      )
    }

    let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)

    let decryptedData = try AES.GCM.open(sealedBox, using: sharedSecret)
    guard let decryptedMessage = String(data: decryptedData, encoding: .utf8) else {
      throw Failure(
        code: .encryptionError,
        message: "Decryption failed"
      )
    }

    return decryptedMessage
  }

  private func refreshSharedSecretIfNeeded() async throws {
    if self.serverPublicKeyExpiresAt < Date() {
      let getServerPublicKeyResponse = try await CryptographyHandler.getServerPublicKey(
        serverURL: serverUrl,
        apiKey: apiKey
      )
      self.serverPublicKeyExpiresAt = getServerPublicKeyResponse.expiresAt
      self.sharedSecret = try CryptographyHandler.deriveSharedSecret(
        privateKey: self.privateKey,
        publicKeyData: getServerPublicKeyResponse.publicKeyBase64
      )
    }
  }

  func refreshEncryptionIdIfNeeded() async throws {
    if self.encryptionIdExpiresAt < Date() {
      let submitClientPublicKeyResponse = try await CryptographyHandler.submitClientPublicKey(
        self.publicKey,
        serverURL: serverUrl,
        apiKey: apiKey
      )
      self.encryptionId = submitClientPublicKeyResponse.encryptionId
      self.encryptionIdExpiresAt = submitClientPublicKeyResponse.expiresAt
    }
  }
}

private struct SubmitClientPublicKeyResponse: Codable {
  let expiresAt: Date
  let encryptionId: String

  enum CodingKeys: String, CodingKey {
    case expiresAt = "expires_at"
    case encryptionId = "encryption_id"
  }
}

private struct GetServerPublicKeyResponse: Codable {
  let publicKeyBase64: Data
  let expiresAt: Date

  enum CodingKeys: String, CodingKey {
    case publicKeyBase64 = "public_key_base64"
    case expiresAt = "expires_at"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let base64String = try container.decode(String.self, forKey: .publicKeyBase64)

    guard let decodedData = Data(base64Encoded: base64String) else {
      throw DecodingError.dataCorruptedError(
        forKey: .publicKeyBase64, in: container, debugDescription: "Invalid Base64 string")
    }

    self.publicKeyBase64 = decodedData
    self.expiresAt = try container.decode(Date.self, forKey: .expiresAt)
  }
}
