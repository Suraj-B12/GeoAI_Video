package com.geoai.capture.io

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.Signature
import java.security.spec.ECGenParameterSpec

/**
 * Signs session manifests with a per-install EC P-256 key held in Android Keystore.
 * Signatures are DER-encoded ECDSA over SHA-256, which Python's `cryptography` verifies with
 * ec.ECDSA(hashes.SHA256()).
 */
object Signer {
    private const val ALIAS = "geoai_session_signing_v1"
    private const val PROVIDER = "AndroidKeyStore"

    data class SignatureResult(
        val signatureB64: String,
        val publicKeyX509B64: String,
        val certificateChainB64: List<String>,
    )

    private fun keyStore(): KeyStore = KeyStore.getInstance(PROVIDER).apply { load(null) }

    private fun ensureKey(ks: KeyStore) {
        if (ks.containsAlias(ALIAS)) return
        fun generate(withAttestation: Boolean) {
            val builder = KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY)
                .setAlgorithmParameterSpec(ECGenParameterSpec("secp256r1"))
                .setDigests(KeyProperties.DIGEST_SHA256)
            if (withAttestation) builder.setAttestationChallenge("geoai-capture-v1".toByteArray())
            KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_EC, PROVIDER).apply {
                initialize(builder.build())
                generateKeyPair()
            }
        }
        try {
            generate(withAttestation = true)
        } catch (e: Exception) {
            generate(withAttestation = false)
        }
    }

    fun sign(data: ByteArray): SignatureResult {
        val ks = keyStore()
        ensureKey(ks)
        val entry = ks.getEntry(ALIAS, null) as KeyStore.PrivateKeyEntry
        val sig = Signature.getInstance("SHA256withECDSA").run {
            initSign(entry.privateKey)
            update(data)
            sign()
        }
        val chain = ks.getCertificateChain(ALIAS)?.map { b64(it.encoded) } ?: emptyList()
        return SignatureResult(b64(sig), b64(entry.certificate.publicKey.encoded), chain)
    }

    private fun b64(bytes: ByteArray): String = Base64.encodeToString(bytes, Base64.NO_WRAP)
}
