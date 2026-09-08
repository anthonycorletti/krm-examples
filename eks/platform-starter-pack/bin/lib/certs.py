"""Development-only local CA and server certificates; invoked by bin/certs."""

import fcntl
import ipaddress
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

HOSTS = ("web.localhost", "api.localhost", "mcp.localhost", "localhost")


def write_file(path: Path, data: bytes, mode: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "wb") as file:
        file.write(data)
    temporary.replace(path)
    path.chmod(mode)


def write_key(path: Path, key: rsa.RSAPrivateKey) -> None:
    write_file(
        path,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        0o600,
    )


def ensure_certificates(directory: Path) -> x509.Certificate:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    # The three dev servers can start together; serialize CA/key creation and renewal.
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ensure_certificates(directory)


def _ensure_certificates(directory: Path) -> x509.Certificate:
    now = datetime.now(UTC)
    ca_key_path, ca_path = directory / "ca.key", directory / "ca.crt"
    if ca_key_path.exists() != ca_path.exists():
        raise RuntimeError("Incomplete local CA; restore the matching ca.key and ca.crt")
    if ca_path.exists():
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise RuntimeError("Unexpected local CA key type")
        ca_key = key
        if ca.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ) != ca_key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ):
            raise RuntimeError("Local CA certificate and private key do not match")
        if ca.not_valid_after_utc < now + timedelta(days=90):
            raise RuntimeError("Local CA needs deliberate renewal and reinstallation of trust")
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        name = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "Platform Starter Pack Local CA")]
        )
        ca = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        write_key(ca_key_path, ca_key)
        write_file(ca_path, ca.public_bytes(serialization.Encoding.PEM), 0o644)
    cert_path, key_path = directory / "localhost.crt", directory / "localhost.key"
    if cert_path.exists() and key_path.exists():
        certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
        try:
            certificate.verify_directly_issued_by(ca)
            existing_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
            matching_key = existing_key.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ) == certificate.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            names = certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            if (
                matching_key
                and set(HOSTS).issubset(names.get_values_for_type(x509.DNSName))
                and certificate.not_valid_after_utc > now + timedelta(days=14)
            ):
                return certificate
        except (ValueError, TypeError, InvalidSignature, x509.ExtensionNotFound):
            pass
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "web.localhost")]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    *[x509.DNSName(host) for host in HOSTS],
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address("::1")),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    write_key(key_path, key)
    write_file(cert_path, certificate.public_bytes(serialization.Encoding.PEM), 0o644)
    return certificate


if __name__ == "__main__":
    certificate = ensure_certificates(Path(__file__).resolve().parents[2] / ".local/tls")
    print(f"Local HTTPS certificate ready; expires {certificate.not_valid_after_utc.date()}")
