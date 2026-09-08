import runpy
import ssl
from pathlib import Path

from cryptography import x509

tooling = runpy.run_path(str(Path(__file__).resolve().parents[2] / "bin/lib/certs.py"))
HOSTS = tooling["HOSTS"]
ensure_certificates = tooling["ensure_certificates"]


def test_local_certificate_names_and_reuse(tmp_path):
    cert = ensure_certificates(tmp_path)
    original_ca = (tmp_path / "ca.crt").read_bytes()
    original_key = (tmp_path / "localhost.key").read_bytes()
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(HOSTS).issubset(names.get_values_for_type(x509.DNSName))
    ca = x509.load_pem_x509_certificate(original_ca)
    cert.verify_directly_issued_by(ca)
    assert (tmp_path / "ca.key").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "localhost.key").stat().st_mode & 0o777 == 0o600
    assert ensure_certificates(tmp_path).serial_number == cert.serial_number
    assert (tmp_path / "localhost.key").read_bytes() == original_key
    assert (tmp_path / "ca.crt").read_bytes() == original_ca
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "localhost.crt", tmp_path / "localhost.key")


def test_leaf_can_be_replaced_without_retrusting_ca(tmp_path):
    original = ensure_certificates(tmp_path)
    ca = (tmp_path / "ca.crt").read_bytes()
    (tmp_path / "localhost.key").unlink()
    replacement = ensure_certificates(tmp_path)
    assert replacement.serial_number != original.serial_number
    assert (tmp_path / "ca.crt").read_bytes() == ca
