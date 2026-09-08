from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.settings import Settings


def configure(settings: Settings, service: str):
    if not settings.otel_endpoint:
        return None
    from opentelemetry.sdk.resources import Resource

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/traces")
        )
    )
    trace.set_tracer_provider(provider)
    return provider
