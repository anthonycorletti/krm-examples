from prometheus_client import Counter, Histogram

REQUESTS = Counter("platform_http_requests_total", "API requests", ["method", "route", "status"])
LATENCY = Histogram("platform_http_request_duration_seconds", "API request duration", ["route"])
