# Security practices

Secrets are never committed to the repository. They are stored in a managed secret store and
injected at runtime. Dependencies are scanned weekly and critical vulnerabilities must be patched
within seven days. All services require authentication, and access tokens expire after one hour.