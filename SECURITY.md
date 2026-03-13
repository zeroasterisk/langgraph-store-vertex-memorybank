# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.3.x | ✅ |
| < 0.3 | ❌ |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue.
2. Email [alan@zeroasterisk.com](mailto:alan@zeroasterisk.com) with details.
3. Include steps to reproduce if possible.

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## Security Considerations

- This package uses `google-cloud-aiplatform` for all API communication. Authentication is handled via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
- No credentials are stored by this package.
- Memory scoping (via namespaces) provides isolation between users, but scope enforcement is handled server-side by Memory Bank.
