# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in reagent-flow, please report it responsibly.

Please open a [GitHub issue](https://github.com/reagent-flow/reagent-flow/issues) with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

We aim to triage within 48 hours and release a fix promptly for critical issues.

## Security Considerations

### Trace Data Contains Sensitive Information

Traces are saved as plain JSON files containing the full tool call arguments and results from your agent runs. **This may include sensitive data** such as:

- API keys and tokens passed as tool arguments
- User PII (names, emails, addresses) processed by tools
- Database query results
- Internal system details and file paths

### Recommendations

1. **Add `.reagent/` to your `.gitignore`** to prevent accidental commits of trace data. Golden baselines may be an exception if they contain only synthetic/test data.

2. **Review trace files before sharing** them with others or uploading to bug trackers.

3. **Sanitize tool inputs/outputs before logging** if your agent handles real user data in production environments.

4. **Use synthetic data in tests and golden baselines** rather than copies of real user data.

### Path Traversal Prevention

Trace names are sanitized before use as filesystem paths. The `_sanitize_name()` function strips path separators, special characters, and leading dots/underscores. Names that sanitize to empty raise `ReagentError`.

### Serialization Safety

Non-JSON-serializable values in trace data are converted to strings with a `ReagentWarning` rather than silently coerced. This makes it visible when unexpected data types are being captured.

### No Network Access

The core library makes no network calls. Traces are stored locally as JSON files. Adapters only wrap existing framework client calls -- they do not make additional network requests.

## Future Work

A built-in redaction framework is planned for a future release to support automatic scrubbing of sensitive fields before traces are persisted.
