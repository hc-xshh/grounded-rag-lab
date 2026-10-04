# API rate limits and errors

## Rate limits

Rate limits are applied per workspace and reset every 60 seconds.

| Plan | Sustained limit | Burst allowance |
| --- | --- | --- |
| Starter | 60 requests per minute | 2x for 10 seconds |
| Growth | 300 requests per minute | 2x for 10 seconds |
| Scale | 1,200 requests per minute | 2x for 10 seconds |

Requests above the sustained limit return HTTP 429 with a `Retry-After` header
that states the number of seconds to wait. Clients should read that header
instead of applying a fixed delay.

## Retry policy

Retries are safe for GET, PUT and DELETE. For POST requests, send an
`Idempotency-Key` header; the key is remembered for 24 hours and a repeated
request with the same key returns the original response instead of creating a
duplicate record.

Use exponential backoff with a maximum of 3 attempts for 429 and 5xx responses.
Requests that fail with 4xx responses other than 429 should not be retried
because the request itself is invalid.

## Error codes

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | invalid_request | The request body could not be parsed |
| 401 | unauthorized | The API key is missing or revoked |
| 403 | forbidden | The key is valid but lacks the required scope |
| 404 | not_found | The referenced object does not exist |
| 409 | conflict | A unique constraint was violated, for example a duplicate invoice number |
| 422 | validation_failed | The payload parsed but failed validation |
| 429 | rate_limited | The workspace exceeded its rate limit |
| 500 | internal_error | An unexpected server error; safe to retry |
| 503 | service_unavailable | Planned maintenance or overload; safe to retry |

## Pagination

List endpoints return at most 100 records per page and expose a `next_cursor`
value. Cursors are stable for 15 minutes; re-request the first page to restart
an expired cursor.
