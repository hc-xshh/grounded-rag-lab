# Single sign-on and data retention

## Single sign-on

SAML 2.0 and OpenID Connect are available on the Growth and Scale plans. SCIM
user provisioning is available on the Scale plan only. Workspaces on the Starter
plan use email and password sign-in with optional two-factor authentication.

## Sessions

Sessions expire after 12 hours of inactivity by default. Administrators on the
Scale plan can lower that value to 8 hours. Revoking a user in the identity
provider ends their active sessions within 5 minutes.

## Identity provider requirements

The identity provider must sign assertions with RSA-SHA256 and publish its
metadata at an HTTPS URL. Assertions older than 5 minutes are rejected to reduce
replay risk.

## Data retention

| Data type | Retention |
| --- | --- |
| Audit logs | 24 months |
| Invoice and tax records | 7 years, as required by tax law |
| API request logs | 30 days |
| Deleted workspace data | Purged after a 30-day grace period |

Customers can export reports and attachments at any time. Exports are generated
as CSV or JSON and are available for 7 days after they are requested.

## Deletion requests

A deletion request can be raised by a workspace administrator from the data
settings page or by email. Requests are processed within 30 days. Backups are
rotated every 35 days, so a record may remain in cold backup until the next
rotation completes.

## Encryption

Data is encrypted at rest with AES-256 and in transit with TLS 1.2 or newer.
Encryption keys are rotated every 12 months.
