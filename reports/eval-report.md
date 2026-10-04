# Evaluation report

Corpus: **5 documents / 29 chunks** · embedder: `tfidf-offline` · answerer: `extractive-offline` · top-k: **5**

## Headline metrics

| metric | value | what it means |
| --- | --- | --- |
| Retrieval recall@k | 1.0 | share of answerable questions whose source document was retrieved |
| MRR | 1.0 | mean reciprocal rank of the first correct document |
| Fact coverage | 0.973 | expected facts present in the answers |
| Citation precision | 0.8649 | citations that point at the expected document |
| Invalid citations | 0 | labels that did not exist in the context (guardrail target: 0) |
| Refusal accuracy | 1.0 | answered what is answerable AND refused what is not |
| Spurious refusals | 0 | answerable questions that were refused anyway (lower is better) |
| Answer support | 1.0 | share of answer sentences backed by their citation |
| Latency p50 | 0.76 | median end-to-end answer latency in ms |
| Latency p95 | 1.29 | 95th percentile latency in ms |

## Per-case results

| # | question | answerable | outcome | recall | support | citations |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | How much does the Growth plan cost per month and how many seats are included? | yes | answered | hit | 1.00 | [S1] |
| 2 | What is the overage price for extra API calls on the Scale plan? | yes | answered | hit | 1.00 | [S1] |
| 3 | How long is the free trial and does it require a credit card? | yes | answered | hit | 1.00 | [S1] |
| 4 | What discount does annual billing give compared with paying monthly? | yes | answered | hit | 1.00 | [S1] |
| 5 | When does a plan downgrade take effect? | yes | answered | hit | 1.00 | [S1] |
| 6 | Are added seats prorated or charged in full? | yes | answered | hit | 1.00 | [S1] [S3] [S4] |
| 7 | When are invoices issued and when are they due? | yes | answered | hit | 1.00 | [S1] |
| 8 | Which credit cards are accepted for payment? | yes | answered | hit | 1.00 | [S1] [S3] [S4] |
| 9 | What is the refund window for a monthly subscription? | yes | answered | hit | 1.00 | [S1] |
| 10 | How long is a paused workspace kept before the data is deleted? | yes | answered | hit | 1.00 | [S1] [S2] |
| 11 | In which countries is GST collected? | yes | answered | hit | 1.00 | [S1] |
| 12 | When does the system retry a failed payment charge? | yes | answered | hit | 1.00 | [S1] |
| 13 | What is the API rate limit on the Growth plan? | yes | answered | hit | 1.00 | [S1] |
| 14 | How should a client respond to an HTTP 429 rate limited error? | yes | answered | hit | 1.00 | [S1] |
| 15 | How long is an idempotency key remembered for POST requests? | yes | answered | hit | 1.00 | [S1] |
| 16 | How many retry attempts are recommended for 5xx responses? | yes | answered | hit | 1.00 | [S1] |
| 17 | How long is a pagination cursor valid? | yes | answered | hit | 1.00 | [S1] [S2] [S3] |
| 18 | Which plans support SAML single sign-on? | yes | answered | hit | 1.00 | [S1] |
| 19 | How long are audit logs retained? | yes | answered | hit | 1.00 | [S1] |
| 20 | How often are encryption keys rotated? | yes | answered | hit | 1.00 | [S1] |
| 21 | How long does a user session last before it expires? | yes | answered | hit | 1.00 | [S1] |
| 22 | How long does a data deletion request take to process? | yes | answered | hit | 1.00 | [S1] [S3] |
| 23 | What is the first response target for a P1 critical incident? | yes | answered | hit | 1.00 | [S1] [S2] [S4] |
| 24 | When is a P1 incident escalated to the on-call engineer? | yes | answered | hit | 1.00 | [S1] |
| 25 | How often is the status page updated during an incident? | yes | answered | hit | 1.00 | [S1] |
| 26 | What is the response target for a P3 request? | yes | answered | hit | 1.00 | [S1] |
| 27 | Which plan includes a named technical account manager? | yes | answered | hit | 1.00 | [S1] |
| 28 | How much service credit do we receive if monthly uptime falls below 99.9 percent? | no | refused (correct) | n/a | 0.00 | - |
| 29 | Is there a mobile app for iOS and Android? | no | refused (correct) | n/a | 0.00 | - |
| 30 | Can we pay in Japanese yen? | no | refused (correct) | n/a | 0.00 | - |
| 31 | How many employees work at the company? | no | refused (correct) | n/a | 0.00 | - |
| 32 | Which programming languages have official client SDKs? | no | refused (correct) | n/a | 0.00 | - |
| 33 | Can I download the SOC 2 Type II report from the trust center? | no | refused (correct) | n/a | 0.00 | - |
| 34 | What is the maximum size for a file attachment upload? | no | refused (correct) | n/a | 0.00 | - |

## Refusals

- `gap-uptime-credits` (expected) - How much service credit do we receive if monthly uptime falls below 99.9 percent? → `low_retrieval_support`
- `gap-mobile-app` (expected) - Is there a mobile app for iOS and Android? → `low_retrieval_support`
- `gap-currency-jpy` (expected) - Can we pay in Japanese yen? → `low_retrieval_support`
- `gap-headcount` (expected) - How many employees work at the company? → `low_retrieval_support`
- `gap-sdks` (expected) - Which programming languages have official client SDKs? → `low_retrieval_support`
- `gap-soc2` (expected) - Can I download the SOC 2 Type II report from the trust center? → `low_retrieval_support`
- `gap-attachment-size` (expected) - What is the maximum size for a file attachment upload? → `low_retrieval_support`

## Reproducing this report

```bash
make install
make eval          # writes reports/ + docs/index.html
```
