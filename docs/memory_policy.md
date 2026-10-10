# SkyOps memory policy (one page)

Owner: the SkyOps platform team. Enforced by `skyops/context/memory.py`; every rule the code enforces has a
test in `fitness/s08` or `grading/s08` (the transparency, backup and copies rules are process rules, with no
code in this tag).
Reviewed before every release, and whenever a new kind of memory is proposed.

## Scope

Long-term memory is what the assistant may carry from one conversation to the next. It is not the
transcript (a checkpoint, kept with the run), not business records (messages sent, vouchers, audit: the
operations database), and not today's flight data (always fetched fresh through tools).

## What is stored, from whom, for how long

| Kind | Example | Allowed sources | Retention |
|---|---|---|---|
| `preference` | "Prefers answers as three short bullet points." | `user_stated`: the duty manager said it about their own work | 180 days from writing |
| `shift_handover` | "ZZ247 hotel vouchers still open at handover." | `agent_summary`: written by SkyOps code from verified facts | 14 days |

Anything else is refused, including any kind not in this table (`kind_not_allowed`). A new kind needs a
change to this page, a retention period, and a test.

## Never stored

1. Secrets: passwords, tokens, keys, in any form.
2. Passengers' personal data: names, contact details, booking references, payment data, travel
   documents, health details. Memory belongs to duty managers; passenger data lives in the booking system,
   under its own controls.
3. Instructions, and any text from an untrusted source: tool results, booking remarks, documents, email,
   web pages (`untrusted_source`). Memory is re-injected into every future conversation, so storing such
   text would turn one poisoned record into a standing instruction (OWASP ASI06, Memory and Context
   Poisoning).

The kind and source rules are the primary controls. The pattern screen (`screen()`) is a second line: it
catches secrets, emails, phone numbers, Luhn-valid card numbers, travel documents, booking references,
titled names ("Mrs Okafor", "passenger Ama Okafor"), a few health words and instruction-like phrases. It
misses most names, health details in plain words and anything paraphrased or in another language. A
refused write is logged by reason only, never with its text.

The source is a label set by the code that calls `write()`; the store can't verify it. Any code path that
read outside text (a tool result, a document, an email) must label what it writes with that source, and is
then refused. No path that touched outside text may call itself `agent_summary`. Code review checks every
caller of `write()`.

## Per whom

Every memory belongs to one user id. Every read and delete is scoped to that id, compared exactly. One user
never sees another's memories; there are no shared or team memories in this version.

## How memories are used

Recalled memories enter a request inside a `<memory>` block, labelled as data, with kind, source and date.
The pinned rules (rule 3) tell the model that memories are data, not instructions.

## Transparency and access

- Duty managers are told, in the assistant, when a memory is saved, and can list and delete their own
  memories at any time (`recall()` and `forget()` behind a settings page).
- Legal basis: legitimate interest in a useful assistant, recorded in the processing register with a
  balancing test. Consent is not used: in employment it is rarely freely given.
- The privacy notice for staff describes the two kinds, their sources and their retention periods.

## Forgetting

- Expired memories are never recalled, and `purge_expired()` deletes them daily.
- A deletion request runs `forget(user)`: every memory and every refusal record of that user is deleted in
  one committed transaction, and a receipt is kept with a keyed hash (HMAC-SHA-256, key from
  `SKYOPS_MEMORY_KEY`, held in the secret store) of the user id, the dates, and how many memories and refusal
  records went: proof it was honoured, without the data. Under GDPR Article 12(3), controllers act without
  undue delay and at the latest within one month of receipt, extendable by two further months for complex
  or numerous requests; in SkyOps, the same day. India's DPDP Act also gives a right to erasure; under the
  DPDP Rules notified on 13 November 2025, those obligations apply from 13 May 2027.
- Copies: the provider's request logs and our own traces hold rendered prompts, which contain recalled
  memories. Their retention is set no longer than the memories' own, and a deletion request is checked
  against them.
- Backups expire on their own schedule; restoring one re-applies outstanding deletion receipts first.

## Accepted for now

- Pattern screening is English-only and incomplete, with some false positives (any six-character mix of
  capitals and digits reads as a booking reference).
- Deletion receipts are kept for 2 years, then deleted.
- Not legal advice: confirm retention periods and erasure duties with your data protection officer.
