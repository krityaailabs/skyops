# Project 1 review checklist

- [ ] Every fact in the answer can be traced to a field of a simulator record.
- [ ] No write tool is reachable from this assistant.
- [ ] A budget stop produces a clear message, not a half answer, and `grounded=False`.
- [ ] Unknown and ambiguous flights are handled explicitly.
- [ ] The most recent lookup decides: a not-found status lookup followed by a Sydney departures lookup answers about Sydney.
- [ ] Free-text fields from records (delay reasons) are length-limited before they reach the user.
- [ ] Tool errors reach the model as data; no exception text is shown to the model or the user.
- [ ] The ten eval cases pass, and you know *why* c10 passes now.
- [ ] The code would still be correct if the model's prose were wrong in every case.
- [ ] Cost per 1,000 questions is known (from the trace), at the course's example prices.
