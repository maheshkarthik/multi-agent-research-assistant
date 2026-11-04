# Evaluation

Six fixed questions, run end to end from the command line with the default settings: 3 sub-questions, reviewer on, `qwen3:4b` on an Apple M2 with 8 GB RAM. The full reports are in [`examples/`](examples/). Local answers were checked against the sample documents' text; web answers were checked against the cited pages and general knowledge.

## Results

| # | Question | Sources | Time | Calls | Cited | Correct | Complete | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | What are the health effects of intermittent fasting? | Web | 237 s | 9 | ✅ | ✅ | ✅ | Good |
| 2 | How well do heat pumps work in cold climates? | Web | 165 s | 9 | ✅ | ✅ | ✅ | Good |
| 3 | Why did the Roman Republic fall? | Web | 170 s | 9 | ✅ | ✅ | ⚠️ | Good, but shallow |
| 4 | How does Quillmark Analytics handle passwords, and why? | Local PDFs | 115 s | 7 | ✅ | ⚠️ | ⚠️ | Partial |
| 5 | What caused Quillmark's incident INC-2026-0417, and what was changed afterwards? | Local PDFs | 98 s | 7 | ✅ | ❌ | ❌ | Half right |
| 6 | How did the printing press change European science? | Both | 204 s | 9 | ⚠️ | ✅ | ⚠️ | Partial |

**Summary:** 3 good, 2 partial, 1 half right. Every report cited its sources, and every run finished, in 98 to 237 seconds each. The web questions did best. The weak spots are local retrieval that misses a page, and the writer filling a gap instead of saying it couldn't find the answer.

## Notes per question

1. **Intermittent fasting.** A balanced answer: benefits, risks, the American Heart Association's view that the heart-health evidence is weak, and the National Institute on Aging's caution. Sources include Wikipedia, PubMed and the Mayo Clinic. One claim about 72-hour fasts comes from a low-quality blog and is stated as fact.
2. **Heat pumps.** Specific and well cited: performance below 40°F, the ENERGY STAR cold-climate standard (70% capacity at 5°F), HSPF ranges, and ground-source vs air-source. The summary gives two different minimum temperatures (−22°F and −13°F) without saying they come from different sources.
3. **Roman Republic.** Correct causes and dates: the Gracchi, Marius's army reforms, the Social War, Sulla, the crossing of the Rubicon on 10 January 49 BCE, and Octavian in 27 BCE. Accurate, but it reads like a list of events rather than an explanation of why they led to collapse.
4. **Passwords.** Correct on the current policy: 14 characters, no scheduled expiry, NIST guidance, 1Password. But:
   - it says phishing-resistant MFA applies to everyone, while the policy limits it to administrators and engineers with production access
   - it repeats the same facts in three sections
   - it misses the conflict with the employee handbook, which still says 10 characters and change every 90 days; the conflict check flagged an MFA difference that isn't a real conflict, and missed this one

   The conflict check found the handbook conflict in [earlier testing](../DESIGN.md), so it is unreliable rather than broken.
5. **Incident.** The root cause is correct and detailed: a non-concurrent `REINDEX` held locks and exhausted a 400-connection pool in nine minutes. The changes are wrong:
   - **The miss:** the corrective actions are a table on page 4 headed "Corrective Actions". Every search, including the reviewer's follow-ups, used words like "changes" and "remediation", so that page was never retrieved. Searching for "corrective actions" finds it at once.
   - **The writer's response:** instead of saying the changes weren't found, it named a monitoring system that isn't in the postmortem. It also presented the security policy's 15-minute SEV-1 rule as a change made after the incident.
6. **Printing press and science (web + local PDF).** Good on how printing spread knowledge: identical copies of diagrams and tables, 110 presses by 1480, Venice, the Aldine Press. It mixes the local history PDF with web pages as intended. But the section on "key scientific works" names no scientific works; Vesalius and Copernicus appear in the sources but not in the report. Several sentences in that section have no citation.

## What the evaluation changed

Running these six questions found five bugs that unit tests with a fake model could not:

| Found | Cause | Fix |
|---|---|---|
| The writer timed out on 4 of 6 questions | The writer asked for a larger context window than the other agents, so Ollama reloaded the model before every report | One context size for all calls |
| Reports were cut off mid-sentence | qwen3 wrote 2,000+ tokens of planning before the report, even with thinking off | The writer returns JSON sections, and code assembles the Markdown. Writing went from about 120 s to about 15 s |
| The planner fell back on 2 of 6 questions | qwen3 sometimes filled every JSON field with an empty string | `minLength` on every string in every schema |
| One report was the writer's instructions copied back | The rules read like field values ("summary: 2 to 3 sentences") | Reworded the rules, and a report that cites no source falls back to the cited findings |
| The conflict check called rewordings conflicts | e.g. "10.0 to 13.0" versus "10-13" | Drop pairs that state the same numbers or share a source; stricter prompt |

Typical run time dropped from 200 to 470 s to 100 to 240 s.

## Known limitations

- **Retrieval misses differently worded pages** (question 5). A fix tried during evaluation, giving the reviewer the earlier search words and asking for synonyms, made the follow-ups worse in its one trial and was not kept.
- **The writer sometimes fills a gap** instead of saying the sources don't cover it.
- **The conflict check is best effort**: it catches some disagreements between sources and reports some false ones.
- **The planner sometimes garbles names** in sub-questions (for example "Quill:mark"). The search queries are usually unaffected.
