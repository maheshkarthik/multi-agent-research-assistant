# What caused Quillmark's incident INC-2026-0417, and what was changed afterwards?

Quillmark's incident INC-2026-0417 was caused by database connection pool exhaustion due to a scheduled database maintenance task that held locks longer than expected after a configuration change to reindex the forecasts table without the CONCURRENTLY option. The incident occurred from 03:14 to 08:06 Pacific Time on April 17, 2026, and the database connection pool of 400 connections was exhausted in nine minutes. After the incident, Quillmark implemented a new database connection pool monitoring system and added a 15-minute response window for SEV-1 incidents.

## Root cause of the incident

The incident INC-2026-0417 was caused by database connection pool exhaustion in the forecasting cluster [1]. This occurred because a scheduled database maintenance task held locks far longer than expected after a configuration change on April 9 to reindex the forecasts table without the CONCURRENTLY option [1][3]. The reindex operation took more than three hours instead of the 20 minutes observed in staging [3], leading to the exhaustion of the database connection pool of 400 connections within nine minutes [3]. The incident itself lasted from 03:14 to 08:06 Pacific Time on April 17, 2026 [1].

## Post-incident changes

Quillmark implemented a new database connection pool monitoring system after the INC-2026-0417 incident [1]. Additionally, Quillmark added a 15-minute response window for SEV-1 incidents following the incident [5]. These changes were specifically introduced to address the database connection pool exhaustion issue identified during the incident. The information security policy updates mentioned in source [4] were not directly tied to the incident response, as the sources indicate that the incident postmortem (source 1) does not explicitly describe changes made in response to the incident [4].

## Conclusion

Quillmark's incident INC-2026-0417 resulted from a database maintenance task that held locks longer than expected after a configuration change, leading to connection pool exhaustion. The immediate response included implementing database connection pool monitoring and a 15-minute response window for critical incidents. These changes were made to prevent similar issues in the future.

## Sources

[1] quillmark-incident-postmortem-inc-2026-0417.pdf, page 1 (local document)  
[3] quillmark-incident-postmortem-inc-2026-0417.pdf, page 3 (local document)  
[4] quillmark-information-security-policy.pdf, page 1 (local document)  
[5] quillmark-information-security-policy.pdf, page 3 (local document)  

_1 further source(s) were read but not cited._

_Generated locally in 98 s with 7 model calls. AI-generated: check important facts against the sources._
