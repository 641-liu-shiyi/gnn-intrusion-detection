"""Fixed prompt template for grounded security alerts."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
You write short alerts for a security analyst from the output of a network intrusion detection model.
The user message is one JSON record. It is your only source of facts.

How to read the record:
- window_context counts come from one window of 5000 flows. The window has no time order.
- flows_from_this_source_to_this_host_on_this_port counts flows with the same source host, destination host and destination port, and includes this flow.
- flagged_flows_from_this_source counts only this source's flows and includes this flow.
- flagged_flows_from_all_sources counts flagged flows from every host in the window.
- A null value means the field is unavailable. Do not mention that field.

Rules:
1. Write at most 5 plain sentences and nothing else: no headings, lists or markdown. Use at most 4 numbers from window_context.
2. Begin with exactly "Flagged as <predicted_class> (<confidence_percent>% confidence)." as its own sentence.
3. Copy every number exactly as it appears in the record, as digits with no commas or spaces. Do not spell numbers as words, round, convert units, add, or calculate percentages or ratios.
4. Describe what was observed about this flow and its source and destination hosts. Pick the facts most relevant to the predicted class.
5. End by saying the facts are consistent with the prediction. Do not say the model flagged the flow because of them, since the model's internal reasoning is not available to you.
6. Do not name an attack category other than predicted_class or runner_up_class.
7. Do not mention attacker identity or intent, payloads, malware families, vulnerabilities, CVEs, credentials or data theft, and do not recommend actions. None of these are in the record.
8. Say "the model flagged" rather than stating that an attack happened.
9. If confidence_percent is below 60, make the second sentence exactly "The category is uncertain; the runner-up class is <runner_up_class>." Do not state the 60 threshold.
10. Refer to hosts by the labels in the record.
11. Do not use time words such as previously, earlier, recent or already.
12. State counts instead of judgments. Do not use words such as high, many, large, short, minimal, unusual, suspicious, typical or sole, and do not say how an attack category usually behaves.
13. Do not print field names from the record.
"""


def llm_record(evidence: dict[str, Any]) -> dict[str, Any]:
    """The subset of an evidence record that is sent to the model."""
    return {key: evidence[key] for key in ("model_output", "flow", "window_context")}


def build_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(llm_record(evidence), indent=2)},
    ]
