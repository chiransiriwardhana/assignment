# Task 2A - Use Case Definition

## Domain-Specific Use Case: Financial Compliance Policy Assistant

Generic chatbot / creative-writing use cases are explicitly excluded by the spec
(capped at 5/30 regardless of execution). This use case is domain-specific,
non-trivial, and has clearly checkable correctness criteria — it fine-tunes a small
open model to act as a **first-line compliance policy assistant** for a financial
services firm, answering employee questions strictly from a provided internal policy
excerpt and flagging when a question must be escalated to human Compliance.

This directly complements Task 1 (equity research): a real trading/research desk needs
both market analysis *and* a way to quickly check "am I allowed to do this" against
internal policy without waiting on the compliance team for routine questions.

### Input

A single turn containing two things:
1. `policy_excerpt` — a short (100–300 word) excerpt from an internal compliance
   policy document (AML/KYC, insider trading & personal account dealing, gifts &
   entertainment, conflicts of interest, market abuse / information barriers,
   whistleblowing, data privacy).
2. `employee_question` — a natural-language question an employee might ask about
   that policy area.

Example input (as the user turn the model sees):
```
Policy excerpt (Personal Account Dealing):
"Employees must pre-clear any personal trade in a security that is on the Firm's
Restricted List through the Compliance pre-clearance portal at least 24 hours before
placing the order. Pre-clearance is valid for 48 hours. Trades below USD 1,000 in
aggregate per security per calendar month are exempt from pre-clearance but must
still be reported within 5 business days."

Employee question: "I want to buy $500 of a stock that's on the restricted list,
do I need to pre-clear it first?"
```

### Output

A single structured JSON object (fine-tuned as the assistant turn):
```json
{
  "answer": "No pre-clearance is required for this trade because it falls below the "
            "USD 1,000 per-security, per-month exemption threshold. You must still "
            "report the trade within 5 business days.",
  "cited_section": "Personal Account Dealing - pre-clearance exemption threshold",
  "requires_escalation": false,
  "confidence": "high"
}
```

### What Constitutes a Correct Response

A response is **correct** if it satisfies all of the following:
1. **Grounded**: the answer only uses facts present in the provided `policy_excerpt`
   — it never invents thresholds, dates, or procedures not stated in the excerpt.
2. **Cited**: `cited_section` names the specific part of the excerpt the answer relies
   on (not just "the policy" generically).
3. **Correctly escalated**: `requires_escalation` is `true` whenever the question
   involves (a) a scenario the excerpt does not clearly cover, (b) actual or suspected
   insider information, (c) amounts/thresholds at or above any stated limit, or (d) any
   request to circumvent a control — and `false` otherwise.
4. **Well-calibrated confidence**: `confidence` is `"low"` whenever the excerpt is
   ambiguous relative to the question, rather than always defaulting to `"high"`.

A response is **partially correct** if the answer and citation are right but the
escalation flag or confidence is wrong (or vice versa).

A response is **hallucinated / incorrect** if it states a threshold, deadline, or rule
not present in the excerpt, or answers confidently on a scenario the excerpt does not
cover.

### Why This Use Case Is Non-Trivial

- The correct answer depends on numeric threshold comparisons stated in the excerpt
  (e.g. "$500 < $1,000 exemption"), which is exactly where small/un-tuned models tend
  to hallucinate or ignore the numeric constraint.
- The escalation flag requires the model to recognize when a question falls *outside*
  what the excerpt covers — a grounding/faithfulness behavior, not just a fluency task.
- Structured JSON output with a fixed schema is required for real downstream use
  (routing to a human queue when `requires_escalation` is true), so output format
  correctness is itself part of "correct."
