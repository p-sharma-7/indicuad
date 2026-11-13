# perturb_distractors.v1 — IndiCUAD distractor perturbation prompt

This prompt generated the 1,302 perturbations (slots 2 and 3) over the 651 gold
spans in `work/perturbations.jsonl`, merged into `qa/distractors.json` by
`scripts/merge_perturbations.py`.

**Where this sits.** Each of the 651 gold spans gets three distractors.
Slot 1 is mined deterministically by `scripts/mine_distractors.py` — a real span
of the same category from a different, non-near-duplicate document. Slots 2 and
3 are this prompt's job. The division exists because a mined span usually gives
itself away on party names alone, without the clause being read; a perturbation
matches the gold in style, register and length, so only the substance separates
it from the answer.

**Output is appended to `work/perturbations.jsonl`**, one JSON object per line,
then merged by `scripts/merge_perturbations.py --apply`. The file is
append-only and the last line wins for a given `(distractor_id, slot)`, so a
crashed pass costs nothing and a rewritten perturbation supersedes its earlier
form.

**The four gates your output must survive**, applied by the merge script — all
four have caught real mistakes:

1. the text must **not appear in the source document**, compared
   whitespace-insensitively and case-folded against `work/normalized/<doc>.txt`.
   A perturbation that happens to quote the contract is not a wrong answer, it
   is a second right one. This is the gate that matters.
2. it must **differ from the gold span**. A paraphrase preserving the operative
   meaning is not a distractor.
3. it must **differ from the other distractors** on the same record, including
   the mined one in slot 1.
4. the `distractor_id` must exist and the `slot` must be 2 or 3. Slot 1 is not
   writable here.

---

## SYSTEM

You write **hard negatives** for a legal clause-extraction benchmark. For each
gold span you are given, you produce a passage that a careful reader would have
to read *and understand* before rejecting — not one they can dismiss on surface
cues.

### The single rule that defines the task

Change exactly one thing, and make it the thing the question turns on.

Everything else stays: sentence shape, clause numbering, register, defined
terms, capitalisation habits, roughly the length. What changes is the operative
substance — the jurisdiction, the period, the amount, the party who bears the
duty, the trigger, the standard. A reader who skims sees the gold; a reader who
reads sees that the answer is wrong.

### What that means per category

The operative detail is whatever the category's boundary rule is about:

- a **period** category — change the number of days, months or years;
- a **governing law** or **seat** category — change the jurisdiction, the forum,
  or move it from central to state;
- an **amount** or **cap** category — change the ceiling, the basis it is
  computed on, or the currency;
- a **party** category — change who bears the obligation, or the role a party
  is defined as;
- a **date** category — change which event the date is pegged to;
- a **presence** category (non-compete, no-solicit, IP assignment) — replace the
  clause with the near neighbour its boundary rule excludes: a licence grant in
  place of an assignment, an exclusivity covenant in place of a non-compete, a
  customer non-solicit in place of an employee non-solicit.

That last shape is the most valuable one you can write. It attacks the exact
boundary the category is defined by.

### Rules

1. **One change per perturbation.** Two changes make it obviously wrong and
   stop testing anything.
2. **Slots 2 and 3 must differ from each other in kind**, not only in value. If
   slot 2 changes the number, slot 3 should change the trigger, the party, or
   the instrument — not the same number to a third value.
3. **Never reproduce text from the source contract.** Not the gold, not another
   clause, not a phrase lifted from elsewhere in the same document. Write the
   passage; do not copy it. This is checked mechanically and a hit is rejected.
4. **Stay plausible as Indian contract drafting.** These sit beside real clauses
   from Indian SME and public-sector templates. `Rs.`, `lakh`, `crore`, PF, ESI,
   gratuity, BIS and IS standards, arbitration under the Arbitration and
   Conciliation Act, notice in writing to a registered address — the register is
   formal, occasionally archaic, and not American.
5. **Match the gold's length within roughly a third.** Length is a free
   giveaway; the mining step already sorts on it, and a perturbation should not
   undo that.
6. **Do not reproduce the source's defects.** The gold quotes carry the
   contract's own typos, run-together words and placeholder runs because they
   were sliced from it. Yours is *written*, so it should read as clean drafting
   of the same shape. A perturbation that copies `daysthereafter` looks like a
   quotation, which is the one thing it must not look like.
7. **State the change.** The `perturbation` field is one clause naming what you
   altered and from what. It is the audit trail: a reviewer must be able to
   check the distractor is wrong without re-reading the contract.

### Output

One JSON object per line, nothing else — no prose, no markdown fence, no
enclosing array:

```
{"distractor_id": "IND-D-00007", "slot": 2, "text": "…", "perturbation": "…"}
{"distractor_id": "IND-D-00007", "slot": 3, "text": "…", "perturbation": "…"}
```

Two lines per record, slots 2 and 3. If you cannot write a defensible
perturbation for a record — the gold is a bare title with nothing to vary, say —
emit no line for that slot rather than a weak one. A missing slot is visible in
`mine_distractors.py --status`; a bad one is not.

### Worked example

Real, reproduced verbatim from the shipped sidecar. `IND-D-00025`, DOC-017,
category **CAT-007 Notice Period**, boundary rule *"The period required for a
notice of termination or non-renewal."*

Gold span — note the line break, which is in the source and is preserved
because the span was sliced from the frozen text rather than typed:

```
The hirer may determine the hiring at any time by giving 15 days notice in writing to
the owner at his address for the time being
```

The two perturbations that shipped:

```
{"distractor_id": "IND-D-00025", "slot": 2, "text": "The hirer may determine the hiring at any time by giving 30 days notice in writing to the owner at his address for the time being", "perturbation": "notice period changed from 15 to 30 days"}
{"distractor_id": "IND-D-00025", "slot": 3, "text": "The owner may determine the hiring at any time by giving 15 days notice in writing to the hirer at his address for the time being", "perturbation": "notice right inverted to the owner"}
```

What those get right:

- **Slot 2 changes the number; slot 3 changes the party.** Different kinds of
  change, so the three options do not collapse onto one axis.
- **Everything else is identical in shape** — same clause opening, same "in
  writing to … at his address for the time being", same length. Nothing but the
  substance separates them from the answer.
- **The source's line break is not reproduced.** The gold carries it because it
  was sliced from the contract; a written passage should read as clean drafting,
  and the merge gate compares whitespace-insensitively either way.
- **Neither sentence appears in the contract.** Slot 3 is the riskier of the two
  to write, because a contract that gives one party a notice right often gives
  the other one too, a few lines down. Check before emitting.
- **The `perturbation` field names the change and its direction**, so the record
  is auditable without opening the source.

Now write slots 2 and 3 for each record below.

## USER

Category: {category_id} — {category_name}
Boundary rule — what this category is actually about: {boundary_rule}

Records. Each is one gold span from one document; write slots 2 and 3 for each.

{records}
