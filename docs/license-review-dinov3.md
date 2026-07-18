# DINOv3 License Review & Gated-Access Request (Task 0 — GATING)

> **Status:** OPEN — this is a hard gate. Production DINOv3 weights **must not**
> be deployed to production until (a) legal has signed off on the Meta AI DINOv3
> license and acceptable-use policy, **and** (b) HuggingFace gated access is
> approved (a token can actually pull the weights) **or** the SigLIP 2 fallback
> decision is recorded below.
>
> All embedding-dependent build work proceeds on the **SigLIP 2 (Apache-2.0)**
> encoder in the meantime — the pipeline is encoder-agnostic, so this gate never
> blocks Phase 1 / Phase 2 delivery.

---

## 1. Purpose & scope

This document (1) records the steps to request gated access to the
`facebook/dinov3-*` model family on HuggingFace, (2) summarizes the Meta AI
DINOv3 custom license and acceptable-use policy for legal review, and (3)
provides a decision record to either **proceed with DINOv3** or **activate the
SigLIP 2 fallback**.

Target model for this project: `facebook/dinov3-vitb16-pretrain-lvd1689m`
(ViT-B/16, embedding dim ≈ 768). Other members of the `facebook/dinov3-*`
family (ViT-S / ViT-L / ViT-H+ / ConvNeXt variants) are governed by the same
license and require the same access request.

---

## 2. HuggingFace gated-access request — step by step

DINOv3 models are **gated** on the HuggingFace Hub: you must accept Meta's
license terms and be granted access before a token can download the weights.

1. **Have a HuggingFace account** on the team/organization that will own the
   access. Prefer a shared org account (not a personal one) so access survives
   staff changes.
2. **Open the model card:** navigate to
   `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`
   (and any sibling variants you may want, e.g. the ViT-L/16 for a
   higher-accuracy option).
3. **Read the license prompt** shown on the gated model page ("You need to agree
   to share your contact information to access this model"). It links to the Meta
   AI DINOv3 License Agreement and the Acceptable Use Policy — **do not click
   accept until §3 legal review below is complete and signed off.**
4. **Submit the access request form.** Meta's form typically asks for:
   contact name, email, affiliation/organization, country, and intended use.
   Fill in the **organization** (not a personal identity) and describe the
   intended use accurately (e.g. "internal product-image visual similarity
   search for a pharmacy e-commerce catalog").
5. **Accept the license** (only after legal sign-off) to complete submission.
6. **Wait for approval.** Approval is **manual and can take days** (sometimes
   longer). Track the pending state on the model page ("Your request to access
   this repo is awaiting review"). **Start this request immediately** — it is the
   long-latency item.
7. **Provision a token for automated pulls.** Once approved, generate a
   **fine-grained read token** scoped to the gated repo(s). Store it as a secret
   referenced by env-var **name** only — `HUGGINGFACE_HUB_TOKEN` (aka `HF_TOKEN`)
   — in the indexing/ML environment. **Never** commit the literal token value to
   the repo or any doc.
8. **Verify access:** with the token exported as `HUGGINGFACE_HUB_TOKEN`, confirm
   a download succeeds, e.g.:

   ```bash
   # token supplied via env var name only — never inline the value
   huggingface-cli whoami
   huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
     --include "*.json" "*.safetensors" --token "$HUGGINGFACE_HUB_TOKEN"
   ```

   A `403`/`GatedRepoError` means access is still pending or the token lacks the
   grant.

> **Per-user gating note:** HuggingFace gated approval is granted to the
> account/token that accepted the terms. Ensure the **service/org** token (not an
> individual's) holds the grant so CI and the indexing host can pull weights
> without depending on one person's account.

---

## 3. Meta AI DINOv3 license — summary for legal review

> This is an engineering summary to orient legal review; it is **not** legal
> advice and does **not** replace reading the full agreement linked on the model
> card. The authoritative text is the **DINOv3 License Agreement** and the
> **DINOv3 Acceptable Use Policy** published by Meta on the model page.

DINOv3 is released under a **custom Meta AI license** (not Apache-2.0 / MIT —
this is the crux of why the gate exists). Key terms to confirm with legal:

### 3.1 Commercial use
- **Commercial use is permitted** under the license grant (subject to the
  conditions below and the acceptable-use policy). This is compatible with our
  commercial pharmacy e-commerce deployment **if** legal confirms the specific
  grant covers our use.

### 3.2 Redistribution & derivatives
- If you **redistribute** the model, its weights, or derivatives (including a
  fine-tuned/derived model or embeddings-producing artifact), you must:
  - **include a copy of the license** with the distribution, and
  - **retain attribution / notices** required by the agreement.
- Our deployment does **not** redistribute weights (weights stay inside our
  private ML service / container registry), which reduces redistribution
  obligations — **confirm with legal** that our container-internal use is "use,"
  not "distribution." If we ever ship a model artifact to a third party, the
  redistribution obligations apply.

### 3.3 Attribution / acknowledgement
- The license carries **attribution/acknowledgement requirements** (e.g.
  crediting the model and Meta AI, and not implying endorsement). Track where we
  must surface attribution (docs, NOTICE file, or user-facing "powered by"
  text if applicable). Confirm exact wording obligations with legal.

### 3.4 Acceptable-use policy — prohibited uses
The Acceptable Use Policy prohibits categories including (non-exhaustive —
confirm the current list):
- **Military, warfare, weapons** development or nuclear applications.
- **Surveillance** that violates people's rights / illegal surveillance, and
  uses that violate privacy or civil rights.
- **Illegal or harmful activity** (violence, exploitation, harassment,
  discrimination), generating disinformation, or violating others' rights.
- Uses that violate applicable laws/regulations, or the rights of third parties.

Our use — visual similarity retrieval over our own pharmacy product-image
catalog — does not fall into these prohibited categories, but legal must confirm
against the current policy text.

### 3.5 Other terms to check
- **Disclaimer of warranty / limitation of liability** (as-is, no warranty).
- **Termination** conditions (grant can terminate on breach).
- Any **jurisdiction / governing-law** clauses.
- Whether any **field-of-use** or **user-count / revenue** thresholds apply
  (some Meta licenses have had such conditions historically — verify current
  DINOv3 terms).

---

## 4. Fallback: SigLIP 2 (Apache-2.0)

If DINOv3 access is not approved in time, or legal finds the terms unacceptable,
we **activate SigLIP 2** as the launch/production encoder:

- **License:** Apache-2.0 — permissive, commercial-use-friendly, no gated
  access, no redistribution/attribution blockers beyond the standard Apache
  NOTICE handling.
- **Embedding dim:** ≈ 1152 (encoder-specific; see per-encoder collection
  naming, e.g. `image_embeddings_siglip2_1152`).
- **Impact:** none on the build path — the encoder is selected via the `ENCODER`
  config (`dinov3` | `siglip2`) and the collection is chosen via
  `ACTIVE_COLLECTION`. Swapping later = index the DINOv3 collection, flip
  `ACTIVE_COLLECTION`, recalibrate thresholds (see `docs/runbook.md`).

SigLIP 2 is in fact the **Phase 1 launch encoder** regardless (see runbook); the
DINOv3 decision only affects whether we later swap in DINOv3 for a possible
accuracy gain.

---

## 5. Decision record

| Field | Value |
|---|---|
| **Decision** | ☐ Proceed with DINOv3   ☐ Activate SigLIP 2 (Apache-2.0) fallback |
| **Target DINOv3 model** | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| **HF access status** | ☐ Approved   ☐ Pending   ☐ Denied / N/A |
| **HF request submitted (date)** | __________ |
| **HF access granted (date)** | __________ |
| **Legal review outcome** | ☐ Terms acceptable   ☐ Terms unacceptable |
| **Token stored as** | env-var name `HUGGINGFACE_HUB_TOKEN` (value in secret store only) |
| **Attribution obligations captured where** | __________ |
| **Decision rationale** | __________ |

### 5.1 Pre-production checklist (all must be ☑ before DINOv3 weights ship to prod)
- [ ] HuggingFace gated access **approved**; service/org token can pull weights.
- [ ] Meta AI DINOv3 **license reviewed and signed off by legal**.
- [ ] Acceptable-use policy reviewed — our use is **not** in a prohibited category.
- [ ] Redistribution posture confirmed (weights not distributed to third parties;
      or redistribution obligations met if they are).
- [ ] Attribution/acknowledgement requirements captured and implemented
      (NOTICE / docs / user-facing as required).
- [ ] Token stored as a secret (`HUGGINGFACE_HUB_TOKEN`) — **no literal value in
      any repo file**.
- [ ] DINOv3 collection indexed and threshold recalibration completed
      (per `docs/runbook.md` Phase 3) **before** flipping production traffic.

### 5.2 Sign-off
| Role | Name | Decision / Notes | Date | Signature |
|---|---|---|---|---|
| Legal | | | | |
| Engineering lead | | | | |
| Product owner | | | | |

---

## 6. Verification (Task 0 exit criteria)

Task 0 is **complete** when either:
- **(A) Proceed:** this doc is signed off by legal **and** HF access is approved
  (token pulls weights) — DINOv3 may be indexed and swapped per the runbook; or
- **(B) Fallback:** the SigLIP 2 fallback decision is recorded and signed off —
  DINOv3 work is deferred/cancelled.

Until then: **DINOv3 weights must not be deployed to production.**
