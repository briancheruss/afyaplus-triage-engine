`# AfyaPlus Triage Engine — Week 1 Capstone

> **Educational prototype, not a medical device.** Do not use to provide real-world triage or to replace a qualified clinician or emergency services. Test with fictional messages only.

A single-file Python application that converts unstructured patient text into AfyaPlus's **exact four-field JSON contract**, using **GPT-4o-mini first** and **local Ollama (`llama3.2`) as a fallback** if cloud connectivity or validation fails. It uses strict timeouts, role-based guardrails, an internal stepwise decision checklist, local schema-constrained output, explicit `json.loads()` and type checks, and safe manual review when both providers are unavailable.

## 1. Quick start (Windows PowerShell)

From the project folder, with Python 3.10+ installed:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Put your real OpenAI API key into `.env` (replace the placeholder); **never paste the key into GitHub or screenshots**.

Install [Ollama](https://ollama.com/download) for Windows, then in PowerShell:

```powershell
ollama pull llama3.2
ollama list
```

The Ollama desktop installation normally starts the background service. If not, run `ollama serve` in a separate terminal. The local API endpoint is `http://localhost:11434/api/chat` by default.

**Run the app:**

```powershell
python app.py "I have a mild cough today."
python app.py "I am pregnant and have a severe headache and blurred vision."
python app.py --demo
```

A plain `python app.py` also works: it uses an explicitly synthetic mild-cough test message. `--demo` requires neither internet nor Ollama; it uses mocked HTTP responses and is marked **SIMULATED**. It proves code wiring, **not** model accuracy or real provider availability.

For Linux/macOS: replace the activation command with `source venv/bin/activate` and `cp .env.example .env`.

## 2. System workflow

```text
 Patient message (untrusted)
           |
  Trusted prompt v3 (default)
           |
  GPT-4o-mini, JSON Mode
  full cloud deadline 4.0 s
       /            \
  valid JSON       network/HTTP/timeout/invalid JSON
    + schema                 |
       |               Ollama llama3.2
       |                JSON schema format
       |                      |
       |                validated JSON?
       |                   /       \
       |                 yes        no / offline
       |                  |               |
       |                  |       manual triage fail-safe
       +------------------+---------------+
                          |
               safety consistency check
                          |
                 exact four-key dict
                          |
              one-line routing decision
```

- Cloud request is an HTTP POST to `https://api.openai.com/v1/chat/completions` using `model="gpt-4o-mini"` and **native `response_format={"type":"json_object"}`**, exactly as specified in the brief. For cloud requests, `asyncio.wait_for(..., 4.0)` is an end-to-end deadline, complemented by `httpx.Timeout(4.0)`.
- Ollama is called **programmatically**, not merely via CLI, using `POST /api/chat`, `stream=false`, and the JSON Schema as the `format` payload.
- Cloud credentials are loaded from an explicit `.env` path beside `app.py` using `python-dotenv`. Missing credentials route to Ollama instead of crashing.
- Model text is passed to `json.loads()` and verified: exact keys, true Boolean, list of strings, summary string, and routing option. Cloud bad-JSON or schema errors also activate Ollama fallback.
- A narrow phrase-based red-flag overlay can escalate explicitly reported danger phrases, and any `is_critical_emergency=true` result must route to emergency services. **This overlay is unvalidated and can miss risks or cause false positives.**
- If neither model works, the program returns the same structure with `MANUAL_TRIAGE_REVIEW` and explicitly says that emergency status is **unknown**. Its required Boolean field is set to false purely as a schema placeholder: this **must never be interpreted as a negative clinical assessment**.

## 3. Exact output contract

The brief lists **four** fields:

```json
{
  "is_critical_emergency": true,
  "detected_symptoms": ["severe headache", "blurred vision"],
  "clinical_reasoning_summary": "Reported pregnancy symptoms require urgent assessment.",
  "routing_destination": "EMERGENCY_SERVICES"
}
```

**Route values used by this prototype:** `EMERGENCY_SERVICES`, `URGENT_CLINICAL_REVIEW`, `ROUTINE_CARE`, `MANUAL_TRIAGE_REVIEW`.

The terminal prints the **parsed Python dictionary** (not a JSON string) and one line `ROUTING: ...`; it also indicates which provider handled the message and timing when live calls are made.

## 4. Prompt engineering log

| Iteration | Prompt style | Changes / rationale | Remaining limitation |
|---|---|---|---|
| **v1** (`--prompt 1`) | Zero-shot | Asks for symptom classification and JSON only. Baseline to compare against stronger versions. | Minimal instruction on untrusted text, routing, and fabrication. |
| **v2** (`--prompt 2`) | Few-shot | Adds contrasting mild-cough vs. breathing/chest-pain examples; provides explicit JSON schema and bans invented facts/calculations. | Examples alone cannot guarantee safe handling of ambiguous symptoms or prompt injection. |
| **v3** (`--prompt 3`, default) | Role-based + stepwise checklist + guardrails | Names a limited assistant role; internally orders extraction → danger check → routing → concise reason → schema validation; treats patient instructions as untrusted; prohibits fluff, extra fields, diagnosis, unverified values, or detailed chain-of-thought. | Prompting cannot guarantee clinical correctness, and no model is clinically validated here. |

The final prompt includes a **step-by-step decision process** as requested by the assignment, but does **not** ask the model to reveal a private reasoning transcript. Its visible `clinical_reasoning_summary` is intentionally short and limited to reported evidence.

To compare prompt variants on *synthetic* examples:

```powershell
python app.py "I have a mild cough today" --prompt 1
python app.py "I have a mild cough today" --prompt 2
python app.py "I have a mild cough today" --prompt 3
```

## 5. Tests and end-to-end examples

```powershell
python -m unittest discover -s tests -v
python app.py --demo
python app.py --demo > demo_run.txt 2>&1
```

- `tests/test_app.py` validates JSON parsing, exact schema, route consistency, emergency overlay, native cloud JSON Mode payload, native Ollama schema payload, cloud HTTP 503 fallback, timeout fallback, invalid JSON fallback, cloud/local comparison, and dual-provider failure.
- `sample_outputs.txt` holds **three actual program executions through mocked HTTP endpoints**: routine cough, reported pregnancy warning signs, and cloud HTTP 503 fallback to a simulated local response.
- **Required real-world evidence still to collect:** after installing Ollama and configuring cloud credentials, run three additional live scenarios and capture the terminal logs or screenshots for submission. At least one screenshot should demonstrate **actual local Ollama** fallback after cloud failure. The included sample file alone is not proof of a live Ollama deployment.

For **a genuine offline failover demo**, first ensure `ollama list` shows `llama3.2`, then temporarily move your key out of `.env` or give `OPENAI_API_KEY` an invalid value, and run:

```powershell
python app.py "I have a mild cough today."
```

You should see `SOURCE: ollama_fallback`, assuming Ollama is running. For an explicit cloud HTTP failure, an invalid (nonempty) key should lead to a 401 and then local fallback. Restore the real key after the demonstration. Do not expose it in screenshots.

## 6. Cloud vs. local latency: measurement plan and honest baseline

**Live numbers have NOT been collected in this deliverable.** The `--demo` mock responses do not measure provider inference and must never be used as benchmark data. The comparison below is a **measurement template**, not invented results.

| Test condition (same synthetic message) | GPT-4o-mini cloud | Ollama llama3.2 local |
|---|---|---|
| Hardware / network | To record | To record CPU, RAM and whether GPU is active |
| Cold start (ms) | Not measured | Not measured |
| Warm run 1 (ms) | Not measured | Not measured |
| Warm run 2 (ms) | Not measured | Not measured |
| Warm run 3 (ms) | Not measured | Not measured |
| Median warm response (ms) | Not measured | Not measured |
| Model availability during internet outage | Needs internet | Works locally when installed and service running |

With both paths configured, run the same text three times, appending numeric results to the CSV:

```powershell
python app.py "I have a mild cough today." --compare --metrics-csv metrics.csv
python app.py "I have a mild cough today." --compare --metrics-csv metrics.csv
python app.py "I have a mild cough today." --compare --metrics-csv metrics.csv
```

For each run, `LATENCY: cloud=Xms, local=Yms` reports actual elapsed Python request time; `.csv` records metadata only (no patient text). Exclude `manual_failsafe`, provider errors, and mocked results from comparison. Use at least three successful runs, label warm/cold conditions, and report the *median* rather than asserting that one architecture is always faster. Cloud may meet or miss the 4 s deadline; local cold loading may exceed 90 s depending on the hardware. Document model size, CPU/RAM/GPU, internet conditions, and whether the Ollama model was already warm.

**Performance trade-off:** cloud access does not require running local model weights but depends on internet and vendor availability; local inference can function offline and keeps messages on the machine, but adds device memory and startup costs. Neither speed nor accuracy is demonstrated until measured.

## 7. Risk register and next steps

| Risk | Impact | Current prototype control | What production would need |
|---|---|---|---|
| Missed emergency / false reassurance | Serious patient harm | Conservative routing instructions, some red-flag escalation, human-review fail-safe | Clinical expert-led validation, approved protocols, prospective monitoring, and emergency escalation procedures |
| False positives | Unnecessary escalation and load | Simple negation guard for a few phrases | A tested triage decision policy and supervised human review |
| Hallucinated clinical facts | Unsafe handling | Prompt bans fabrication + schema validator | Grounding on verified patient facts, clinical audits, regression suite |
| Model prompt injection | Output manipulation | Trusted system instructions separate from untrusted patient text | Adversarial testing, stronger policies and hard output enforcement |
| Outages / cloud latency | Stalled or lost messages | Strict 4 s cloud deadline, classified errors, Ollama fallback, manual fail-safe | Resilient infrastructure, monitoring, retries with backoff where suitable, on-call coverage |
| Patient data privacy | Exposure of sensitive data | No patient content in error logs or metrics CSV; `.env` gitignored | Consent, encryption, retention policy, access controls, applicable Kenyan privacy/legal review |
| False schema confidence | Valid JSON mistaken for correct medicine | Distinguishes JSON validity from clinical truth | Validation of clinical decisions, audited escalation rates |

**Deployment readiness:** this is a *learning prototype*, not a production-ready clinical system, despite the ambitious wording of the assignment. No clinical evaluation, real service reliability test, privacy impact review, or operational sign-off has been completed. Do not feed real patient data into it.

## 8. Grading rubric self-check

| Grading category | What is covered | What still needs live evidence |
|---|---|---|
| Prompt Quality (25) | 3 versions, role / internal reasoning checklist / guarded output | Evaluate outcomes on a varied synthetic set; include reflections |
| JSON enforcement (20) | Native JSON Mode, explicit `json.loads`, exact-key/type checks; tested | Capture live model outputs |
| API resilience (20) | 4 s cloud deadline, timeout/network/HTTP handling, local failover, dual-outage fail-safe; tested with mocked transports | Demonstrate actual cloud failure → real local model |
| Infrastructure comparison (20) | Both cloud and local HTTP integrations; benchmark flag and CSV | Install/test local model, fill **real latency data table** |
| 5-minute presentation (15) | Slide deck and speaker notes included | Present/record approximately five minutes |

## 9. Repository contents

```text
AfyaPlus-Triage-Engine/
├── app.py                  # Single executable engine requested by brief
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── sample_outputs.txt      # Explicitly simulated HTTP demonstrations
├── tests/
│   └── test_app.py
├── AfyaPlus_5_Minute_Presentation.pptx
└── presentation_speaker_notes.md
```

## 10. Publish and submit

Create an empty repository on GitHub; from the project directory:

```powershell
git init
git add .
git commit -m "Build AfyaPlus triage capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/afyaplus-triage-engine.git
git push -u origin main
```

**Submission checklist:** push the code / docs / slides, attach or link **real** logs/screenshots for 3 scenarios (including local fallback), enter real latency observations in the README, record the talk or present the slides, and submit the GitHub link. Check that `.env`, personal information, and `.csv` files with unintended content were not committed.

### API docs

- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat/create
- Ollama Chat endpoint: https://docs.ollama.com/api/chat
- Ollama Structured Outputs: https://docs.ollama.com/capabilities/structured-outputs
