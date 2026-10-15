# HACKATHON BRIEF — AMD Developer Hackathon: ACT III (lablab.ai)

**Repo location:** `docs/HACKATHON_BRIEF.md`
**Source:** https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii (checked Wed 14 Oct 2026, 23:00 Sydney)
**Our track:** Track 1 — **Intelligent Industry** (track partner: Vibe Generation)
**Our project:** Wafer-map yield copilot — classifies wafer defect patterns, traces them to the faulty tool, and tells a yield engineer what to do.

> **For AI coding agents:** this file defines what the judges want and the rules we must not break. Before building any feature, check it against Sections 3, 4 and 6. If a feature doesn't help a judging criterion or a "good projects will show" item, it's low priority. Technical specs live in `docs/specs/` (start with `SPEC_phase1_phase2.md`). If the event page changes, update this file and note it in Section 10.

---

## 1. Key dates (converted to Sydney time, AEDT = UTC+11)

| Event | Official (UTC) | Sydney time |
|---|---|---|
| Kickoff / registration closed | Mon 12 Oct, 15:00 | Tue 13 Oct, 02:00 |
| Online build phase | 12–18 Oct | 13–19 Oct |
| On-site phase (Rome, Milan, Imperia) | 17–18 Oct | Not attending (travel not covered) |
| **Submissions close** | **Sun 18 Oct, 15:00** | **Mon 19 Oct, 02:00** |
| **Our internal deadline** | | **Sat 17 Oct, 19:00** (team plan) |

- Event schedule (workshops, office hours) is still "To be announced" on the page. Check the lablab.ai Discord for updates.
- Treat our internal deadline as the real one. The buffer is for upload problems.

---

## 2. What the event is about

- Build AI agents or high-performance AI apps that run on **AMD GPUs in the cloud** using the **ROCm** software stack.
- Prize pool: **$17,000+** across the event (per-track amounts are not listed on the page).
- Any approach is allowed: language models, computer vision, audio, agents, predictive models, or a combination.
- The organisers' framing: the product should **work**, solve a **real problem for a real user**, and AMD technology should power a **meaningful part** of it.

---

## 3. Our track brief — Track 1: Intelligent Industry

**Tagline:** help factories spot problems earlier and run better.

**Problem framing:** factories already produce plenty of data (machines, cameras, sensors, maintenance reports, manuals). The hard part is knowing what matters and what to do next.

**Goal:** build a product that helps a manufacturer do at least one of:
- reduce downtime
- improve quality
- increase safety
- use resources more efficiently

**Example ideas listed by the organisers:**
- Predict equipment failures
- Detect defects from images or video
- Explain production anomalies
- Help technicians search manuals and maintenance records
- Recommend maintenance actions
- Monitor safety conditions
- Reduce energy or material waste

**Demo requirement:** show the product processing **realistic industrial data** and producing a useful **alert, explanation, recommendation or action**. Make it clear **who uses it** and **what result it improves**.

**Good projects will show:**
1. A real manufacturing problem
2. A working product, **not only a dashboard or presentation**
3. Results that operators can understand
4. A measurable improvement in time, cost, safety, quality or efficiency

### 3.1 How our project maps to the brief

| Brief item | Our answer | Where it lives |
|---|---|---|
| Real manufacturing problem | Yield excursions in semiconductor fabs: a faulty tool keeps ruining wafers until someone notices and traces it | Pitch, `docs/` |
| Ideas we cover | Detect defects from images (wafer maps) · Explain production anomalies · Recommend maintenance actions · Search maintenance knowledge | Phases 2, 4, 5 |
| Realistic industrial data | **Real** fab wafer maps (WM-811K); equipment history is simulated and labelled as such | Phases 1, 4 |
| Alert / explanation / recommendation / action | Alert fires → agent explains pattern, likely cause, suspect tool, confidence, $ at risk → engineer can hold lots / open ticket | Phases 4, 5, 6 |
| Who uses it | Yield / process engineers in a fab | Pitch, dashboard copy |
| Working product, not just a dashboard | Live pipeline: stream → classifier → alert logic → agent → dashboard, all running end to end | Phase 7 |
| Results operators understand | Plain-language explanation, wafer image + heatmap + box, "needs review" flag | Phases 2, 5, 6 |
| Measurable improvement | Lots caught early, $ saved, time-to-detect vs end-of-shift review; model metrics with confidence intervals | Phases 2, 4 |

---

## 4. Hard requirements (breaking these = not eligible for prizes)

### 4.1 AMD technology (mandatory)
- AMD is the **required technology** for prize eligibility.
- A **meaningful part of the workload** must run on AMD infrastructure or hardware. Allowed: AMD Developer Cloud, AMD Instinct accelerators, ROCm, Ryzen AI, Radeon hardware, or other approved AMD infrastructure provided during the event.
- **The AMD component must be part of the working product shown to the judges.** Training on AMD and then running everything elsewhere in the demo is risky. In our demo, the LLM agent (Qwen via vLLM) and ideally the classifier should visibly run on AMD.
- We may choose our own models, frameworks and tools, as long as they're compatible.

**Our AMD usage (keep this true and documented):**
- ~~CNN training on AMD Instinct MI300X with PyTorch on ROCm~~: **no GPU credits; the CNN is trained on CPU. Do not claim AMD training** (15 Oct)
- **The agent's LLM via AMD Token Factory** (free models on AMD GPUs; Phase 5): the AMD component of the live demo
- Benchmark numbers (training time, inference speed on AMD) for the slides

### 4.2 Event FAQ (new on the page — verify with the team)
- The event is hosted on **notebooks.amd.com/tokenfactory**. No billing or credit card is needed.
- You **don't need the $100 credit** to take part; the API from the token factory is enough.
- **Action:** whoever owns the AMD setup must confirm what token factory gives us (notebooks with GPU access for training? an LLM inference API? both?) and post it in team chat. If it provides a hosted LLM API, the agent can use it instead of self-hosting vLLM, which saves GPU time. Record the decision in Section 10.

### 4.3 Registration and team rules
- Every member must be enrolled on lablab.ai **and** signed up for the **AMD AI Developer Program (ADP)**. Also join the lablab.ai Discord.
- Teams are 1–6 people. We are 5.
- Submissions must be **original** and **MIT-compliant**. Add an MIT `LICENSE` file to the repo. Check that every dataset, model and library we use allows this (e.g. WM-811K usage terms, Qwen licence).
- Rules, prizes and terms can change; prizes may take up to 90 days to arrive.
- Required reading: lablab Hackathon Guidelines, Getting Started Guide, Hackathon Rule Book (links on the event page).

---

## 5. What we must submit

| Item | Owner | Notes |
|---|---|---|
| Project title | Person 5 | Short and memorable |
| Short description | Person 5 | One or two sentences: user + problem + result |
| Long description | Person 5 | Problem, user, how it works, AMD usage, results, limitations |
| Selected challenge | Person 5 | Track 1: Intelligent Industry |
| Technology & category tags | Person 5 | e.g. AMD, ROCm, MI300X, PyTorch, vLLM, Qwen, computer vision, agents, manufacturing |
| Cover image | Person 5 | |
| **Video presentation (demo video)** | Person 5 + all | Show the full flow live. Keep a backup recording (team plan: Saturday morning) |
| **Slide presentation (pitch deck)** | Person 5 | Must clearly explain the problem and the intended user |
| **Public GitHub repository** | SWE | README that actually runs; no secrets committed |
| Demo application platform + **application URL** | SWE + Person 4 | A hosted, working app judges can open |
| **Working prototype** | Everyone | Must run end to end |

**Repo hygiene (agents must enforce):**
- No API keys, Kaggle tokens or credentials in the repo. Use environment variables and a `.env.example`.
- No datasets or large binaries committed (see the spec's model release rules).
- README must cover: what it is, who it's for, architecture diagram, how AMD is used, how to run it, results, limitations, team.

---

## 6. Judging criteria (main hackathon)

There are four criteria. The page gives no weightings, so treat them as equal.

| Criterion | What the judges look for | How we win it |
|---|---|---|
| **Application of Technology** | How effectively the chosen models are integrated into the solution | CNN + calibrated confidence + Grad-CAM, agent with real tools (classify, root cause lookup, lot history, impact), all running on AMD. Show the pieces working together, not just existing |
| **Presentation** | Clarity and effectiveness of the presentation | Tight 3-minute demo: normal shift → tool drifts → alert → explanation → action → $ saved. One clear user, one clear story |
| **Business Value** | Impact and practical value, and how well it fits a business area | Yield excursions are costly; show lots caught early and $ impact with stated assumptions (adjustable $ per wafer) |
| **Originality** | Uniqueness and creativity; showing the approach and the system's behaviour | Many people classify WM-811K; few go pattern → faulty tool → recommended action. Also: lot-grouped honest evaluation, calibrated confidence, and a model that says "I'm not sure" |

### 6.1 Vibe Generation special prize (applies to our track's partner)
Prize: AirPods 5 for every team member, for the best solution. Judged on three criteria:
- **Hybrid:** strategic thinking, ability to **quantify value**, commercial attitude → have a clear $ model and a "who pays for this" slide
- **Becreatives:** quality of technical execution, **orchestration of AI tools**, ability to ship → working deployed app, agent using multiple tools
- **Vibe Generation:** taste, originality of concept, **design of the human + AI workflow** → show the engineer approving the hold/ticket; the AI recommends, the human decides

**Takeaway:** the "human in the loop" design and the quantified business value matter for both the main judging and this prize. Don't cut them.

---

## 7. Partner tracks and other prizes (decided: not pursuing, for reference)

- **Evolus — Business Agents on AMD:** requires an Evolus event workspace, our own open model via vLLM on AMD, at least two Evolus building blocks, driving Evolus through its APIs, plus a containerized app, public repo and runnable README. The Evolus team checks workspace usage stats. **Team decision: skip** (too much extra work for our one strong software engineer). 500 Evolus coupons are available first-come, first-served if we change our minds.
- **Google partner prize:** mentioned on the page as optional, with no details given.
- **Imment startup incubation:** for the best **Italian** teams. Not applicable to us.
- **Tracks 2–4** (Health, Commerce, New Experiences): not our track. Ignore.

---

## 8. Pitch checklist (built from the criteria)

- [ ] Opening: who the user is (yield engineer) and the problem in one sentence
- [ ] Why it matters: cost of a yield excursion, with a sourced number and stated assumptions
- [ ] Live demo of the full flow (Section 3.1)
- [ ] The AI recommends, the human decides (hold lots / open ticket)
- [ ] Results: model metrics from the final test run only, with confidence intervals; detection speed and lots saved from the simulation
- [ ] Honesty slide: real wafer maps, simulated equipment history, lot-grouped evaluation, known limitations
- [ ] AMD slide: what ran where (MI300X, ROCm, vLLM), with real timings
- [ ] Business slide: who pays, how it plugs into a fab's existing systems, value per line/fab
- [ ] Every number is traceable to `docs/results/` or the impact calculator

---

## 9. Common ways to lose (agents: flag these if you see them)

- The demo works locally but the **application URL** doesn't, or the AMD endpoint is switched off when judges try it
- AMD is only used for training and nothing in the live product runs on AMD
- Showing a dashboard with mock data instead of a working pipeline
- Headline accuracy that is inflated by the 85% `none` class or by a leaky split
- Claims that can't be traced to a result file
- Secrets committed to the public repo
- Missing video, slides or cover image at submission time
- Running out of GPU access because vLLM was left running

---

## 10. Changes and decisions log

| Date (Sydney) | Who | Change / decision |
|---|---|---|
| 14 Oct | brief | Created from the event page. Prize pool now shows $17,000+ (was $12,000+). New FAQ: event hosted on notebooks.amd.com/tokenfactory, no billing, $100 credit not required |
| 14 Oct | team | Track 1 chosen; Evolus partner track skipped |
| 15 Oct | team | No AMD cloud GPU credits: CNN trained on CPU; AMD component = agent LLM on Token Factory. MIT `LICENSE` added |
| 14 Oct | team | Token Factory = **LLM API only**: one shared team key `rc-...` for free OpenAI/Anthropic-compatible models on AMD GPUs (30 RPM, 8 concurrent, daily cap, team-wide). The agent uses it, which is our live AMD component. **No GPU instances** in our dashboard: CNN training needs AMD Developer Cloud (the $100 credit) or a teammate's AMD GPU; otherwise it trains on CPU and the pitch must not claim AMD training |
