"""Author 75 deterministic scenarios with real semantic chunk IDs and seed-13 stratified splitting."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def generate_75_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    # 1. 30 Compound Scenarios
    compound_pairs = [
        ('What does the Growth plan cost per month?', ["nx-pricing§growth"],
         'how are API requests authenticated with an API key?', ["nx-api-auth§api-keys"]),
        ('How do ticket waitlists work when an event sells out?', ["nx-feature-ticketing§waitlists"],
         'what uptime does the Nexora SLA commit to?', ["nx-sla§uptime"]),
        ('How long is attendee personal data retained after an event?', ["nx-privacy-gdpr§data-retention"],
         'what are the API rate limits for Growth workspaces?', ["nx-api-rate-limits§limits-by-plan"]),
        ('How does the Check-in app scan tickets and print badges on site?', ["nx-feature-ticketing§check-in"],
         'which scopes are available for API keys and OAuth tokens?', ["nx-api-auth§scopes"]),
        ('What is included in the Enterprise plan?', ["nx-pricing§enterprise"],
         'how does the Salesforce integration sync attendees to campaigns?', ["nx-integration-salesforce§overview"]),
        ('How is the attendee engagement score calculated?', ["nx-feature-attendee-tracking§engagement-score"],
         'what service credits apply when uptime falls below 99.9%?', ["nx-sla§service-credits"]),
        ('Who founded Nexora and when?', ["nx-company§history"],
         'what does the SOC 2 Type II report cover?', ["nx-security§soc2"]),
        ('What new features shipped in the v2.4 release highlights?', ["nx-release-2-4§highlights"],
         'how do I handle HTTP 429 Too Many Requests responses?', ["nx-api-rate-limits§handling-429"]),
        ('How do breakout rooms and networking lounges work?', ["nx-feature-virtual-events§breakout-rooms"],
         'how is data encrypted in transit and at rest?', ["nx-security§encryption"]),
        ('How do discount codes and pricing rules work?', ["nx-feature-ticketing§discount-codes"],
         'how do I set up the HubSpot integration?', ["nx-integration-hubspot§setup"]),
        ('What widgets does the analytics dashboard show for each event?', ["nx-feature-analytics§dashboard-widgets"],
         'how does cursor pagination work in the API?', ["nx-api-overview§pagination"]),
        ('How do I invite my team and assign workspace roles?', ["nx-onboarding§team-invites"],
         'what is the support response time for Severity 1 tickets?', ["nx-sla§support-response"]),
        ('What does the AI agenda builder do?', ["nx-feature-copilot§agenda-builder"],
         'how are webhook deliveries signed and retried?', ["nx-api-endpoints§webhooks"]),
        ('How are refunds and ticket transfers handled?', ["nx-feature-ticketing§refunds-transfers"],
         'what does the Starter plan include?', ["nx-pricing§starter"]),
        ('Where is workspace data hosted and can support access be restricted to the EU?', ["nx-privacy-gdpr§data-residency"],
         'which Zapier triggers and actions are available?', ["nx-integration-zapier§triggers-actions"]),
        ('How does lead retrieval work for exhibitors at their booth?', ["nx-feature-attendee-tracking§lead-retrieval"],
         'what were the v2.5 release highlights?', ["nx-release-2-5§highlights"]),
        ('How do hybrid events combine in-person and virtual attendees?', ["nx-feature-virtual-events§hybrid-events"],
         'how do I rotate and revoke an API key without downtime?', ["nx-api-auth§key-rotation"]),
        ('How long are session recordings stored and when are they available on demand?', ["nx-feature-virtual-events§recordings"],
         'what are the Copilot usage limits for each plan?', ["nx-feature-copilot§copilot-limits"]),
        ('How do I create my first event?', ["nx-onboarding§first-event"],
         'which API endpoints register and check in attendees?', ["nx-api-endpoints§attendees-endpoints"]),
        ('How do I import attendee lists from a CSV file?', ["nx-onboarding§data-import"],
         'what does the Slack daily digest notification contain?', ["nx-integration-slack§notifications"]),
        ('What real-time metrics does the dashboard show during a live event?', ["nx-feature-analytics§real-time-metrics"],
         'how does SAML single sign-on and SCIM provisioning work?', ["nx-security§access-control"]),
        ('How does smart matchmaking recommend people to meet?', ["nx-feature-copilot§smart-matchmaking"],
         'how do bulk endpoints help stay within rate limits?', ["nx-api-rate-limits§bulk-endpoints"]),
        ('What is Nexora Stage and how does live streaming work?', ["nx-feature-virtual-events§streaming"],
         'how are data subject deletion requests handled?', ["nx-privacy-gdpr§subject-requests"]),
        ('How does ROI and pipeline attribution work?', ["nx-feature-analytics§roi-attribution"],
         'what does the HubSpot integration sync to contacts?', ["nx-integration-hubspot§overview"]),
        ('How much does annual billing save and how long is the free trial?', ["nx-pricing§billing"],
         'what is the API base URL and current version?', ["nx-api-overview§base-url"]),
        ('How many employees does Nexora have and where are its offices?', ["nx-company§team-offices"],
         'how does the Salesforce sync retry failed records?', ["nx-integration-salesforce§sync-behavior"]),
        ('What should I verify on the go-live checklist before an event?', ["nx-onboarding§go-live-checklist"],
         'which rate limit headers does the API return?', ["nx-api-rate-limits§rate-limit-headers"]),
        ('Which add-ons can be purchased for extra attendees or Copilot requests?', ["nx-pricing§add-ons"],
         'how do I set up the Slack integration for a private channel?', ["nx-integration-slack§setup"]),
        ('How much funding has Nexora raised and from whom?', ["nx-company§funding"],
         'which HTTP error codes does the API return?', ["nx-api-overview§errors"]),
        ('What fixes and deprecations came in release v2.3?', ["nx-release-2-3§fixes"],
         'how do I troubleshoot a Zapier zap that stopped running?', ["nx-integration-zapier§troubleshooting"]),
    ]
    for idx, (q1, g1, q2, g2) in enumerate(compound_pairs, start=1):
        scenarios.append({
            "scenario_id": f"compound_{idx:02d}",
            "category": "compound",
            "query": f"{q1} and {q2}",
            "sub_intents": [
                {"id": "sub_1", "query": q1, "gold_chunk_ids": g1},
                {"id": "sub_2", "query": q2, "gold_chunk_ids": g2}
            ],
            "is_unanswerable": False
        })

    # 2. 15 Single-Early Scenarios
    single_early_queries = [
        ('How long does a waitlist offer hold a released ticket before it moves on?', ["nx-feature-ticketing§waitlists"]),
        ('How many API keys can an Enterprise workspace create?', ["nx-api-overview§api-availability"]),
        ('How long are OAuth access tokens and refresh tokens valid?', ["nx-api-auth§oauth"]),
        ('What discount do nonprofit and education organizations receive?', ["nx-pricing§billing"]),
        ('How much does an additional 1,000 attendees per event cost as an add-on?', ["nx-pricing§add-ons"]),
        ('Which identity providers can be used for SAML single sign-on?', ["nx-security§access-control"]),
        ('How many times are failed webhook deliveries retried?', ["nx-api-endpoints§webhooks"]),
        ('What is the maximum page size for API list endpoints?', ["nx-api-overview§pagination"]),
        ('How much did Nexora raise in its Series B round?', ["nx-company§funding"]),
        ('Who is the Chief Financial Officer of Nexora?', ["nx-company§leadership"]),
        ('When does Nexora perform scheduled maintenance?', ["nx-sla§maintenance-incidents"]),
        ('How quickly are critical penetration test findings fixed?', ["nx-security§testing-disclosure"]),
        ('How many concurrent live stream viewers does Nexora Stage support on Growth?', ["nx-support-faq§faq-virtual"]),
        ('How many rows can a single attendee CSV import contain?', ["nx-onboarding§data-import"]),
        ('What time is the Slack daily registration digest posted?', ["nx-integration-slack§notifications"]),
    ]
    for idx, (q, g) in enumerate(single_early_queries, start=1):
        scenarios.append({
            "scenario_id": f"single_early_{idx:02d}",
            "category": "single-early",
            "query": q,
            "sub_intents": [
                {"id": "sub_1", "query": q, "gold_chunk_ids": g}
            ],
            "is_unanswerable": False
        })

    # 3. 15 Presentation Scenarios
    presentation_queries = [
        ('Summarize the Nexora Enterprise plan features.', ["nx-pricing§enterprise"]),
        ('Summarize the GDPR roles and the data processing agreement.', ["nx-privacy-gdpr§gdpr-role"]),
        ('Summarize how Nexora encrypts data in transit and at rest.', ["nx-security§encryption"]),
        ('Summarize the service credits in the Nexora SLA.', ["nx-sla§service-credits"]),
        ('Summarize the v2.4 release highlights for hybrid events and Slack.', ["nx-release-2-4§highlights"]),
        ('Summarize the Salesforce integration setup steps.', ["nx-integration-salesforce§setup"]),
        ('Summarize the Nexora company history since its founding.', ["nx-company§history"]),
        ('Summarize how the attendee engagement score is calculated.', ["nx-feature-attendee-tracking§engagement-score"]),
        ('Summarize the Copilot AI session summaries feature.', ["nx-feature-copilot§session-summaries"]),
        ('Summarize the HubSpot integration sync behavior.', ["nx-integration-hubspot§sync-behavior"]),
        ('Summarize the analytics dashboard exports and custom reports.', ["nx-feature-analytics§exports-reports"]),
        ('Summarize the API error codes and error responses.', ["nx-api-overview§errors"]),
        ('Summarize the ticket types organizers can create.', ["nx-feature-ticketing§ticket-types"]),
        ('Summarize the v2.5 improvements to the HubSpot and Salesforce integrations.', ["nx-release-2-5§improvements"]),
        ('Summarize how session attendance is tracked with badge scans.', ["nx-feature-attendee-tracking§session-attendance"]),
    ]
    for idx, (q, g) in enumerate(presentation_queries, start=1):
        scenarios.append({
            "scenario_id": f"presentation_{idx:02d}",
            "category": "presentation",
            "query": q,
            "sub_intents": [
                {"id": "sub_1", "query": q, "gold_chunk_ids": g}
            ],
            "is_unanswerable": False
        })

    # 4. 15 Unanswerable Scenarios
    unanswerable_queries = [
        "What was the stock market trading volume of Nexora in 1920?",
        "What are the weather conditions on Mars right now?",
        "Who won the soccer world cup final in the year 1850?",
        "What is the secret baking recipe for Nexora brand chocolate cookies?",
        "Which airline operates daily direct flights from Tokyo to Atlantis?",
        "What is the average lifespan of a wild dragon in medieval folklore?",
        "How many electric cars were manufactured in California during 1880?",
        "What is the municipal tax rate of the floating city of El Dorado?",
        "Who was the prime minister of Antarctica during the nineteenth century?",
        "What is the current price of interstellar warp drive engines?",
        "Which baseball team won the championship on the moon in 1969?",
        "What is the official currency exchange rate of Narnia lion coins?",
        "How many coffee cups were consumed during the signing of Magna Carta?",
        "What is the repair manual for time-travel tachyon flux capacitors?",
        "Which company invented underwater supersonic passenger trains in 1910?"
    ]
    for idx, q in enumerate(unanswerable_queries, start=1):
        scenarios.append({
            "scenario_id": f"unanswerable_{idx:02d}",
            "category": "unanswerable",
            "query": q,
            "sub_intents": [
                {"id": "sub_1", "query": q, "gold_chunk_ids": []}
            ],
            "is_unanswerable": True
        })

    return scenarios


# Phase 7: two-turn late-detail sessions. Turn 1 is a content question (its sub-intents carry
# the gold chunks); turn 2 ("follow_up") is a constraint that modifies one sub-intent. Gold:
# relation, affected_sub_intents and the delta evidence the constraint should retrieve.
LATE_DETAIL_SESSIONS = [
    ('What does the Growth plan cost per month?', ["nx-pricing§growth"],
     'how do API keys authenticate requests?', ["nx-api-auth§api-keys"],
     'Assume we pay annually instead of monthly.', 'sub_1', ["nx-pricing§billing"]),
    ('How do ticket waitlists work?', ["nx-feature-ticketing§waitlists"],
     'what uptime does the Nexora SLA commit to?', ["nx-sla§uptime"],
     'Assume we are on the Enterprise plan and monthly uptime drops below 99.9%.', 'sub_2', ["nx-sla§service-credits"]),
    ('How long is attendee personal data retained after an event?', ["nx-privacy-gdpr§data-retention"],
     'how does the HubSpot integration sync contacts?', ["nx-integration-hubspot§overview"],
     'Assume an attendee asks us to delete their personal data.', 'sub_1', ["nx-privacy-gdpr§subject-requests"]),
    ('How does the Check-in app scan tickets on site?', ["nx-feature-ticketing§check-in"],
     'how are API rate limits applied per workspace?', ["nx-api-rate-limits§limits-by-plan"],
     'Assume our sync script exceeds the rate limit and receives HTTP 429 responses.', 'sub_2', ["nx-api-rate-limits§handling-429"]),
    ('How do I create my first event?', ["nx-onboarding§first-event"],
     None, None,
     'Assume the event is virtual and we need a rehearsal before going live.', 'sub_1', ["nx-onboarding§go-live-checklist"]),
    ('What widgets does the analytics dashboard show?', ["nx-feature-analytics§dashboard-widgets"],
     'how are webhook deliveries signed?', ["nx-api-endpoints§webhooks"],
     'Assume we want to stream the event data into Snowflake.', 'sub_1', ["nx-feature-analytics§exports-reports"]),
    ('How does the Salesforce integration sync attendees?', ["nx-integration-salesforce§overview"],
     'what does the Starter plan include?', ["nx-pricing§starter"],
     'Assume some records fail to sync to Salesforce.', 'sub_1', ["nx-integration-salesforce§sync-behavior"]),
    ('How does live streaming work in Nexora Stage?', ["nx-feature-virtual-events§streaming"],
     'how is data encrypted at rest?', ["nx-security§encryption"],
     'Assume the event also has an in-person venue with checked-in attendees.', 'sub_1', ["nx-feature-virtual-events§hybrid-events"]),
    ("Explain Raft leader election and log replication.", ["DOC002§raft_consensus"],
     None, None,
     "Assume the Raft cluster has five nodes and must tolerate two failures.", "sub_1", ["DOC002§raft_consensus"]),
    ("How do LSM trees handle writes?", ["DOC003§lsm_trees"],
     "how do B+Trees organize disk pages?", ["DOC003§b_trees"],
     "Assume the LSM tree uses leveled compaction to merge overlapping SSTables.", "sub_1", ["DOC003§lsm_trees"]),
    ("How does HNSW search high-dimensional vectors?", ["DOC004§hnsw_indexing"],
     "how does Okapi BM25 rank documents?", ["DOC004§bm25_lexical"],
     "Assume the BM25 parameters k1 and b are tuned for document length normalization.", "sub_2", ["DOC004§bm25_lexical"]),
    ("What is the surface code?", ["DOC001§error_correction"],
     "how do vector clocks detect causality?", ["DOC002§vector_clocks"],
     "Assume a node receives a message carrying another vector clock.", "sub_2", ["DOC002§vector_clocks"]),
    ('How are API keys created and sent with requests?', ["nx-api-auth§api-keys"],
     'what does the Slack integration post to channels?', ["nx-integration-slack§overview"],
     'Assume one of our API keys was exposed publicly and must be revoked.', 'sub_1', ["nx-api-auth§key-rotation"]),
    ('What does the AI agenda builder do?', ["nx-feature-copilot§agenda-builder"],
     'which Zapier triggers are available?', ["nx-integration-zapier§triggers-actions"],
     'Assume we are on Starter with only 50 Copilot requests per month.', 'sub_1', ["nx-feature-copilot§copilot-limits"]),
    ('How are refunds handled for ticket orders?', ["nx-feature-ticketing§refunds-transfers"],
     'who leads Nexora?', ["nx-company§leadership"],
     'Assume the refund is for a group order with several tickets.', 'sub_1', ["nx-release-2-5§fixes"]),
]


def generate_late_detail_scenarios() -> List[Dict[str, Any]]:
    """15 two-turn late-detail scenarios (category "late_detail")."""
    scenarios: List[Dict[str, Any]] = []
    for idx, (q1, g1, q2, g2, follow_up, affected, delta_gold) in enumerate(LATE_DETAIL_SESSIONS, start=1):
        subs = [{"id": "sub_1", "query": q1, "gold_chunk_ids": g1}]
        if q2:
            subs.append({"id": "sub_2", "query": q2, "gold_chunk_ids": g2})
        scenarios.append({
            "scenario_id": f"late_{idx:02d}",
            "category": "late_detail",
            "query": f"{q1} and {q2}" if q2 else q1,
            "sub_intents": subs,
            "is_unanswerable": False,
            "follow_up": {
                "text": follow_up,
                "relation": "modifies",
                "affected_sub_intents": [affected],
                "delta_gold_chunk_ids": delta_gold,
            },
            "unaffected_claims_expected": True,
        })
    return scenarios


def split_late_detail(scenarios: List[Dict[str, Any]], seed: int = 13) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Seed-13 ~50/50 split of the late-detail category with its own RNG, so adding it leaves the
    frozen split of the original 75 scenarios unchanged."""
    return make_stratified_split(scenarios, seed=seed)


def make_stratified_split(
    scenarios: List[Dict[str, Any]],
    seed: int = 13
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for s in scenarios:
        categories.setdefault(s["category"], []).append(s)

    tune_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for cat in sorted(categories.keys()):
        items = list(categories[cat])
        rng.shuffle(items)
        split_idx = len(items) // 2
        tune_set.extend(items[:split_idx])
        test_set.extend(items[split_idx:])

    tune_set.sort(key=lambda s: s["scenario_id"])
    test_set.sort(key=lambda s: s["scenario_id"])
    return tune_set, test_set


SPLIT_SEED = 13


def write_split_files(base_dir: str = "eval/scenarios") -> None:
    """Write tune.jsonl / test.jsonl (+ late_tune.jsonl / late_test.jsonl) under base_dir and the frozen
    split manifest at <base_dir>/../split.json."""
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    scenarios = generate_75_scenarios()
    tune_set, test_set = make_stratified_split(scenarios, seed=SPLIT_SEED)

    for name, items in (("tune", tune_set), ("test", test_set)):
        with open(p / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    late_tune, late_test = split_late_detail(generate_late_detail_scenarios(), seed=SPLIT_SEED)
    for name, items in (("late_tune", late_tune), ("late_test", late_test)):
        with open(p / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    manifest = {
        "seed": SPLIT_SEED,
        "tune": [s["scenario_id"] for s in tune_set],
        "test": [s["scenario_id"] for s in test_set],
        "late_detail": {
            "tune": [s["scenario_id"] for s in late_tune],
            "test": [s["scenario_id"] for s in late_test],
        },
    }
    with open(p.parent / "split.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
