import json
import os
import time
from typing import Any, Literal, Optional

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase, Query, READ_ACCESS
from openai import OpenAI
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="Financial Watchlist Graph Assistant",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      .small-note {font-size: 0.88rem; color: #6b7280;}
      .intent-badge {
          display: inline-block;
          padding: 0.25rem 0.65rem;
          border-radius: 999px;
          background: #eef2ff;
          color: #3730a3;
          font-weight: 600;
          font-size: 0.85rem;
          margin-bottom: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

SETTING_NAMES = {
    "foundry_api_key": "FOUNDRY_API_KEY",
    "foundry_base_url": "FOUNDRY_BASE_URL",
    "foundry_deployment": "FOUNDRY_DEPLOYMENT",
    "neo4j_uri": "NEO4J_URI",
    "neo4j_username": "NEO4J_USERNAME",
    "neo4j_password": "NEO4J_PASSWORD",
}


def get_setting(logical_name: str) -> str:
    env_name = SETTING_NAMES[logical_name]
    value = os.getenv(env_name)

    if not value:
        try:
            value = st.secrets[logical_name]
        except (KeyError, FileNotFoundError):
            value = None

    if not value:
        raise RuntimeError(
            f"Missing configuration '{logical_name}'. "
            f"Set environment variable {env_name} or add it to "
            ".streamlit/secrets.toml."
        )

    return str(value).strip().strip('"').strip("'")


@st.cache_resource(show_spinner=False)
def get_foundry_client() -> OpenAI:
    return OpenAI(
        base_url=get_setting("foundry_base_url").rstrip("/") + "/",
        api_key=get_setting("foundry_api_key"),
        timeout=60.0,
        max_retries=1,
    )


@st.cache_resource(show_spinner=False)
def get_neo4j_driver():
    driver = GraphDatabase.driver(
        get_setting("neo4j_uri"),
        auth=(
            get_setting("neo4j_username"),
            get_setting("neo4j_password"),
        ),
        connection_timeout=15.0,
    )
    driver.verify_connectivity()
    return driver


def get_deployment_name() -> str:
    return get_setting("foundry_deployment")


class AMLQuestionRoute(BaseModel):
    intent: Literal[
        "ENTITY_RELATIONSHIP",
        "ENTITY_LAUNDERING_TRANSACTIONS",
        "TOP_LAUNDERING_SENDERS",
        "UNSUPPORTED",
    ]
    source_entity_name: Optional[str] = None
    target_entity_name: Optional[str] = None
    entity_name: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=20)
    reason: str


ROUTER_SYSTEM_PROMPT = """
You route questions for a synthetic AML knowledge graph.

Supported intents:

1. ENTITY_RELATIONSHIP
Use when the user asks about direct transactions or the relationship
between two named entities.
Required: source_entity_name and target_entity_name.

2. ENTITY_LAUNDERING_TRANSACTIONS
Use when the user asks for laundering-labelled transactions connected
to one named entity.
Required: entity_name.

3. TOP_LAUNDERING_SENDERS
Use when the user asks which entities sent the most
laundering-labelled transactions.
Required: top_n, default 5.

4. UNSUPPORTED
Use for watchlists, adverse media, merchant-risk categories,
country risk, sanctions, alerts, documents, predictions, or other
information not represented in the current graph.

Rules:
- Preserve entity names exactly as written.
- Do not invent missing entity names.
- Do not generate Cypher.
- is_laundering is synthetic IBM dataset ground truth.
""".strip()


def route_question(question: str) -> AMLQuestionRoute:
    completion = get_foundry_client().beta.chat.completions.parse(
        model=get_deployment_name(),
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format=AMLQuestionRoute,
    )

    route = completion.choices[0].message.parsed
    if route is None:
        raise ValueError("The model could not route this question.")
    return route


SOURCE_TO_TARGET_QUERY = """
MATCH
    (source:Entity {name: $source_name})
    -[:OWNS]->
    (source_account:Account)
    -[:SENT]->
    (t:Transaction)
    -[:RECEIVED_BY]->
    (target_account:Account)
    <-[:OWNS]-
    (target:Entity {name: $target_name})
RETURN
    source.id AS source_entity_id,
    source.name AS source_entity,
    target.id AS target_entity_id,
    target.name AS target_entity,
    'SOURCE_TO_TARGET' AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    t.payment_format AS payment_format,
    t.is_laundering AS is_laundering
ORDER BY t.timestamp
LIMIT 50
""".strip()


TARGET_TO_SOURCE_QUERY = """
MATCH
    (target:Entity {name: $target_name})
    -[:OWNS]->
    (target_account:Account)
    -[:SENT]->
    (t:Transaction)
    -[:RECEIVED_BY]->
    (source_account:Account)
    <-[:OWNS]-
    (source:Entity {name: $source_name})
RETURN
    source.id AS source_entity_id,
    source.name AS source_entity,
    target.id AS target_entity_id,
    target.name AS target_entity,
    'TARGET_TO_SOURCE' AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    t.payment_format AS payment_format,
    t.is_laundering AS is_laundering
ORDER BY t.timestamp
LIMIT 50
""".strip()


OUTGOING_LAUNDERING_QUERY = """
MATCH
    (subject:Entity {name: $entity_name})
    -[:OWNS]->
    (subject_account:Account)
    -[:SENT]->
    (t:Transaction)
    -[:RECEIVED_BY]->
    (counterparty_account:Account)
    <-[:OWNS]-
    (counterparty:Entity)
MATCH (subject_account)-[:HELD_AT]->(subject_bank:Bank)
MATCH (counterparty_account)-[:HELD_AT]->(counterparty_bank:Bank)
WHERE t.is_laundering = true
RETURN
    subject.id AS searched_entity_id,
    subject.name AS searched_entity,
    'OUTGOING' AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    t.payment_format AS payment_format,
    subject_bank.name AS searched_entity_bank,
    counterparty.id AS counterparty_id,
    counterparty.name AS counterparty,
    counterparty_bank.name AS counterparty_bank,
    t.is_laundering AS is_laundering
ORDER BY t.timestamp
LIMIT 50
""".strip()


INCOMING_LAUNDERERING_QUERY = """
MATCH
    (counterparty:Entity)
    -[:OWNS]->
    (counterparty_account:Account)
    -[:SENT]->
    (t:Transaction)
    -[:RECEIVED_BY]->
    (subject_account:Account)
    <-[:OWNS]-
    (subject:Entity {name: $entity_name})
MATCH (subject_account)-[:HELD_AT]->(subject_bank:Bank)
MATCH (counterparty_account)-[:HELD_AT]->(counterparty_bank:Bank)
WHERE t.is_laundering = true
RETURN
    subject.id AS searched_entity_id,
    subject.name AS searched_entity,
    'INCOMING' AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    t.payment_format AS payment_format,
    subject_bank.name AS searched_entity_bank,
    counterparty.id AS counterparty_id,
    counterparty.name AS counterparty,
    counterparty_bank.name AS counterparty_bank,
    t.is_laundering AS is_laundering
ORDER BY t.timestamp
LIMIT 50
""".strip()


TOP_LAUNDERING_SENDERS_QUERY = """
MATCH
    (entity:Entity)
    -[:OWNS]->
    (:Account)
    -[:SENT]->
    (t:Transaction)
WHERE t.is_laundering = true
WITH
    entity,
    t.payment_currency AS currency,
    count(t) AS transaction_count_for_currency,
    sum(t.amount_paid) AS total_amount_for_currency
WITH
    entity,
    sum(transaction_count_for_currency) AS laundering_transaction_count,
    collect({
        currency: currency,
        transaction_count: transaction_count_for_currency,
        total_amount: total_amount_for_currency
    }) AS totals_by_currency
RETURN
    entity.id AS entity_id,
    entity.name AS entity_name,
    laundering_transaction_count,
    totals_by_currency
ORDER BY
    laundering_transaction_count DESC,
    entity.id ASC
LIMIT $top_n
""".strip()


def run_read_query(cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    with get_neo4j_driver().session(
        database="neo4j",
        default_access_mode=READ_ACCESS,
    ) as session:
        result = session.run(Query(cypher, timeout=30.0), parameters)
        rows = [record.data() for record in result]
        result.consume()
    return rows


def generate_answer(
    question: str,
    intent: str,
    evidence: list[dict[str, Any]],
) -> str:
    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    prompt = f"""
You are a careful financial-crime graph assistant.

Answer the question using only the supplied Neo4j evidence.

Rules:
- Do not invent information.
- Keep the answer concise and easy to understand.
- Explain transaction direction where relevant.
- Mention important amounts, currencies, dates, counterparties,
  and payment formats.
- is_laundering=true is synthetic IBM AML dataset ground truth.
- It is not a real SAR, conviction, analyst decision, or model prediction.
- If evidence is empty, state that no matching records were found
  in this 50,000-transaction synthetic sample.
- Do not claim that the graph contains sanctions, watchlists,
  merchant-risk categories, adverse media, or documents.

Intent:
{intent}

Question:
{question}

Neo4j evidence:
{evidence_json}
""".strip()

    response = get_foundry_client().responses.create(
        model=get_deployment_name(),
        input=prompt,
        max_output_tokens=700,
    )
    return response.output_text.strip()


def ask_aml_graph(question: str) -> dict[str, Any]:
    total_started = time.perf_counter()

    route_started = time.perf_counter()
    route = route_question(question)
    routing_seconds = time.perf_counter() - route_started

    if route.intent == "UNSUPPORTED":
        return {
            "question": question,
            "intent": route.intent,
            "route": route.model_dump(),
            "answer": (
                "This proof of concept does not yet contain the data "
                "required to answer that question. It currently contains "
                "synthetic entities, accounts, banks, transactions, and "
                "the IBM synthetic laundering ground-truth label. "
                "Watchlists, adverse-media documents, merchant-risk "
                "categories, sanctions, and external risk information "
                "are planned extensions."
            ),
            "evidence": [],
            "evidence_row_count": 0,
            "metrics": {
                "routing_seconds": round(routing_seconds, 3),
                "neo4j_seconds": 0.0,
                "answer_generation_seconds": 0.0,
                "total_seconds": round(time.perf_counter() - total_started, 3),
            },
        }

    neo4j_started = time.perf_counter()

    if route.intent == "ENTITY_RELATIONSHIP":
        if not route.source_entity_name or not route.target_entity_name:
            raise ValueError("Two entity names are required.")
        params = {
            "source_name": route.source_entity_name,
            "target_name": route.target_entity_name,
        }
        evidence = (
            run_read_query(SOURCE_TO_TARGET_QUERY, params)
            + run_read_query(TARGET_TO_SOURCE_QUERY, params)
        )

    elif route.intent == "ENTITY_LAUNDERING_TRANSACTIONS":
        if not route.entity_name:
            raise ValueError("An entity name is required.")
        params = {"entity_name": route.entity_name}
        evidence = (
            run_read_query(OUTGOING_LAUNDERING_QUERY, params)
            + run_read_query(INCOMING_LAUNDERERING_QUERY, params)
        )

    elif route.intent == "TOP_LAUNDERING_SENDERS":
        evidence = run_read_query(
            TOP_LAUNDERING_SENDERS_QUERY,
            {"top_n": min(max(route.top_n, 1), 20)},
        )

    else:
        raise ValueError(f"Unknown intent: {route.intent}")

    evidence.sort(key=lambda row: str(row.get("timestamp", "")))
    neo4j_seconds = time.perf_counter() - neo4j_started

    answer_started = time.perf_counter()
    answer = generate_answer(question, route.intent, evidence)
    answer_seconds = time.perf_counter() - answer_started

    return {
        "question": question,
        "intent": route.intent,
        "route": route.model_dump(),
        "answer": answer,
        "evidence": evidence,
        "evidence_row_count": len(evidence),
        "metrics": {
            "routing_seconds": round(routing_seconds, 3),
            "neo4j_seconds": round(neo4j_seconds, 3),
            "answer_generation_seconds": round(answer_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }


EXAMPLES = [
    "What is the relationship between Partnership #2370 and Corporation #24457?",
    "Show laundering-labelled transactions connected to Partnership #2370.",
    "Which five entities sent the most laundering-labelled transactions?",
    "Which customers are linked to high-risk merchants?",
]


def evidence_dataframe(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    if not evidence:
        return pd.DataFrame()

    rows = []
    for row in evidence:
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                cleaned[key] = json.dumps(value, ensure_ascii=False, default=str)
            elif value is None or isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        rows.append(cleaned)
    return pd.DataFrame(rows)


if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def load_example():
    st.session_state.question = st.session_state.example_question


st.title("🔎 Financial Watchlist Graph Assistant")
st.caption(
    "Intent-routed, graph-grounded AML proof of concept using "
    "Microsoft Foundry and Neo4j AuraDB."
)

with st.sidebar:
    st.subheader("About this POC")
    st.markdown(
        """
        **Data:** IBM synthetic AML sample  
        **Graph:** Neo4j AuraDB  
        **LLM:** Microsoft Foundry  
        **Retrieval:** Trusted parameterised Cypher templates
        """
    )
    st.warning(
        "`is_laundering=true` is synthetic ground truth. "
        "It is not a real SAR, conviction, or model prediction."
    )
    st.selectbox(
        "Example question",
        EXAMPLES,
        key="example_question",
        on_change=load_example,
    )

with st.form("question_form"):
    st.text_area(
        "Ask a question",
        key="question",
        height=90,
        help="Use one of the example entities or ask for the top laundering-labelled senders.",
    )
    submitted = st.form_submit_button(
        "Analyse graph",
        type="primary",
        use_container_width=True,
    )

if submitted:
    question = st.session_state.question.strip()
    if not question:
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner(
                "Routing question, querying Neo4j, and generating a grounded answer..."
            ):
                st.session_state.last_result = ask_aml_graph(question)
        except Exception as exc:
            st.session_state.last_result = None
            st.error(f"Request failed: {type(exc).__name__}: {exc}")

result = st.session_state.last_result

if result:
    st.divider()
    st.markdown(
        f'<span class="intent-badge">{result["intent"]}</span>',
        unsafe_allow_html=True,
    )

    st.subheader("Grounded answer")
    st.markdown(result["answer"])

    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence rows", result["evidence_row_count"])
    c2.metric("Routing", f'{metrics["routing_seconds"]:.2f}s')
    c3.metric("Neo4j", f'{metrics["neo4j_seconds"]:.2f}s')
    c4.metric("Total", f'{metrics["total_seconds"]:.2f}s')

    st.subheader("Neo4j evidence")
    evidence_df = evidence_dataframe(result["evidence"])

    if evidence_df.empty:
        st.info("No graph records were returned.")
    else:
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)

    with st.expander("Technical details"):
        st.markdown("**Routing output**")
        st.json(result["route"])
        st.markdown("**Raw evidence**")
        st.json(result["evidence"])

st.markdown(
    '<p class="small-note">Portfolio POC: answers are limited to the data represented in the current graph.</p>',
    unsafe_allow_html=True,
)
