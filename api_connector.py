# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %pip install -q --upgrade openai mlflow

# COMMAND ----------

# MAGIC %pip uninstall -y neo4j-driver
# MAGIC %pip install -q --upgrade --force-reinstall "neo4j==6.2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Enable MLflow Tracing
import mlflow

# Enable automatic tracing for OpenAI API calls
mlflow.openai.autolog()

print("✅ MLflow tracing enabled for OpenAI")
print("All OpenAI API calls will be automatically logged to MLflow")

# COMMAND ----------

from openai import OpenAI

SECRET_SCOPE = "azure_secrets"

def get_secret(name: str) -> str:
    """Read a required Databricks secret and reject empty values."""
    value = dbutils.secrets.get(
        scope=SECRET_SCOPE,
        key=name
    ).strip()

    if not value:
        raise ValueError(f"Secret '{name}' is empty.")

    return value

foundry_base_url = get_secret("foundry_base_url")
foundry_api_key = get_secret("foundry_api_key")
foundry_deployment = get_secret("foundry_deployment")

client = OpenAI(
    base_url=foundry_base_url,
    api_key=foundry_api_key
)

response = client.responses.create(
    model=foundry_deployment,
    input="What is the capital of France?",
)

print(f"answer: {response.output[0]}")


# COMMAND ----------

from openai import OpenAI
from neo4j import GraphDatabase


SECRET_SCOPE = "azure_secrets"


def get_secret(name: str) -> str:
    """Read a required Databricks secret and reject empty values."""
    value = dbutils.secrets.get(
        scope=SECRET_SCOPE,
        key=name
    ).strip()

    if not value:
        raise ValueError(f"Secret '{name}' is empty.")

    return value

neo4j_uri = get_secret("neo4j_uri")
neo4j_username = get_secret("neo4j_username")
neo4j_password = get_secret("neo4j_password")

neo4j_driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_username, neo4j_password)
)

neo4j_driver.verify_connectivity()

records, summary, keys = neo4j_driver.execute_query(
    """
    MATCH (e:Entity)
    RETURN count(e) AS entity_count
    """,
    database_="neo4j"
)

print("Neo4j connection successful")
print("Entity count:", records[0]["entity_count"])

neo4j_driver.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 17 — Complete working POCStep 17 — Complete working POC

# COMMAND ----------

import json
import re
import time
from typing import Any

from openai import OpenAI
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


# ============================================================
# Configuration
# ============================================================

SECRET_SCOPE = "azure_secrets"
NEO4J_DATABASE = "neo4j"
MAX_RESULT_ROWS = 50
MAX_RELATIONSHIP_PATTERNS = 10


def get_secret(name: str) -> str:
    """Read and clean a required Databricks secret."""
    value = (
        dbutils.secrets
        .get(scope=SECRET_SCOPE, key=name)
        .strip()
        .strip('"')
        .strip("'")
    )

    if not value:
        raise ValueError(f"Secret '{name}' is empty.")

    return value


# ============================================================
# Clients
# ============================================================

foundry_client = OpenAI(
    base_url=get_secret("foundry_base_url").rstrip("/") + "/",
    api_key=get_secret("foundry_api_key")
)

foundry_deployment = get_secret("foundry_deployment")

neo4j_driver = GraphDatabase.driver(
    get_secret("neo4j_uri"),
    auth=(
        get_secret("neo4j_username"),
        get_secret("neo4j_password")
    )
)

neo4j_driver.verify_connectivity()


# ============================================================
# Graph schema supplied to the LLM
# ============================================================

GRAPH_SCHEMA = """
NODE LABELS AND PROPERTIES

Entity
- id: STRING, unique
- name: STRING

Account
- account_key: STRING, unique
- account_number: STRING

Bank
- bank_id: INTEGER, unique
- name: STRING

Transaction
- transaction_id: STRING, unique
- timestamp: LOCAL DATETIME
- amount_received: FLOAT
- receiving_currency: STRING
- amount_paid: FLOAT
- payment_currency: STRING
- payment_format: STRING
- is_laundering: BOOLEAN

RELATIONSHIPS

(Entity)-[:OWNS]->(Account)
(Account)-[:HELD_AT]->(Bank)
(Account)-[:SENT]->(Transaction)
(Transaction)-[:RECEIVED_BY]->(Account)

BUSINESS MEANING

- Entity represents a synthetic customer, company, partnership,
  corporation, or sole proprietorship.
- Account represents a bank account.
- Transaction represents one synthetic financial transaction.
- is_laundering=true is synthetic dataset ground truth.
- It is not a real SAR, analyst decision, or model prediction.
- There is currently no watchlist, adverse-media, merchant-category,
  risk-category, country-risk, or unstructured-document data.
"""


# ============================================================
# Few-shot examples
# ============================================================

CYPHER_EXAMPLES = """
EXAMPLE 1

Question:
Show laundering-labelled transactions connected to Partnership #2370.

Cypher:
MATCH (subject:Entity)-[:OWNS]->(subject_account:Account)
WHERE toLower(subject.name) = toLower('Partnership #2370')

MATCH (sender_entity:Entity)-[:OWNS]->(sender_account:Account)
MATCH (sender_account)-[:SENT]->(t:Transaction)
MATCH (t)-[:RECEIVED_BY]->(receiver_account:Account)
MATCH (receiver_entity:Entity)-[:OWNS]->(receiver_account)

WHERE t.is_laundering = true
  AND (
      sender_account = subject_account
      OR receiver_account = subject_account
  )

RETURN
    subject.id AS searched_entity_id,
    subject.name AS searched_entity,
    CASE
        WHEN sender_account = subject_account
        THEN 'OUTGOING'
        ELSE 'INCOMING'
    END AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    sender_entity.name AS sender_entity,
    receiver_entity.name AS receiver_entity
ORDER BY t.timestamp
LIMIT 50


EXAMPLE 2

Question:
What is the relationship between Partnership #2370
and Corporation #24457?

Cypher:
MATCH (source:Entity)-[:OWNS]->(source_account:Account)
WHERE toLower(source.name) = toLower('Partnership #2370')

MATCH (target:Entity)-[:OWNS]->(target_account:Account)
WHERE toLower(target.name) = toLower('Corporation #24457')

MATCH (sender_account:Account)-[:SENT]->(t:Transaction)
MATCH (t)-[:RECEIVED_BY]->(receiver_account:Account)

WHERE
    (
        sender_account = source_account
        AND receiver_account = target_account
    )
    OR
    (
        sender_account = target_account
        AND receiver_account = source_account
    )

RETURN
    source.name AS source_entity,
    target.name AS target_entity,
    CASE
        WHEN sender_account = source_account
        THEN 'SOURCE_TO_TARGET'
        ELSE 'TARGET_TO_SOURCE'
    END AS direction,
    t.transaction_id AS transaction_id,
    t.timestamp AS timestamp,
    t.amount_paid AS amount,
    t.payment_currency AS currency,
    t.payment_format AS payment_format,
    t.is_laundering AS is_laundering
ORDER BY t.timestamp
LIMIT 50


EXAMPLE 3

Question:
Which five entities sent the most laundering-labelled transactions?

Cypher:
MATCH (e:Entity)-[:OWNS]->(:Account)-[:SENT]->(t:Transaction)
WHERE t.is_laundering = true
RETURN
    e.id AS entity_id,
    e.name AS entity_name,
    count(t) AS laundering_transaction_count
ORDER BY laundering_transaction_count DESC
LIMIT 5
"""


# ============================================================
# LLM helper
# ============================================================

def call_llm(prompt: str, max_output_tokens: int = 1200):
    """Call Microsoft Foundry and return text, usage, and latency."""
    started = time.perf_counter()

    response = foundry_client.responses.create(
        model=foundry_deployment,
        input=prompt,
        max_output_tokens=max_output_tokens
    )

    latency_seconds = time.perf_counter() - started

    usage = getattr(response, "usage", None)

    usage_data = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

    return response.output_text.strip(), usage_data, latency_seconds


# ============================================================
# Cypher generation
# ============================================================

def build_cypher_prompt(question: str) -> str:
    return f"""
You are an expert Neo4j Cypher query generator for a synthetic
anti-money-laundering graph.

Generate exactly one read-only Cypher query that answers the question.

STRICT RULES

1. Return only Cypher. Do not use Markdown or explanations.
2. Use only labels, properties, and relationships from the supplied schema.
3. Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP,
   ALTER, LOAD CSV, CALL, APOC, FOREACH, GRANT, DENY, or REVOKE.
4. Never use variable-length relationship paths.
5. Do not invent watchlists, risk scores, documents, countries,
   categories, alerts, or other unavailable data.
6. User terms such as customer, company, business, or organisation
   normally refer to Entity.
7. Match entity IDs exactly using Entity.id.
8. Match entity names case-insensitively using toLower().
9. Return useful evidence, not entire nodes.
10. Include a LIMIT no greater than {MAX_RESULT_ROWS}.
11. For relationships between two entities, search transactions
    in both directions.
12. When the requested information is not represented in the graph,
    return this read-only query:

RETURN 'This POC does not yet contain the requested data.' AS message
LIMIT 1

GRAPH SCHEMA

{GRAPH_SCHEMA}

VALID EXAMPLES

{CYPHER_EXAMPLES}

USER QUESTION

{question}

CYPHER
""".strip()


def clean_cypher(raw_text: str) -> str:
    """Remove common formatting added around a generated query."""
    text = raw_text.strip()

    text = re.sub(
        r"^```(?:cypher)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s*```$", "", text)

    # Remove a possible leading label such as "Cypher:"
    text = re.sub(
        r"^\s*cypher\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.strip()

    # Remove only a final semicolon.
    if text.endswith(";"):
        text = text[:-1].strip()

    return text


# ============================================================
# Safety guardrails
# ============================================================

FORBIDDEN_CYPHER = re.compile(
    r"\b("
    r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|"
    r"LOAD\s+CSV|CALL|APOC|FOREACH|GRANT|DENY|REVOKE|"
    r"TERMINATE|START\s+DATABASE|STOP\s+DATABASE|"
    r"CREATE\s+INDEX|CREATE\s+CONSTRAINT"
    r")\b",
    flags=re.IGNORECASE
)


def validate_read_only_cypher(cypher: str) -> str:
    """
    Reject unsafe or unnecessarily broad Cypher and add a result limit.
    This is a POC application guardrail, not a replacement for DB RBAC.
    """
    if not cypher:
        raise ValueError("The LLM returned an empty Cypher query.")

    if FORBIDDEN_CYPHER.search(cypher):
        raise ValueError(
            "Generated Cypher contains a forbidden write or "
            "administrative operation."
        )

    if re.search(r"//|/\*|\*/", cypher):
        raise ValueError("Cypher comments are not allowed.")

    if ";" in cypher:
        raise ValueError("Multiple Cypher statements are not allowed.")

    if not re.match(
        r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|RETURN)\b",
        cypher,
        flags=re.IGNORECASE
    ):
        raise ValueError(
            "Cypher must begin with MATCH, OPTIONAL MATCH, WITH, or RETURN."
        )

    if not re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE):
        raise ValueError("Generated Cypher must contain RETURN.")

    # Reject variable-length patterns such as [:OWNS*1..3]
    if re.search(r"\[[^\]]*\*", cypher):
        raise ValueError(
            "Variable-length relationship patterns are not allowed."
        )

    relationship_count = len(
        re.findall(r"-\s*\[", cypher)
    )

    if relationship_count > MAX_RELATIONSHIP_PATTERNS:
        raise ValueError(
            f"Query contains {relationship_count} relationship patterns; "
            f"maximum allowed is {MAX_RELATIONSHIP_PATTERNS}."
        )

    limit_match = re.search(
        r"\bLIMIT\s+(\d+)\b",
        cypher,
        flags=re.IGNORECASE
    )

    if limit_match:
        requested_limit = int(limit_match.group(1))

        if requested_limit > MAX_RESULT_ROWS:
            cypher = re.sub(
                r"\bLIMIT\s+\d+\b",
                f"LIMIT {MAX_RESULT_ROWS}",
                cypher,
                flags=re.IGNORECASE
            )
    else:
        cypher = f"{cypher}\nLIMIT {MAX_RESULT_ROWS}"

    return cypher.strip()


# ============================================================
# Neo4j execution
# ============================================================

def explain_query(cypher: str) -> None:
    """Compile the query without retrieving the full result."""
    neo4j_driver.execute_query(
        f"EXPLAIN\n{cypher}",
        database_=NEO4J_DATABASE,
        routing_="r"
    )


def execute_query(cypher: str) -> tuple[list[dict[str, Any]], float]:
    """Execute a validated read query and return plain dictionaries."""
    started = time.perf_counter()

    records, summary, keys = neo4j_driver.execute_query(
        cypher,
        database_=NEO4J_DATABASE,
        routing_="r"
    )

    latency_seconds = time.perf_counter() - started

    rows = [record.data() for record in records]

    return rows, latency_seconds


# ============================================================
# One automatic repair attempt
# ============================================================

def generate_valid_cypher(question: str):
    generation_prompt = build_cypher_prompt(question)

    raw_cypher, usage, generation_latency = call_llm(
        generation_prompt,
        max_output_tokens=1400
    )

    try:
        cypher = validate_read_only_cypher(
            clean_cypher(raw_cypher)
        )
        explain_query(cypher)

        return cypher, usage, generation_latency

    except (ValueError, Neo4jError) as first_error:
        repair_prompt = f"""
You generated an invalid Cypher query.

ORIGINAL QUESTION
{question}

GRAPH SCHEMA
{GRAPH_SCHEMA}

INVALID CYPHER
{raw_cypher}

ERROR
{str(first_error)}

Return one corrected read-only Cypher query only.
Do not include Markdown or explanation.
Use only the supplied graph schema.
Include LIMIT {MAX_RESULT_ROWS} or less.
""".strip()

        repaired_text, repair_usage, repair_latency = call_llm(
            repair_prompt,
            max_output_tokens=1400
        )

        repaired_cypher = validate_read_only_cypher(
            clean_cypher(repaired_text)
        )

        explain_query(repaired_cypher)

        combined_usage = {
            "input_tokens": (
                (usage.get("input_tokens") or 0)
                + (repair_usage.get("input_tokens") or 0)
            ),
            "output_tokens": (
                (usage.get("output_tokens") or 0)
                + (repair_usage.get("output_tokens") or 0)
            ),
            "total_tokens": (
                (usage.get("total_tokens") or 0)
                + (repair_usage.get("total_tokens") or 0)
            )
        }

        return (
            repaired_cypher,
            combined_usage,
            generation_latency + repair_latency
        )


# ============================================================
# Grounded answer generation
# ============================================================

def generate_grounded_answer(
    question: str,
    cypher: str,
    rows: list[dict[str, Any]]
):
    context_json = json.dumps(
        rows[:MAX_RESULT_ROWS],
        ensure_ascii=False,
        indent=2,
        default=str
    )

    answer_prompt = f"""
You are a careful financial-crime data assistant.

Answer the user's question using only the Neo4j records supplied below.

RULES

1. Do not invent facts.
2. When the records are empty, clearly state that no matching records
   were found in this 50,000-transaction synthetic sample.
3. Treat is_laundering=true only as synthetic IBM dataset ground truth.
4. Do not describe it as a real SAR, conviction, analyst decision,
   or model prediction.
5. Distinguish direct transaction relationships from inferred risk.
6. Mention important amounts, currencies, directions, counterparties,
   and transaction IDs when relevant.
7. Keep the answer concise and understandable.
8. Do not expose or discuss credentials.
9. Do not claim that this POC contains watchlist or adverse-media data.

USER QUESTION
{question}

EXECUTED CYPHER
{cypher}

NEO4J RECORDS
{context_json}

GROUNDED ANSWER
""".strip()

    return call_llm(
        answer_prompt,
        max_output_tokens=900
    )


# ============================================================
# Public assistant function
# ============================================================

def ask_financial_graph(question: str) -> dict[str, Any]:
    """
    Natural language question -> safe Cypher -> Neo4j evidence
    -> grounded natural-language answer.
    """
    total_started = time.perf_counter()

    cypher, cypher_usage, cypher_llm_latency = (
        generate_valid_cypher(question)
    )

    rows, neo4j_latency = execute_query(cypher)

    answer, answer_usage, answer_llm_latency = (
        generate_grounded_answer(
            question=question,
            cypher=cypher,
            rows=rows
        )
    )

    total_latency = time.perf_counter() - total_started

    return {
        "question": question,
        "answer": answer,
        "generated_cypher": cypher,
        "evidence_row_count": len(rows),
        "evidence": rows,
        "metrics": {
            "cypher_llm_seconds": round(cypher_llm_latency, 3),
            "neo4j_seconds": round(neo4j_latency, 3),
            "answer_llm_seconds": round(answer_llm_latency, 3),
            "total_seconds": round(total_latency, 3),
            "cypher_generation_usage": cypher_usage,
            "answer_generation_usage": answer_usage
        }
    }


print("✅ Financial Graph Text2Cypher assistant is ready.")

# COMMAND ----------

from openai import OpenAI

foundry_client = OpenAI(
    base_url=get_secret("foundry_base_url").rstrip("/") + "/",
    api_key=get_secret("foundry_api_key"),
    timeout=60.0,
    max_retries=0
)

# COMMAND ----------

import neo4j
from neo4j import GraphDatabase, Query, READ_ACCESS

print("Neo4j driver version:", neo4j.__version__)

SECRET_SCOPE = "azure_secrets"


def get_secret(name: str) -> str:
    value = (
        dbutils.secrets
        .get(scope=SECRET_SCOPE, key=name)
        .strip()
        .strip('"')
        .strip("'")
    )

    if not value:
        raise ValueError(f"Secret '{name}' is empty.")

    return value


neo4j_driver = GraphDatabase.driver(
    get_secret("neo4j_uri"),
    auth=(
        get_secret("neo4j_username"),
        get_secret("neo4j_password")
    ),
    connection_timeout=15.0
)

neo4j_driver.verify_connectivity()

with neo4j_driver.session(
    database="neo4j",
    default_access_mode=READ_ACCESS
) as session:
    result = session.run(
        Query(
            "RETURN 1 AS connection_test",
            timeout=10.0
        )
    )

    record = result.single()
    result.consume()

print("Neo4j connection test:", record["connection_test"])

# COMMAND ----------

import json
import time

from neo4j import Query, READ_ACCESS


optimized_cypher = """
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

UNION ALL

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
""".strip()


started = time.perf_counter()

with neo4j_driver.session(
    database="neo4j",
    default_access_mode=READ_ACCESS
) as session:

    result = session.run(
        Query(
            optimized_cypher,
            timeout=30.0
        ),
        source_name="Partnership #2370",
        target_name="Corporation #24457"
    )

    rows = [record.data() for record in result]
    result.consume()

elapsed = time.perf_counter() - started

print(f"Neo4j query completed in {elapsed:.2f} seconds")
print(f"Rows returned: {len(rows)}")
print(json.dumps(rows, indent=2, default=str))

# COMMAND ----------

import json
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from neo4j import GraphDatabase, Query, READ_ACCESS


# ============================================================
# Configuration
# ============================================================

SECRET_SCOPE = "azure_secrets"
NEO4J_DATABASE = "neo4j"


def get_secret(name: str) -> str:
    """Read and clean a required Databricks secret."""
    value = (
        dbutils.secrets
        .get(scope=SECRET_SCOPE, key=name)
        .strip()
        .strip('"')
        .strip("'")
    )

    if not value:
        raise ValueError(f"Secret '{name}' is empty.")

    return value


# ============================================================
# Connections
# ============================================================

foundry_client = OpenAI(
    base_url=get_secret("foundry_base_url").rstrip("/") + "/",
    api_key=get_secret("foundry_api_key"),
    timeout=60.0,
    max_retries=0
)

foundry_deployment = get_secret("foundry_deployment")

neo4j_driver = GraphDatabase.driver(
    get_secret("neo4j_uri"),
    auth=(
        get_secret("neo4j_username"),
        get_secret("neo4j_password")
    ),
    connection_timeout=15.0
)

neo4j_driver.verify_connectivity()


# ============================================================
# Structured LLM output
# ============================================================

class EntityRelationshipRequest(BaseModel):
    source_entity_name: str
    target_entity_name: str


def extract_relationship_entities(
    question: str
) -> EntityRelationshipRequest:
    """
    Extract the two entity names from a natural-language question.
    No Cypher is generated by the LLM.
    """

    completion = foundry_client.beta.chat.completions.parse(
        model=foundry_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the two financial entity names from the "
                    "question. Preserve names exactly as written. "
                    "The first entity is source_entity_name and the "
                    "second is target_entity_name."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format=EntityRelationshipRequest
    )

    parsed = completion.choices[0].message.parsed

    if parsed is None:
        raise ValueError(
            "The model could not extract two entity names."
        )

    return parsed

# COMMAND ----------

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
LIMIT 50
""".strip()

# COMMAND ----------

def run_read_query(
    cypher: str,
    parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    """Execute one trusted read-only query template."""

    with neo4j_driver.session(
        database=NEO4J_DATABASE,
        default_access_mode=READ_ACCESS
    ) as session:

        result = session.run(
            Query(
                cypher,
                timeout=30.0
            ),
            parameters
        )

        rows = [record.data() for record in result]
        result.consume()

    return rows


def generate_relationship_answer(
    question: str,
    source_name: str,
    target_name: str,
    rows: list[dict[str, Any]]
) -> str:
    """Generate a short answer grounded only in Neo4j evidence."""

    evidence_json = json.dumps(
        rows,
        indent=2,
        ensure_ascii=False,
        default=str
    )

    prompt = f"""
You are a careful financial-crime graph assistant.

Answer the question using only the Neo4j evidence below.

Important rules:
- Do not invent information.
- Explain whether transactions are source-to-target,
  target-to-source, or both.
- Mention amount, currency, payment format, and date when available.
- is_laundering=true is synthetic IBM AML dataset ground truth.
- Do not call it a real SAR, conviction, or model prediction.
- If there are no records, say that no direct transaction relationship
  was found in the 50,000-transaction synthetic sample.
- Keep the answer concise.

Question:
{question}

Source entity:
{source_name}

Target entity:
{target_name}

Neo4j evidence:
{evidence_json}
""".strip()

    response = foundry_client.responses.create(
        model=foundry_deployment,
        input=prompt,
        max_output_tokens=500
    )

    return response.output_text.strip()


def ask_entity_relationship(
    question: str
) -> dict[str, Any]:
    """
    Natural-language relationship question
    -> structured extraction
    -> optimized graph traversal
    -> grounded answer.
    """

    total_started = time.perf_counter()

    print("1. Extracting entity names...")
    extraction_started = time.perf_counter()

    request = extract_relationship_entities(question)

    extraction_seconds = (
        time.perf_counter() - extraction_started
    )

    print(
        "   Source:",
        request.source_entity_name
    )
    print(
        "   Target:",
        request.target_entity_name
    )

    parameters = {
        "source_name": request.source_entity_name,
        "target_name": request.target_entity_name
    }

    print("2. Querying Neo4j...")
    neo4j_started = time.perf_counter()

    source_to_target = run_read_query(
        SOURCE_TO_TARGET_QUERY,
        parameters
    )

    target_to_source = run_read_query(
        TARGET_TO_SOURCE_QUERY,
        parameters
    )

    rows = source_to_target + target_to_source

    rows.sort(
        key=lambda row: str(row.get("timestamp", ""))
    )

    neo4j_seconds = time.perf_counter() - neo4j_started

    print(f"   Evidence rows: {len(rows)}")

    print("3. Generating grounded answer...")
    answer_started = time.perf_counter()

    answer = generate_relationship_answer(
        question=question,
        source_name=request.source_entity_name,
        target_name=request.target_entity_name,
        rows=rows
    )

    answer_seconds = time.perf_counter() - answer_started
    total_seconds = time.perf_counter() - total_started

    return {
        "question": question,
        "source_entity": request.source_entity_name,
        "target_entity": request.target_entity_name,
        "answer": answer,
        "evidence": rows,
        "evidence_row_count": len(rows),
        "metrics": {
            "entity_extraction_seconds": round(
                extraction_seconds, 3
            ),
            "neo4j_seconds": round(
                neo4j_seconds, 3
            ),
            "answer_generation_seconds": round(
                answer_seconds, 3
            ),
            "total_seconds": round(
                total_seconds, 3
            )
        }
    }


print("✅ Relationship GraphRAG POC is ready.")

# COMMAND ----------

result = ask_entity_relationship(
    "What is the relationship between Partnership #2370 "
    "and Corporation #24457?"
)

print("\nANSWER")
print(result["answer"])

print("\nEVIDENCE")
print(
    json.dumps(
        result["evidence"],
        indent=2,
        default=str
    )
)

print("\nMETRICS")
print(
    json.dumps(
        result["metrics"],
        indent=2
    )
)

# COMMAND ----------

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AMLQuestionRoute(BaseModel):
    intent: Literal[
        "ENTITY_RELATIONSHIP",
        "ENTITY_LAUNDERING_TRANSACTIONS",
        "TOP_LAUNDERING_SENDERS",
        "UNSUPPORTED"
    ]

    source_entity_name: Optional[str] = None
    target_entity_name: Optional[str] = None
    entity_name: Optional[str] = None

    top_n: int = Field(default=5, ge=1, le=20)

    reason: str


def route_aml_question(question: str) -> AMLQuestionRoute:
    """
    Classify the question and extract only the parameters required
    by a trusted Cypher template.
    """

    completion = foundry_client.beta.chat.completions.parse(
        model=foundry_deployment,
        messages=[
            {
                "role": "system",
                "content": """
You route questions for a synthetic AML knowledge graph.

Supported intents:

1. ENTITY_RELATIONSHIP
Use when the user asks about direct transactions or the relationship
between two named entities.

Required:
- source_entity_name
- target_entity_name

2. ENTITY_LAUNDERING_TRANSACTIONS
Use when the user asks for laundering-labelled transactions connected
to one named entity.

Required:
- entity_name

3. TOP_LAUNDERING_SENDERS
Use when the user asks which entities sent the most
laundering-labelled transactions.

Required:
- top_n, default 5

4. UNSUPPORTED
Use for watchlists, adverse media, merchant-risk categories,
country risk, sanctions, alerts, documents, predictions, or other
information that is not currently represented in the graph.

Rules:
- Preserve entity names exactly as written.
- Do not invent missing entity names.
- Do not generate Cypher.
- is_laundering is synthetic dataset ground truth.
""".strip()
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format=AMLQuestionRoute
    )

    route = completion.choices[0].message.parsed

    if route is None:
        raise ValueError("The LLM could not route the question.")

    return route

# COMMAND ----------

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


INCOMING_LAUNDERING_QUERY = """
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

# COMMAND ----------

def generate_routed_answer(
    question: str,
    intent: str,
    evidence: list[dict]
) -> str:
    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        default=str
    )

    prompt = f"""
You are a careful financial-crime graph assistant.

Answer the question using only the supplied Neo4j evidence.

Rules:
- Do not invent information.
- Keep the answer concise.
- Explain transaction direction where relevant.
- Mention important amounts, currencies, dates, counterparties,
  and payment formats.
- is_laundering=true is synthetic IBM AML dataset ground truth.
- It is not a real SAR, conviction, analyst decision,
  or model prediction.
- If evidence is empty, state that no matching records were found
  in the 50,000-transaction synthetic sample.
- Do not claim that the graph contains sanctions, watchlists,
  merchant-risk categories, adverse media, or documents.

Intent:
{intent}

Question:
{question}

Neo4j evidence:
{evidence_json}
""".strip()

    response = foundry_client.responses.create(
        model=foundry_deployment,
        input=prompt,
        max_output_tokens=600
    )

    return response.output_text.strip()

# COMMAND ----------

def ask_aml_graph(question: str) -> dict:
    """
    Route a natural-language AML question to one of several
    trusted and optimized Cypher templates.
    """

    total_started = time.perf_counter()

    # --------------------------------------------------------
    # 1. Route the question
    # --------------------------------------------------------

    routing_started = time.perf_counter()
    route = route_aml_question(question)
    routing_seconds = time.perf_counter() - routing_started

    print("Intent:", route.intent)

    evidence = []

    # --------------------------------------------------------
    # 2. Execute the selected trusted template
    # --------------------------------------------------------

    neo4j_started = time.perf_counter()

    if route.intent == "ENTITY_RELATIONSHIP":

        if not route.source_entity_name or not route.target_entity_name:
            raise ValueError(
                "Two entity names are required for a relationship query."
            )

        parameters = {
            "source_name": route.source_entity_name,
            "target_name": route.target_entity_name
        }

        evidence = (
            run_read_query(SOURCE_TO_TARGET_QUERY, parameters)
            + run_read_query(TARGET_TO_SOURCE_QUERY, parameters)
        )

    elif route.intent == "ENTITY_LAUNDERING_TRANSACTIONS":

        if not route.entity_name:
            raise ValueError(
                "An entity name is required for this query."
            )

        parameters = {
            "entity_name": route.entity_name
        }

        evidence = (
            run_read_query(OUTGOING_LAUNDERING_QUERY, parameters)
            + run_read_query(INCOMING_LAUNDERING_QUERY, parameters)
        )

    elif route.intent == "TOP_LAUNDERING_SENDERS":

        top_n = min(max(route.top_n, 1), 20)

        evidence = run_read_query(
            TOP_LAUNDERING_SENDERS_QUERY,
            {"top_n": top_n}
        )

    elif route.intent == "UNSUPPORTED":

        answer = (
            "This proof of concept does not yet contain the data "
            "required to answer that question. It currently contains "
            "synthetic entities, accounts, banks, transactions, and "
            "the IBM synthetic laundering ground-truth label. "
            "Watchlists, adverse-media documents, merchant-risk "
            "categories, sanctions, and external risk information "
            "will be added in a later GraphRAG extension."
        )

        return {
            "question": question,
            "intent": route.intent,
            "answer": answer,
            "evidence": [],
            "evidence_row_count": 0,
            "route": route.model_dump(),
            "metrics": {
                "routing_seconds": round(routing_seconds, 3),
                "neo4j_seconds": 0.0,
                "answer_generation_seconds": 0.0,
                "total_seconds": round(
                    time.perf_counter() - total_started,
                    3
                )
            }
        }

    else:
        raise ValueError(f"Unknown intent: {route.intent}")

    neo4j_seconds = time.perf_counter() - neo4j_started

    evidence.sort(
        key=lambda row: str(row.get("timestamp", ""))
    )

    # --------------------------------------------------------
    # 3. Generate the grounded answer
    # --------------------------------------------------------

    answer_started = time.perf_counter()

    answer = generate_routed_answer(
        question=question,
        intent=route.intent,
        evidence=evidence
    )

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
            "answer_generation_seconds": round(
                answer_seconds,
                3
            ),
            "total_seconds": round(
                time.perf_counter() - total_started,
                3
            )
        }
    }


print("✅ Routed AML Graph Assistant is ready.")

# COMMAND ----------

poc_questions = [
    (
        "What is the relationship between Partnership #2370 "
        "and Corporation #24457?"
    ),
    (
        "Show laundering-labelled transactions connected to "
        "Partnership #2370."
    ),
    "Which five entities sent the most laundering-labelled transactions?",
    "Which customers are linked to high-risk merchants?"
]


for question in poc_questions:
    print("\n" + "=" * 100)
    print("QUESTION")
    print(question)

    try:
        result = ask_aml_graph(question)

        print("\nINTENT")
        print(result["intent"])

        print("\nANSWER")
        print(result["answer"])

        print("\nEVIDENCE ROWS")
        print(result["evidence_row_count"])

        print("\nMETRICS")
        print(
            json.dumps(
                result["metrics"],
                indent=2
            )
        )

    except Exception as error:
        print("\nERROR")
        print(type(error).__name__, str(error))

# COMMAND ----------

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
    sum(transaction_count_for_currency)
        AS laundering_transaction_count,
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

ORDER BY laundering_transaction_count DESC
LIMIT $top_n
""".strip()

# COMMAND ----------

result = ask_aml_graph(
    "Which five entities sent the most laundering-labelled transactions?"
)

print(result["answer"])
print(json.dumps(result["evidence"], indent=2, default=str))