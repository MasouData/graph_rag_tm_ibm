# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "kagglehub",
# ]
# ///
# MAGIC %pip install kagglehub

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use schema default;
# MAGIC CREATE VOLUME IF NOT EXISTS main.default.aml_dataset;

# COMMAND ----------

# -- 2.1 Load and Inspect the Dataset
import pandas as pd
import os
import glob
import shutil

# ------
volume_path = "/Volumes/main/default/aml_dataset"

# 2. Check if the Volume already contains files
if not os.path.exists(volume_path) or len(os.listdir(volume_path)) == 0:
    print("🔄 Volume is empty. Downloading dataset from Kaggle...")
    import kagglehub
    
    # Download locally to the driver
    download_path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    
    # Create the volume directory if it doesn't exist yet
    os.makedirs(volume_path, exist_ok=True)
    
    # Copy files from local driver to the permanent Volume
    for file_path in glob.glob(os.path.join(download_path, "*")):
        filename = os.path.basename(file_path)
        destination = os.path.join(volume_path, filename)
        
        # Use shutil to copy files (volumes are accessible as regular filesystem paths)
        shutil.copy2(file_path, destination)
    print("✅ Files successfully saved to Databricks Volume!")
else:
    print("😎 Files already exist in Volume. Skipping download.")

# 3. List the permanent files
files = [os.path.join(volume_path, f) for f in os.listdir(volume_path)]
print("\nAvailable files in Volume:")
for f in files:
    print(f"  - {os.path.basename(f)}")

# Look for files with "transaction" in the name
transaction_files = [f for f in files if 'transaction' in f.lower()]
if transaction_files:
    df_transactions = pd.read_csv(transaction_files[0])
    print(f"\n📊 Transactions shape: {df_transactions.shape}")
    print("\n📋 Columns:")
    print(df_transactions.columns.tolist())
    print("\n👀 First 5 rows:")
    print(df_transactions.head())
    print("\n🔍 Data types:")
    print(df_transactions.dtypes)
    print("\n📈 Summary statistics:")
    print(df_transactions.describe())

# COMMAND ----------

# Load all CSV files
all_data = {}
for file in files:
    if file.endswith('.csv') and "Small" in file:
        name = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        all_data[name] = df
        print(f"\n📁 {name}: {df.shape}")

# Let's look at what each file contains
for name, df in all_data.items():
    print(f"\n{'='*50}")
    print(f"📁 {name}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:10]}...")  # First 10 columns
    print(f"   Sample data:")
    print(df.head(2))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id

# Load the small transaction file
df_trans = spark.read.csv(
    "/Volumes/main/default/aml_dataset/HI-Small_Trans.csv", 
    header=True, 
    inferSchema=True
)

print(f"Total transactions: {df_trans.count()}")

# SAMPLE 50,000 rows for development 
df_trans_sample = df_trans.sample(fraction=0.01, seed=42).limit(50000)
print(f"Sampled transactions: {df_trans_sample.count()}")

# Show the schema and sample
df_trans_sample.printSchema()
df_trans_sample.show(5, truncate=False)

# COMMAND ----------

# Load account data (entity-to-account mapping)
df_accounts = spark.read.csv(
    "/Volumes/main/default/aml_dataset/HI-Small_accounts.csv", 
    header=True, 
    inferSchema=True
)

print(f"Total accounts: {df_accounts.count()}")
df_accounts.printSchema()
df_accounts.show(5, truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 1: Validate account identity and account metadata
# ============================================================

df_accounts_clean = (
    df_accounts
    .select(
        F.trim(F.col("Account Number")).alias("account_number"),
        F.col("Bank ID").cast("long").alias("bank_id"),
        F.trim(F.col("Bank Name")).alias("bank_name"),
        F.trim(F.col("Entity ID")).alias("entity_id"),
        F.trim(F.col("Entity Name")).alias("entity_name")
    )
    .filter(
        F.col("account_number").isNotNull()
        & (F.col("account_number") != "")
    )
)

# ------------------------------------------------------------
# 1. General account statistics
# ------------------------------------------------------------

account_summary = (
    df_accounts_clean
    .agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("account_number").alias(
            "distinct_account_numbers"
        ),
        F.countDistinct(
            "bank_id",
            "account_number"
        ).alias("distinct_bank_account_pairs"),
        F.countDistinct("entity_id").alias(
            "distinct_entities"
        ),
        F.countDistinct("bank_id").alias(
            "distinct_banks"
        )
    )
)

print("=== ACCOUNT SUMMARY ===")
account_summary.show(truncate=False)

# ------------------------------------------------------------
# 2. Check whether an account number occurs at multiple banks
#    or belongs to multiple entities
# ------------------------------------------------------------

account_identity_conflicts = (
    df_accounts_clean
    .groupBy("account_number")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("bank_id").alias("number_of_banks"),
        F.countDistinct("entity_id").alias("number_of_entities"),
        F.collect_set("bank_id").alias("bank_ids"),
        F.collect_set("entity_id").alias("entity_ids")
    )
    .filter(
        (F.col("number_of_banks") > 1)
        | (F.col("number_of_entities") > 1)
    )
)

conflict_count = account_identity_conflicts.count()

print(
    "Account numbers connected to multiple banks "
    f"or entities: {conflict_count}"
)

account_identity_conflicts.show(20, truncate=False)

# ------------------------------------------------------------
# 3. Check for duplicate metadata rows
# ------------------------------------------------------------

duplicate_metadata_rows = (
    df_accounts_clean
    .groupBy(
        "bank_id",
        "account_number",
        "entity_id"
    )
    .count()
    .filter(F.col("count") > 1)
)

duplicate_metadata_count = duplicate_metadata_rows.count()

print(
    "Duplicate bank-account-entity records: "
    f"{duplicate_metadata_count}"
)

duplicate_metadata_rows.show(20, truncate=False)

# ------------------------------------------------------------
# 4. Check missing values
# ------------------------------------------------------------

missing_value_summary = (
    df_accounts_clean
    .agg(
        F.sum(
            F.when(F.col("bank_id").isNull(), 1).otherwise(0)
        ).alias("missing_bank_id"),
        
        F.sum(
            F.when(
                F.col("bank_name").isNull()
                | (F.col("bank_name") == ""),
                1
            ).otherwise(0)
        ).alias("missing_bank_name"),
        
        F.sum(
            F.when(
                F.col("entity_id").isNull()
                | (F.col("entity_id") == ""),
                1
            ).otherwise(0)
        ).alias("missing_entity_id"),
        
        F.sum(
            F.when(
                F.col("entity_name").isNull()
                | (F.col("entity_name") == ""),
                1
            ).otherwise(0)
        ).alias("missing_entity_name")
    )
)

print("=== MISSING-VALUE SUMMARY ===")
missing_value_summary.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### creates the stable key by joining the bank ID and account number with the | separator.

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 2: Build Account nodes using bank + account number
# ============================================================

# ------------------------------------------------------------
# 1. Extract sender accounts with their bank IDs
# ------------------------------------------------------------

sender_accounts = (
    df_trans_sample
    .select(
        F.col("From Bank").cast("long").alias("bank_id"),
        F.trim(F.col("Account2")).alias("account_number")
    )
)

# ------------------------------------------------------------
# 2. Extract receiver accounts with their bank IDs
# ------------------------------------------------------------

receiver_accounts = (
    df_trans_sample
    .select(
        F.col("To Bank").cast("long").alias("bank_id"),
        F.trim(F.col("Account4")).alias("account_number")
    )
)

# ------------------------------------------------------------
# 3. Combine both transaction sides
# ------------------------------------------------------------

sample_bank_accounts = (
    sender_accounts
    .unionByName(receiver_accounts)
    .filter(
        F.col("bank_id").isNotNull()
        & F.col("account_number").isNotNull()
        & (F.col("account_number") != "")
    )
    .dropDuplicates(["bank_id", "account_number"])
)

print("=== SAMPLE BANK-ACCOUNT PAIRS ===")

sample_bank_accounts.show(10, truncate=False)

sample_pair_count = sample_bank_accounts.count()

print(
    f"Distinct bank-account pairs in transaction sample: "
    f"{sample_pair_count:,}"
)

# ------------------------------------------------------------
# 4. Prepare unique account metadata
# ------------------------------------------------------------

account_metadata = (
    df_accounts_clean
    .dropDuplicates(["bank_id", "account_number"])
    .select(
        "bank_id",
        "account_number",
        "bank_name",
        "entity_id",
        "entity_name"
    )
)

# ------------------------------------------------------------
# 5. Join using BOTH bank_id and account_number
# ------------------------------------------------------------

df_account_nodes = (
    sample_bank_accounts.alias("sample")
    .join(
        account_metadata.alias("metadata"),
        on=["bank_id", "account_number"],
        how="left"
    )
    .withColumn(
        "account_key",
        F.concat_ws(
            "|",
            F.col("bank_id").cast("string"),
            F.col("account_number")
        )
    )
    .select(
        "account_key",
        "account_number",
        "bank_id",
        "bank_name",
        "entity_id",
        "entity_name"
    )
    .withColumn("label", F.lit("Account"))
)

print("\n=== CORRECTED ACCOUNT NODES ===")

df_account_nodes.show(10, truncate=False)

# ------------------------------------------------------------
# 6. Validate node count and uniqueness
# ------------------------------------------------------------

account_node_summary = (
    df_account_nodes
    .agg(
        F.count("*").alias("account_node_rows"),
        F.countDistinct("account_key").alias(
            "distinct_account_keys"
        ),
        F.countDistinct(
            "bank_id",
            "account_number"
        ).alias("distinct_bank_account_pairs"),
        F.countDistinct("account_number").alias(
            "distinct_account_numbers"
        ),
        F.countDistinct("entity_id").alias(
            "connected_entities"
        ),
        F.countDistinct("bank_id").alias(
            "connected_banks"
        )
    )
)

print("\n=== ACCOUNT NODE SUMMARY ===")

account_node_summary.show(truncate=False)

# ------------------------------------------------------------
# 7. Check whether metadata was found for every sampled account
# ------------------------------------------------------------

accounts_without_metadata = (
    df_account_nodes
    .filter(
        F.col("bank_name").isNull()
        | F.col("entity_id").isNull()
        | F.col("entity_name").isNull()
    )
)

accounts_without_metadata_count = accounts_without_metadata.count()

print(
    "\nAccounts without complete metadata: "
    f"{accounts_without_metadata_count:,}"
)

accounts_without_metadata.show(20, truncate=False)

# ------------------------------------------------------------
# 8. Check for duplicate account keys
# ------------------------------------------------------------

duplicate_account_keys = (
    df_account_nodes
    .groupBy("account_key")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_account_key_count = duplicate_account_keys.count()

print(
    "\nDuplicate account keys: "
    f"{duplicate_account_key_count:,}"
)

duplicate_account_keys.show(20, truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 3: Build relevant Bank and Entity nodes
# ============================================================

# ------------------------------------------------------------
# 1. Validate that every bank ID has one consistent bank name
# ------------------------------------------------------------

bank_name_conflicts = (
    df_account_nodes
    .groupBy("bank_id")
    .agg(
        F.countDistinct("bank_name").alias("number_of_names"),
        F.collect_set("bank_name").alias("bank_names")
    )
    .filter(F.col("number_of_names") > 1)
)

bank_name_conflict_count = bank_name_conflicts.count()

print("=== BANK NAME VALIDATION ===")
print(
    f"Bank IDs associated with multiple names: "
    f"{bank_name_conflict_count:,}"
)

bank_name_conflicts.show(20, truncate=False)


# ------------------------------------------------------------
# 2. Validate that every entity ID has one consistent name
# ------------------------------------------------------------

entity_name_conflicts = (
    df_account_nodes
    .groupBy("entity_id")
    .agg(
        F.countDistinct("entity_name").alias("number_of_names"),
        F.collect_set("entity_name").alias("entity_names")
    )
    .filter(F.col("number_of_names") > 1)
)

entity_name_conflict_count = entity_name_conflicts.count()

print("\n=== ENTITY NAME VALIDATION ===")
print(
    f"Entity IDs associated with multiple names: "
    f"{entity_name_conflict_count:,}"
)

entity_name_conflicts.show(20, truncate=False)


# ------------------------------------------------------------
# 3. Stop if conflicting metadata exists
# ------------------------------------------------------------

if bank_name_conflict_count > 0:
    raise ValueError(
        "Some bank IDs have multiple bank names. "
        "Resolve these conflicts before constructing Bank nodes."
    )

if entity_name_conflict_count > 0:
    raise ValueError(
        "Some entity IDs have multiple entity names. "
        "Resolve these conflicts before constructing Entity nodes."
    )


# ------------------------------------------------------------
# 4. Construct relevant Bank nodes
# ------------------------------------------------------------

df_bank_nodes = (
    df_account_nodes
    .select(
        F.col("bank_id"),
        F.col("bank_name").alias("name")
    )
    .dropDuplicates(["bank_id"])
    .withColumn("label", F.lit("Bank"))
)

print("\n=== BANK NODES ===")

df_bank_nodes.show(10, truncate=False)


# ------------------------------------------------------------
# 5. Construct relevant Entity nodes
# ------------------------------------------------------------

df_entity_nodes = (
    df_account_nodes
    .select(
        F.col("entity_id").alias("id"),
        F.col("entity_name").alias("name")
    )
    .dropDuplicates(["id"])
    .withColumn("label", F.lit("Entity"))
)

print("\n=== ENTITY NODES ===")

df_entity_nodes.show(10, truncate=False)


# ------------------------------------------------------------
# 6. Validate Bank node uniqueness
# ------------------------------------------------------------

bank_node_summary = (
    df_bank_nodes
    .agg(
        F.count("*").alias("bank_node_rows"),
        F.countDistinct("bank_id").alias("distinct_bank_ids"),
        F.sum(
            F.when(
                F.col("name").isNull() |
                (F.trim(F.col("name")) == ""),
                1
            ).otherwise(0)
        ).alias("missing_bank_names")
    )
)

print("\n=== BANK NODE SUMMARY ===")

bank_node_summary.show(truncate=False)


# ------------------------------------------------------------
# 7. Validate Entity node uniqueness
# ------------------------------------------------------------

entity_node_summary = (
    df_entity_nodes
    .agg(
        F.count("*").alias("entity_node_rows"),
        F.countDistinct("id").alias("distinct_entity_ids"),
        F.sum(
            F.when(
                F.col("name").isNull() |
                (F.trim(F.col("name")) == ""),
                1
            ).otherwise(0)
        ).alias("missing_entity_names")
    )
)

print("\n=== ENTITY NODE SUMMARY ===")

entity_node_summary.show(truncate=False)


# ------------------------------------------------------------
# 8. Compare node counts against Account metadata
# ------------------------------------------------------------

expected_counts = (
    df_account_nodes
    .agg(
        F.countDistinct("bank_id").alias("expected_banks"),
        F.countDistinct("entity_id").alias("expected_entities")
    )
)

print("\n=== EXPECTED COUNTS FROM ACCOUNT NODES ===")

expected_counts.show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 4: Build and validate Transaction nodes   
# (Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)
# ============================================================

# ------------------------------------------------------------
# 1. Prepare transaction data and composite account keys
# ------------------------------------------------------------

df_transaction_base = (
    df_trans_sample
    .select(
        F.trim(F.col("Timestamp")).alias("timestamp_raw"),
        F.col("From Bank").cast("long").alias("from_bank_id"),
        F.trim(F.col("Account2")).alias("from_account_number"),
        F.col("To Bank").cast("long").alias("to_bank_id"),
        F.trim(F.col("Account4")).alias("to_account_number"),
        F.col("Amount Received").cast("double").alias("amount_received"),
        F.trim(F.col("Receiving Currency")).alias("receiving_currency"),
        F.col("Amount Paid").cast("double").alias("amount_paid"),
        F.trim(F.col("Payment Currency")).alias("payment_currency"),
        F.trim(F.col("Payment Format")).alias("payment_format"),
        F.col("Is Laundering").cast("integer").alias("is_laundering_raw")
    )
    .withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("timestamp_raw"),
            "yyyy/MM/dd HH:mm"
        )
    )
    .withColumn(
        "from_account_key",
        F.concat_ws(
            "|",
            F.col("from_bank_id").cast("string"),
            F.col("from_account_number")
        )
    )
    .withColumn(
        "to_account_key",
        F.concat_ws(
            "|",
            F.col("to_bank_id").cast("string"),
            F.col("to_account_number")
        )
    )
    .withColumn(
        "is_laundering",
        F.col("is_laundering_raw") == 1
    )
)

# COMMAND ----------

# ------------------------------------------------------------
# 2. Create a deterministic transaction ID
# ------------------------------------------------------------

transaction_hash_input = F.concat_ws(
    "||",
    F.coalesce(F.col("timestamp_raw"), F.lit("<NULL>")),
    F.coalesce(F.col("from_bank_id").cast("string"), F.lit("<NULL>")),
    F.coalesce(F.col("from_account_number"), F.lit("<NULL>")),
    F.coalesce(F.col("to_bank_id").cast("string"), F.lit("<NULL>")),
    F.coalesce(F.col("to_account_number"), F.lit("<NULL>")),
    F.coalesce(F.col("amount_received").cast("string"), F.lit("<NULL>")),
    F.coalesce(F.col("receiving_currency"), F.lit("<NULL>")),
    F.coalesce(F.col("amount_paid").cast("string"), F.lit("<NULL>")),
    F.coalesce(F.col("payment_currency"), F.lit("<NULL>")),
    F.coalesce(F.col("payment_format"), F.lit("<NULL>")),
    F.coalesce(F.col("is_laundering_raw").cast("string"), F.lit("<NULL>"))
)

df_transaction_nodes = (
    df_transaction_base
    .withColumn(
        "transaction_id",
        F.sha2(transaction_hash_input, 256)
    )
    .select(
        "transaction_id",
        "timestamp",
        "from_account_key",
        "to_account_key",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering"
    )
    .withColumn("label", F.lit("Transaction"))
)

print("=== TRANSACTION NODES ===")
df_transaction_nodes.show(10, truncate=False)

# COMMAND ----------

# ------------------------------------------------------------
# 3. Transaction summary
# ------------------------------------------------------------

transaction_summary = (
    df_transaction_nodes
    .agg(
        F.count("*").alias("transaction_node_rows"),
        F.countDistinct("transaction_id").alias(
            "distinct_transaction_ids"
        ),
        F.sum(
            F.when(F.col("timestamp").isNull(), 1).otherwise(0)
        ).alias("invalid_timestamps"),
        F.sum(
            F.when(
                F.col("from_account_key") == F.col("to_account_key"),
                1
            ).otherwise(0)
        ).alias("self_transfers"),
        F.sum(
            F.when(F.col("is_laundering"), 1).otherwise(0)
        ).alias("laundering_transactions")
    )
)

print("\n=== TRANSACTION NODE SUMMARY ===")
transaction_summary.show(truncate=False)


# ------------------------------------------------------------
# 4. Check for duplicate generated transaction IDs
# ------------------------------------------------------------

duplicate_transaction_ids = (
    df_transaction_nodes
    .groupBy("transaction_id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_transaction_id_count = duplicate_transaction_ids.count()

print(
    "Duplicate transaction IDs: "
    f"{duplicate_transaction_id_count:,}"
)

duplicate_transaction_ids.show(20, truncate=False)


# ------------------------------------------------------------
# 5. Confirm that every transaction account exists
# ------------------------------------------------------------

transaction_account_keys = (
    df_transaction_nodes
    .select(F.col("from_account_key").alias("account_key"))
    .unionByName(
        df_transaction_nodes.select(
            F.col("to_account_key").alias("account_key")
        )
    )
    .distinct()
)

unknown_account_keys = (
    transaction_account_keys
    .join(
        df_account_nodes.select("account_key"),
        on="account_key",
        how="left_anti"
    )
)

unknown_account_key_count = unknown_account_keys.count()

print(
    "Transaction account keys missing from Account nodes: "
    f"{unknown_account_key_count:,}"
)

unknown_account_keys.show(20, truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 5: Build graph relationships

# (Entity)-[:OWNS]->(Account)
# (Account)-[:HELD_AT]->(Bank)
# (Account)-[:SENT]->(Transaction)
# (Transaction)-[:RECEIVED_BY]->(Account)
# ============================================================

# ------------------------------------------------------------
# 1. Entity -> Account
# ------------------------------------------------------------

df_owns_relationships = (
    df_account_nodes
    .select(
        F.col("entity_id"),
        F.col("account_key")
    )
    .dropDuplicates(["entity_id", "account_key"])
)

print("=== OWNS RELATIONSHIPS ===")
df_owns_relationships.show(10, truncate=False)

# ------------------------------------------------------------
# 2. Account -> Bank
# ------------------------------------------------------------

df_held_at_relationships = (
    df_account_nodes
    .select(
        F.col("account_key"),
        F.col("bank_id")
    )
    .dropDuplicates(["account_key", "bank_id"])
)

print("\n=== HELD_AT RELATIONSHIPS ===")
df_held_at_relationships.show(10, truncate=False)

# ------------------------------------------------------------
# 3. Account -> Transaction
# ------------------------------------------------------------

df_sent_relationships = (
    df_transaction_nodes
    .select(
        F.col("from_account_key").alias("account_key"),
        F.col("transaction_id")
    )
)

print("\n=== SENT RELATIONSHIPS ===")
df_sent_relationships.show(10, truncate=False)

# ------------------------------------------------------------
# 4. Transaction -> Account
# ------------------------------------------------------------

df_received_by_relationships = (
    df_transaction_nodes
    .select(
        F.col("transaction_id"),
        F.col("to_account_key").alias("account_key")
    )
)

print("\n=== RECEIVED_BY RELATIONSHIPS ===")
df_received_by_relationships.show(10, truncate=False)

# COMMAND ----------

# ============================================================
# Validate relationship counts and endpoints
# ============================================================

relationship_summary = spark.createDataFrame(
    [
        (
            "OWNS",
            df_owns_relationships.count(),
            df_owns_relationships
                .select("entity_id", "account_key")
                .distinct()
                .count()
        ),
        (
            "HELD_AT",
            df_held_at_relationships.count(),
            df_held_at_relationships
                .select("account_key", "bank_id")
                .distinct()
                .count()
        ),
        (
            "SENT",
            df_sent_relationships.count(),
            df_sent_relationships
                .select("account_key", "transaction_id")
                .distinct()
                .count()
        ),
        (
            "RECEIVED_BY",
            df_received_by_relationships.count(),
            df_received_by_relationships
                .select("transaction_id", "account_key")
                .distinct()
                .count()
        )
    ],
    [
        "relationship_type",
        "row_count",
        "distinct_relationships"
    ]
)

print("=== RELATIONSHIP SUMMARY ===")
relationship_summary.show(truncate=False)

# COMMAND ----------

# ------------------------------------------------------------
# OWNS endpoint checks
# ------------------------------------------------------------

owns_missing_entities = (
    df_owns_relationships
    .join(
        df_entity_nodes.select(
            F.col("id").alias("entity_id")
        ),
        on="entity_id",
        how="left_anti"
    )
    .count()
)

owns_missing_accounts = (
    df_owns_relationships
    .join(
        df_account_nodes.select("account_key"),
        on="account_key",
        how="left_anti"
    )
    .count()
)

# ------------------------------------------------------------
# HELD_AT endpoint checks
# ------------------------------------------------------------

held_at_missing_accounts = (
    df_held_at_relationships
    .join(
        df_account_nodes.select("account_key"),
        on="account_key",
        how="left_anti"
    )
    .count()
)

held_at_missing_banks = (
    df_held_at_relationships
    .join(
        df_bank_nodes.select("bank_id"),
        on="bank_id",
        how="left_anti"
    )
    .count()
)

# ------------------------------------------------------------
# SENT endpoint checks
# ------------------------------------------------------------

sent_missing_accounts = (
    df_sent_relationships
    .join(
        df_account_nodes.select("account_key"),
        on="account_key",
        how="left_anti"
    )
    .count()
)

sent_missing_transactions = (
    df_sent_relationships
    .join(
        df_transaction_nodes.select("transaction_id"),
        on="transaction_id",
        how="left_anti"
    )
    .count()
)

# ------------------------------------------------------------
# RECEIVED_BY endpoint checks
# ------------------------------------------------------------

received_missing_transactions = (
    df_received_by_relationships
    .join(
        df_transaction_nodes.select("transaction_id"),
        on="transaction_id",
        how="left_anti"
    )
    .count()
)

received_missing_accounts = (
    df_received_by_relationships
    .join(
        df_account_nodes.select("account_key"),
        on="account_key",
        how="left_anti"
    )
    .count()
)

print("=== MISSING RELATIONSHIP ENDPOINTS ===")
print(f"OWNS missing entities:          {owns_missing_entities}")
print(f"OWNS missing accounts:          {owns_missing_accounts}")
print(f"HELD_AT missing accounts:       {held_at_missing_accounts}")
print(f"HELD_AT missing banks:          {held_at_missing_banks}")
print(f"SENT missing accounts:          {sent_missing_accounts}")
print(f"SENT missing transactions:      {sent_missing_transactions}")
print(f"RECEIVED_BY missing transactions: {received_missing_transactions}")
print(f"RECEIVED_BY missing accounts:   {received_missing_accounts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Export the corrected graph CSV files

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# STEP 6A: Prepare final Neo4j CSV exports
# ============================================================

# Account nodes
df_accounts_export = (
    df_account_nodes
    .select(
        "account_key",
        "account_number"
    )
    .orderBy("account_key")
)

# Bank nodes
df_banks_export = (
    df_bank_nodes
    .select(
        F.col("bank_id").cast("long").alias("bank_id"),
        "name"
    )
    .orderBy("bank_id")
)

# Entity nodes
df_entities_export = (
    df_entity_nodes
    .select(
        "id",
        "name"
    )
    .orderBy("id")
)

# Transaction nodes
df_transactions_export = (
    df_transaction_nodes
    .select(
        "transaction_id",

        # Export as an ISO-like local datetime string
        F.date_format(
            F.col("timestamp"),
            "yyyy-MM-dd'T'HH:mm:ss"
        ).alias("timestamp"),

        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        F.col("is_laundering").cast("boolean").alias("is_laundering")
    )
    .orderBy("transaction_id")
)

# Entity -> Account
df_owns_export = (
    df_owns_relationships
    .select(
        "entity_id",
        "account_key"
    )
    .orderBy("entity_id", "account_key")
)

# Account -> Bank
df_held_at_export = (
    df_held_at_relationships
    .select(
        "account_key",
        F.col("bank_id").cast("long").alias("bank_id")
    )
    .orderBy("account_key")
)

# Account -> Transaction
df_sent_export = (
    df_sent_relationships
    .select(
        "account_key",
        "transaction_id"
    )
    .orderBy("transaction_id")
)

# Transaction -> Account
df_received_by_export = (
    df_received_by_relationships
    .select(
        "transaction_id",
        "account_key"
    )
    .orderBy("transaction_id")
)

print("✅ Final export DataFrames prepared.")

# COMMAND ----------

# ============================================================
# STEP 6B: Export each DataFrame as one CSV file
# ============================================================

output_dir = (
    "/Volumes/main/default/aml_dataset/"
    "graph_rag_data_v2"
)

# Start with a clean output directory.
# This does NOT delete your original IBM dataset.
dbutils.fs.rm(output_dir, recurse=True)
dbutils.fs.mkdirs(output_dir)


def write_single_csv(df, filename):
    """
    Write a Spark DataFrame as one header-containing CSV file
    inside the Unity Catalog Volume.

    Intended for the current development-sized graph export.
    """

    base_name = filename.replace(".csv", "")
    temporary_dir = f"{output_dir}/_{base_name}_temporary"
    final_path = f"{output_dir}/{filename}"

    # Remove leftovers from an earlier execution
    dbutils.fs.rm(temporary_dir, recurse=True)
    dbutils.fs.rm(final_path, recurse=True)

    (
        df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("emptyValue", "")
        .option("nullValue", "")
        .csv(temporary_dir)
    )

    generated_csv_files = [
        file_info
        for file_info in dbutils.fs.ls(temporary_dir)
        if file_info.name.startswith("part-")
        and file_info.name.endswith(".csv")
    ]

    if len(generated_csv_files) != 1:
        raise RuntimeError(
            f"Expected exactly one CSV part for {filename}, "
            f"but found {len(generated_csv_files)}."
        )

    # Move and rename the generated part file
    dbutils.fs.mv(
        generated_csv_files[0].path,
        final_path
    )

    # Remove _SUCCESS and temporary directory
    dbutils.fs.rm(temporary_dir, recurse=True)

    print(f"✅ Created: {final_path}")


exports = {
    "accounts.csv": df_accounts_export,
    "banks.csv": df_banks_export,
    "entities.csv": df_entities_export,
    "transactions.csv": df_transactions_export,
    "owns.csv": df_owns_export,
    "held_at.csv": df_held_at_export,
    "sent.csv": df_sent_export,
    "received_by.csv": df_received_by_export
}

for filename, dataframe in exports.items():
    write_single_csv(dataframe, filename)

# COMMAND ----------

# ============================================================
# STEP 6C: Read back and validate exported CSV files
# ============================================================

expected_row_counts = {
    "accounts.csv": 71267,
    "banks.csv": 3639,
    "entities.csv": 56514,
    "transactions.csv": 50000,
    "owns.csv": 71267,
    "held_at.csv": 71267,
    "sent.csv": 50000,
    "received_by.csv": 50000
}

validation_results = []

for filename, expected_count in expected_row_counts.items():
    file_path = f"{output_dir}/{filename}"

    exported_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(file_path)
    )

    actual_count = exported_df.count()

    validation_results.append(
        (
            filename,
            expected_count,
            actual_count,
            actual_count == expected_count,
            ",".join(exported_df.columns)
        )
    )

df_export_validation = spark.createDataFrame(
    validation_results,
    [
        "filename",
        "expected_rows",
        "actual_rows",
        "count_matches",
        "columns"
    ]
)

print("=== CSV EXPORT VALIDATION ===")

df_export_validation.show(
    truncate=False
)

# COMMAND ----------

print("=== FINAL GRAPH EXPORT FILES ===")

for file_info in dbutils.fs.ls(output_dir):
    print(
        f"{file_info.name:<24} "
        f"{file_info.size:>12,} bytes"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export to Azure Blob Storage
# MAGIC
# MAGIC ### Step 1: Create Azure Storage Account (Free Tier)
# MAGIC
# MAGIC 1. **Sign in to Azure Portal**: Go to [portal.azure.com](https://portal.azure.com)
# MAGIC 2. **Create Storage Account**:
# MAGIC    - Click "Create a resource"
# MAGIC    - Search for "Storage account"
# MAGIC    - Click "Create"
# MAGIC 3. **Configure**:
# MAGIC    - **Subscription**: Select your free subscription
# MAGIC    - **Resource group**: Create new or use existing
# MAGIC    - **Storage account name**: Choose a unique name (e.g., `amlgraphdata123`)
# MAGIC    - **Region**: Choose closest to you
# MAGIC    - **Performance**: Standard
# MAGIC    - **Redundancy**: LRS (Locally-redundant storage) — cheapest option
# MAGIC 4. **Review + Create**: Click and wait for deployment
# MAGIC
# MAGIC ### Step 2: Get Connection String
# MAGIC
# MAGIC 1. Go to your new Storage Account
# MAGIC 2. In left menu: **Security + networking** → **Access keys**
# MAGIC 3. Click **Show** next to "Connection string" under key1
# MAGIC 4. **Copy** the entire connection string
# MAGIC 5. Paste it in the cell below replacing `YOUR_CONNECTION_STRING_HERE`
# MAGIC
# MAGIC ### Step 3: Run the cells below
# MAGIC
# MAGIC - Cell 15: Installs Azure SDK
# MAGIC - Cell 16: Uploads your 4 CSV files to Azure Blob
# MAGIC
# MAGIC ### Free Tier Limits
# MAGIC - 5 GB of locally redundant storage
# MAGIC - 20,000 read operations
# MAGIC - 10,000 write operations per month
# MAGIC

# COMMAND ----------

# DBTITLE 1,Export to Azure Blob Storage
# MAGIC %pip install azure-storage-blob

# COMMAND ----------

# DBTITLE 1,Store Azure credentials securely
# MAGIC %md
# MAGIC ## Securely Store Your Azure Connection String
# MAGIC
# MAGIC ### Option 1: Using Databricks Secrets (Recommended)
# MAGIC
# MAGIC **Step 1: Create a secret scope** (one-time setup)
# MAGIC ```bash
# MAGIC # Run this in your terminal (requires Databricks CLI)
# MAGIC databricks secrets create-scope --scope azure_secrets
# MAGIC ```
# MAGIC
# MAGIC **Step 2: Store your connection string**
# MAGIC ```bash
# MAGIC # This stores your connection string securely
# MAGIC databricks secrets put --scope azure_secrets --key storage_connection_string
# MAGIC # An editor will open - paste your connection string, save, and close
# MAGIC ```
# MAGIC
# MAGIC **Alternative: Use the UI**
# MAGIC 1. Go to: `https://<your-workspace>.cloud.databricks.com/#secrets/createScope`
# MAGIC 2. Create scope: `azure_secrets`
# MAGIC 3. Add secret: `storage_connection_string` with your connection string value
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Option 2: Using Environment Variables (Simpler for Testing)
# MAGIC
# MAGIC Set it in a separate cell that you **don't commit** to Git:
# MAGIC ```python
# MAGIC import os
# MAGIC os.environ['AZURE_CONNECTION_STRING'] = "your-actual-connection-string-here"
# MAGIC ```
# MAGIC
# MAGIC Then add this cell to your `.gitignore`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The cell below uses **Option 1 (Databricks Secrets)** - the most secure approach.

# COMMAND ----------

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings
)
from azure.core.exceptions import ResourceExistsError
import os

# ============================================================
# STEP 7: Upload corrected graph CSVs to Azure Blob Storage
# ============================================================

source_dir = (
    "/Volumes/main/default/aml_dataset/"
    "graph_rag_data_v2"
)

container_name = "aml-graph-data"

# Use a prefix so the old and new files remain separate.
blob_prefix = "v2"

expected_files = [
    "accounts.csv",
    "banks.csv",
    "entities.csv",
    "transactions.csv",
    "owns.csv",
    "held_at.csv",
    "sent.csv",
    "received_by.csv"
]

# ------------------------------------------------------------
# 1. Retrieve the Azure connection string securely
# ------------------------------------------------------------

connection_string = dbutils.secrets.get(
    scope="azure_secrets",
    key="storage_connection_string"
)

# Never print the connection string.

# ------------------------------------------------------------
# 2. Connect to Azure Blob Storage
# ------------------------------------------------------------

blob_service_client = (
    BlobServiceClient.from_connection_string(
        connection_string
    )
)

# Create the container only if it does not already exist.
try:
    container_client = (
        blob_service_client.create_container(
            container_name
        )
    )

    print(f"Created private container: {container_name}")

except ResourceExistsError:
    container_client = (
        blob_service_client.get_container_client(
            container_name
        )
    )

    print(f"Using existing container: {container_name}")

# ------------------------------------------------------------
# 3. Upload the eight corrected files
# ------------------------------------------------------------

upload_results = []

for filename in expected_files:

    local_path = os.path.join(source_dir, filename)

    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Required export file not found: {local_path}"
        )

    blob_name = f"{blob_prefix}/{filename}"

    local_size = os.path.getsize(local_path)

    blob_client = container_client.get_blob_client(
        blob=blob_name
    )

    with open(local_path, "rb") as file_data:
        blob_client.upload_blob(
            file_data,
            overwrite=True,
            content_settings=ContentSettings(
                content_type="text/csv; charset=utf-8"
            ),
            metadata={
                "project": "financial-watchlist-graphrag",
                "dataset": "ibm-aml-synthetic",
                "version": "v2"
            }
        )

    blob_properties = blob_client.get_blob_properties()

    upload_results.append(
        (
            filename,
            blob_name,
            local_size,
            blob_properties.size,
            blob_properties.content_settings.content_type,
            local_size == blob_properties.size
        )
    )

    print(
        f"Uploaded {blob_name}: "
        f"{blob_properties.size:,} bytes"
    )

print("\nAll files uploaded.")

# COMMAND ----------

# ============================================================
# Display upload validation
# ============================================================

df_upload_validation = spark.createDataFrame(
    upload_results,
    [
        "filename",
        "blob_name",
        "local_bytes",
        "azure_bytes",
        "content_type",
        "size_matches"
    ]
)

print("=== AZURE UPLOAD VALIDATION ===")

df_upload_validation.show(
    truncate=False
)

# COMMAND ----------

# ============================================================
# List and validate blobs under the v2 prefix
# ============================================================

uploaded_blobs = list(
    container_client.list_blobs(
        name_starts_with=f"{blob_prefix}/"
    )
)

print("=== BLOBS UNDER v2/ ===")

for blob in uploaded_blobs:
    print(
        f"{blob.name:<32} "
        f"{blob.size:>12,} bytes"
    )

print(
    f"\nNumber of blobs under {blob_prefix}/: "
    f"{len(uploaded_blobs)}"
)

if len(uploaded_blobs) != len(expected_files):
    raise ValueError(
        f"Expected {len(expected_files)} blobs under "
        f"{blob_prefix}/ but found {len(uploaded_blobs)}."
    )

expected_blob_names = {
    f"{blob_prefix}/{filename}"
    for filename in expected_files
}

actual_blob_names = {
    blob.name
    for blob in uploaded_blobs
}

missing_blobs = expected_blob_names - actual_blob_names
unexpected_blobs = actual_blob_names - expected_blob_names

print(f"Missing blobs: {sorted(missing_blobs)}")
print(f"Unexpected blobs: {sorted(unexpected_blobs)}")

if missing_blobs or unexpected_blobs:
    raise ValueError(
        "The Azure v2 blob set does not match "
        "the expected graph files."
    )

print("\nAzure upload validation passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Generate temporary read-only SAS URLs

# COMMAND ----------

from azure.storage.blob import (
    BlobSasPermissions,
    generate_blob_sas
)
from datetime import datetime, timedelta, timezone
import requests

# ============================================================
# STEP 9: Generate temporary read-only SAS URLs
# ============================================================

container_name = "aml-graph-data"
blob_prefix = "v2"

expected_files = [
    "accounts.csv",
    "banks.csv",
    "entities.csv",
    "transactions.csv",
    "owns.csv",
    "held_at.csv",
    "sent.csv",
    "received_by.csv"
]

# Retrieve the connection string securely.
connection_string = dbutils.secrets.get(
    scope="azure_secrets",
    key="storage_connection_string"
)

# Parse the connection string without printing it.
connection_parts = {}

for part in connection_string.split(";"):
    if "=" in part:
        key, value = part.split("=", 1)
        connection_parts[key] = value

account_name = connection_parts.get("AccountName")
account_key = connection_parts.get("AccountKey")
endpoint_suffix = connection_parts.get(
    "EndpointSuffix",
    "core.windows.net"
)

if not account_name or not account_key:
    raise ValueError(
        "The storage connection string must contain "
        "AccountName and AccountKey."
    )

# Start slightly in the past to avoid clock-skew problems.
sas_start = datetime.now(timezone.utc) - timedelta(minutes=5)

# Two hours is enough for the import.
sas_expiry = datetime.now(timezone.utc) + timedelta(hours=2)

sas_urls = {}

for filename in expected_files:
    blob_name = f"{blob_prefix}/{filename}"

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        start=sas_start,
        expiry=sas_expiry,
        protocol="https"
    )

    blob_url = (
        f"https://{account_name}.blob.{endpoint_suffix}/"
        f"{container_name}/{blob_name}"
    )

    sas_urls[filename] = f"{blob_url}?{sas_token}"

print("Generated temporary read-only SAS URLs.")
print(f"Expiry UTC: {sas_expiry.isoformat()}")

# COMMAND ----------

# ============================================================
# Validate SAS URLs without downloading complete files
# ============================================================

validation_results = []

for filename, url in sas_urls.items():
    response = requests.get(
        url,
        headers={"Range": "bytes=0-200"},
        timeout=30
    )

    accessible = response.status_code in (200, 206)

    validation_results.append(
        (
            filename,
            response.status_code,
            accessible,
            response.headers.get("Content-Type"),
            len(response.content)
        )
    )

    response.close()

df_sas_validation = spark.createDataFrame(
    validation_results,
    [
        "filename",
        "http_status",
        "accessible",
        "content_type",
        "downloaded_test_bytes"
    ]
)

print("=== SAS URL VALIDATION ===")
df_sas_validation.show(truncate=False)

# COMMAND ----------

# Display URLs. It must be secret!

# for filename, url in sas_urls.items():
#     print(f"\n--- {filename} ---")
#     print(url)