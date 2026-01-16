import os
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j environment variables")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(
    title="Course Prerequisite API",
    version="1.0.0"
)

# ===============================
# CYPHER QUERY
# ===============================
CYPHER_QUERY = """
MATCH (root:Course {course_master_id: $course_id})

OPTIONAL MATCH prereq_path =
  (p:Course)-[:PREREQUISITE*1..]->(root)

OPTIONAL MATCH (g:PrereqGroup)-[:REQUIRES_FOR]->(root)
OPTIONAL MATCH (g)-[m:HAS_MEMBER]->(member:Course)

OPTIONAL MATCH coreq_path =
  (root)-[:COREQUISITE*1..]->(coreq:Course)

WITH
  root,
  collect(DISTINCT p.course_master_id) AS simple_prereqs,
  collect(DISTINCT {
    group_id: g.id,
    op: g.op,
    members: collect(DISTINCT {
      course: member.course_master_id,
      concurrent: coalesce(m.concurrent, false)
    })
  }) AS logical_groups,
  collect(DISTINCT coreq.course_master_id) AS coreqs

RETURN {
  course_id: root.course_master_id,
  title: root.title,
  prerequisites: simple_prereqs,
  logical_groups: [g IN logical_groups WHERE g.group_id IS NOT NULL],
  corequisites: coreqs
} AS result
"""

# ===============================
# API ENDPOINT
# ===============================
@app.get("/course/{course_id}")
def get_course_prerequisites(course_id: str):
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            CYPHER_QUERY,
            course_id=course_id
        ).single()

        if not record:
            raise HTTPException(status_code=404, detail="Course not found")

        return record["result"]


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health():
    return {"status": "ok"}
