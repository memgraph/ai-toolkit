from langchain_core.prompts.prompt import PromptTemplate

MEMGRAPH_GENERATION_TEMPLATE = """Your task is to directly translate natural language inquiry into precise and executable Cypher query for Memgraph database. 
You will utilize a provided database schema to understand the structure, nodes and relationships within the Memgraph database.
Instructions: 
- Use provided node and relationship labels and property names from the
schema which describes the database's structure. Upon receiving a user
question, synthesize the schema to craft a precise Cypher query that
directly corresponds to the user's intent. 
- Generate valid executable Cypher queries on top of Memgraph database. 
Any explanation, context, or additional information that is not a part 
of the Cypher query syntax should be omitted entirely. 
- Use Memgraph MAGE procedures instead of Neo4j APOC procedures. 
- Do not include any explanations or apologies in your responses. 
- Do not include any text except the generated Cypher statement.
- For queries that ask for information or functionalities outside the direct
generation of Cypher queries, use the Cypher query format to communicate
limitations or capabilities. For example: RETURN "I am designed to generate
Cypher queries based on the provided schema only."
Schema: 
{schema}

With all the above information and instructions, generate Cypher query for the
user question. 

The question is:
{question}"""

MEMGRAPH_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=MEMGRAPH_GENERATION_TEMPLATE
)


MEMGRAPH_QA_TEMPLATE = """Your task is to form nice and human
understandable answers. The information part contains the provided
information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to
use your internal knowledge to correct it. Make the answer sound as a
response to the question. Do not mention that you based the result on the
given information. Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers. If the provided information is
empty, say that you don't know the answer.

Information:
{context}

Question: {question}
Helpful Answer:"""
MEMGRAPH_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=MEMGRAPH_QA_TEMPLATE
)
