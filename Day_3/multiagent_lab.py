# ============================================================
# DAY 3 LAB — Build a Multi Agent 
# ============================================================
#Using Day 2 SKELETON
# Fill in every TODO. Each step tells you exactly WHERE in the
# LangGraph docs to look. Don't copy from the solution file
# (enterprise_research_agent.py) until you've tried each step —
# the point of Day 2 is learning to THINK in state graphs.
#
# The system you're building:
#
#   START → collect → store_memory → analyze → evaluate
#              ↑                                  │
#              └── quality < 7 (max 3 tries) ─────┤
#                                                 └ quality >= 7
#                                                       ↓
#                                          report → audit → END
#
# Recommended reading order BEFORE you start (30 min total):
#   1. "Thinking in LangGraph" (the mental model):
#      https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
#   2. Graph API concepts (State, Nodes, Edges):
#      https://docs.langchain.com/oss/python/langgraph/graph-api
#   3. Using the Graph API (code patterns you'll copy):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api
#
# API reference (exact signatures when docs aren't enough):
#   https://reference.langchain.com/python/langgraph/
#
# Setup: pip install -r requirements.txt, then create .env
# (or set USE_FAKE=1 — see README.md).
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Dict
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.vectorstores import InMemoryVectorStore
# TODO STEP 0 — import the graph building blocks from langgraph.
# You need: StateGraph, START, END from langgraph.graph
#           InMemorySaver from langgraph.checkpoint.memory
# WHERE TO LOOK: "Graph API" docs, first code example on the page.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

if not os.getenv("OPENROUTER_BASE_URL"):
    raise ValueError("OPENROUTER_BASE_URL is missing from .env")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing from .env")
# ============================================================
# STEP 1 — THE STATE  (the "digital clipboard" from the slides)
# ============================================================
# Define a TypedDict with everything the workflow needs to remember:
#   topic (str), search_query (str), collected_data (List[Dict]),
#   analyzed_data (List[Dict]), quality_score (int),
#   iteration_count (int), final_report (str), execution_logs
#
# KEY IDEA: execution_logs should use a REDUCER so every node can
# APPEND log lines instead of overwriting the list:
#     execution_logs: Annotated[List[str], operator.add]
#
# WHERE TO LOOK: Graph API docs → "State" section → "Reducers".
#   https://docs.langchain.com/oss/python/langgraph/graph-api
# ASK YOURSELF: what happens to a plain (non-reducer) key when two
# nodes write it? What happens with operator.add?

class AgentState(TypedDict):
    topic: str
    # TODO: add the remaining 6 keys (one uses Annotated + operator.add)
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    report_file_path: str
    execution_logs: Annotated[List[str], operator.add]

    
# ============================================================
# STEP 2 — MODEL, SEARCH TOOL, EMBEDDINGS
# ============================================================
# Create:
#   llm          = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#   search_tool  = TavilySearch(max_results=5)   # langchain_tavily!
#   vector_store = a Chroma or InMemoryVectorStore with OpenAIEmbeddings
#
# GOTCHA: the old imports you'll find in 2023-24 tutorials
# (langchain.vectorstores, langchain_community.tools.tavily_search)
# are DEAD. Current homes:
#   - TavilySearch:      https://docs.langchain.com/oss/python/integrations/providers/tavily
#   - Chat models:       https://docs.langchain.com/oss/python/langchain/models
#   - InMemoryVectorStore: langchain_core.vectorstores
#
# NOTE: TavilySearch.invoke({"query": q}) returns a DICT — the
# actual sources are under the "results" key. print() it once to see.

# TODO: your code here
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

search_tool = TavilySearch(max_results=5)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

vector_store = InMemoryVectorStore(embeddings)


# ============================================================
# STEP 3 — STRUCTURED OUTPUT for the quality score
# ============================================================
# Never parse int(response.content) out of free text. Define a
# Pydantic schema and use llm.with_structured_output(...) so the
# model is FORCED to return valid data.
#
# WHERE TO LOOK: https://docs.langchain.com/oss/python/langchain/structured-output
# ASK YOURSELF: what does with_structured_output return — a string,
# a dict, or a QualityScore object?

class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")

# TODO: 
evaluator = llm.with_structured_output(QualityScore)


# ============================================================
# STEP 4 — NODES
# ============================================================
# A node is just a function: takes state, returns a PARTIAL update
# (a dict with ONLY the keys it changed). LangGraph merges it in.
# Do NOT mutate state in place; do NOT return the whole state.
#
# WHERE TO LOOK: Use Graph API docs → "Define and update state".
#   https://docs.langchain.com/oss/python/langgraph/use-graph-api

def collect_node(state: AgentState):
    """Search the web. On retries, CHANGE the query!"""
    # TODO:
    # 1. iteration = state["iteration_count"] + 1
    # 2. Build a query that DIFFERS per iteration (why? see Step 5)
    # 3. results = search_tool.invoke({"query": query})["results"]
    # 4. return {"search_query": ..., "collected_data": ...,
    #            "iteration_count": ..., "execution_logs": [...]}

    iteration = state["iteration_count"] + 1

    if iteration == 1:
        query = f"{state['topic']} latest trends and developments"
    else:
        query = (
            f"{state['topic']} enterprise use cases, challenges, "
            f"and recent research iteration {iteration}"
        )

    response = search_tool.invoke({"query": query})
    results = response["results"]

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": [
            f"Iteration {iteration}: collected {len(results)} sources"
        ],
    }

def store_memory_node(state: AgentState):
    """Save source contents into the vector store."""
    # TODO: vector_store.add_texts([...contents...])

    contents = [
        source["content"]
        for source in state["collected_data"]
        if source.get("content")
    ]

    if contents:
        vector_store.add_texts(contents)

    return {
        "execution_logs": [
            f"Stored {len(contents)} source contents in vector memory"
        ]
    }

def analyze_node(state: AgentState):
    """LLM-analyze each source. Bonus: retrieve related past
    research with vector_store.similarity_search(content, k=2)
    and include it in the prompt — that's what makes this RAG."""
    # TODO

    analyzed_results = []

    for source in state["collected_data"]:
        content = source.get("content", "")

        if not content:
            continue

        related_documents = vector_store.similarity_search(
            content,
            k=2,
        )

        related_context = "\n\n".join(
            document.page_content
            for document in related_documents
        )

        prompt = f"""
You are an enterprise research analyst.

Research topic:
{state["topic"]}

Current source title:
{source.get("title", "Unknown title")}

Current source URL:
{source.get("url", "Unknown URL")}

Current source content:
{content}

Related information retrieved from vector memory:
{related_context}

Analyze the source and provide:
1. A concise summary.
2. The main findings.
3. Enterprise opportunities.
4. Risks or challenges.
5. How relevant the source is to the research topic.

Keep the response clear and concise.
"""

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        analyzed_results.append({
            "title": source.get("title", "Unknown title"),
            "url": source.get("url", ""),
            "analysis": response.content,
        })

    return {
        "analyzed_data": analyzed_results,
        "execution_logs": [
            f"Analyzed {len(analyzed_results)} sources using LLM and RAG"
        ],
    }

def evaluate_node(state: AgentState):
    """Score the research with the STRUCTURED evaluator (Step 3)."""
    # TODO: return {"quality_score": result.score, "execution_logs": [...]}
    prompt = f"""
You are evaluating the quality of enterprise research.

Research topic:
{state["topic"]}

Analyzed sources:
{state["analyzed_data"]}

Evaluate the overall research quality from 1 to 10.

Consider:
- relevance to the topic
- clarity of analysis
- diversity of sources
- usefulness for enterprise decision-making
- coverage of opportunities and risks
"""

    result = evaluator.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "quality_score": result.score,
        "execution_logs": [
            f"Research quality scored {result.score}/10: {result.reasoning}"
        ],
    }

def export_report(report_content: str) -> str:
    """Export the generated report as a Markdown file."""

    output_directory = Path("outputs")
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"enterprise_report_{timestamp}.md"

    report_path = output_directory / file_name

    report_path.write_text(
        report_content,
        encoding="utf-8",
    )

    return str(report_path)

def report_agent_node(state: AgentState):
    """Generate and export a professional report."""
    # TODO
    prompt = f"""
You are the Report Agent in a multi-agent research system.

Your responsibility is to transform the approved research analysis
into a clear, professional, and evidence-based enterprise report.

Research topic:
{state["topic"]}

Analyzed research:
{state["analyzed_data"]}

Quality score:
{state["quality_score"]}/10

Write a clear enterprise research report with these sections:

1. Executive Summary
2. Key Findings
3. Enterprise Opportunities
4. Risks and Challenges
5. Recommendations
6. Conclusion
7. Sources
List every supplied source with:
- Source title
- URL

Use only the supplied research.
Do not invent unsupported facts.

When referencing evidence inside the report, use the source title
instead of labels such as "source 1" or "source 2".

Clearly distinguish between academic, official, industry, news,
conference, blog, and professional-network sources.

Do not describe a source as peer-reviewed, official, or authoritative
unless that classification is explicitly supported by the supplied data.

Avoid presenting numerical claims unless they are clearly supported
by at least one listed source.

Return the complete report in Markdown format.
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    report_content = response.content
    report_file_path = export_report(report_content)

    return {
        "final_report": report_content,
        "report_file_path": report_file_path,
        "execution_logs": [
        f"Report Agent generated and exported the report to {report_file_path}"
        ],
    }


def audit_node(state: AgentState):
    """Log completion stats."""
    # TODO
    
    return {
        "execution_logs": [
            (
                f"Research completed after {state['iteration_count']} "
                f"iteration(s), with {len(state['collected_data'])} sources "
                f"and a quality score of {state['quality_score']}/10"
            )
        ]
    }

# ============================================================
# STEP 5 — THE CONDITIONAL EDGE (the heart of this lab)
# ============================================================
# Write a router function: takes state, RETURNS THE NAME of the
# next node as a string.
#
# CRITICAL — loops must terminate. Two rules:
#   a) every retry must change something (your query, Step 4.2),
#   b) hard-cap the retries with iteration_count.
# Without both, same search → same score → infinite loop → LangGraph
# kills the run at recursion limit 25 with GraphRecursionError.
#
# WHERE TO LOOK (read BOTH):
#   - "Conditional branching":
#     https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
#   - "Create and control loops":
#     https://docs.langchain.com/oss/python/langgraph/use-graph-api#create-and-control-loops
#
# EXPERIMENT: comment out the iteration cap, force low scores, run,
# and read the GraphRecursionError message. Now you understand why
# the docs insist on termination conditions.

def quality_router(state: AgentState) -> str:
    # TODO: return "report" or "collect"
    quality_score = state["quality_score"]
    iteration_count = state["iteration_count"]

    if quality_score >= 7:
        return "report_agent"

    if iteration_count >= 3:
        return "report_agent"

    return "collect"


# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================
# 1. workflow = StateGraph(AgentState)
# 2. add_node(...) for all six nodes
# 3. add_edge(START, "collect")        <- START, not set_entry_point
# 4. linear edges: collect → store_memory → analyze → evaluate
# 5. add_conditional_edges("evaluate", quality_router,
#        {"collect": "collect", "report": "report"})
#    (the dict maps router RETURN VALUES to NODE NAMES)
# 6. report → audit → END
#
# WHERE TO LOOK: Graph API docs → "Edges".

# TODO: your code here
workflow = StateGraph(AgentState)

workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report_agent", report_agent_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "collect")

workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {
        "collect": "collect",
        "report_agent": "report_agent",
    },
)

workflow.add_edge("report_agent", "audit")
workflow.add_edge("audit", END)

# ============================================================
# STEP 7 — COMPILE with a checkpointer, VISUALIZE, RUN
# ============================================================
# 1. app = workflow.compile(checkpointer=InMemorySaver())
#    A checkpointer saves state after every node → enables resume,
#    time-travel debugging, and human-in-the-loop.
#    WHERE TO LOOK: https://docs.langchain.com/oss/python/langgraph/persistence
#
# 2. Visualize what you built:
#       print(app.get_graph().draw_mermaid())
#    → paste the output into https://mermaid.live
#    Does the picture match the diagram at the top of this file?
#
# 3. Run with STREAMING so you watch state evolve node by node:
#       config = {"configurable": {"thread_id": "run-1"}}  # required
#       for chunk in app.stream(initial_state, config,
#                               stream_mode="values"):
#           ...
#    WHERE TO LOOK: https://docs.langchain.com/oss/python/langgraph/streaming
#
# 4. BONUS — human-in-the-loop: compile with
#       interrupt_before=["report"]
#    then inspect state and resume. WHERE TO LOOK:
#       https://docs.langchain.com/oss/python/langgraph/interrupts

if __name__ == "__main__":
    initial_state = {
        "topic": "Brain-Computer Interfaces and the Future of Human Communication",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "report_file_path": "",
        "execution_logs": [],
    }
    # TODO: compile, visualize, stream, print final report + logs

    # 1. Compile the graph with an in-memory checkpointer
    app = workflow.compile(
        checkpointer=InMemorySaver()
    )

    # 2. Print the Mermaid diagram
    print("\n=== GRAPH DIAGRAM ===\n")
    print(app.get_graph().draw_mermaid())

    # 3. Required when using a checkpointer
    config = {
        "configurable": {
            "thread_id": "run-1"
        }
    }

    final_state = None

    # 4. Stream the full state after every graph step
    print("\n=== EXECUTION STREAM ===\n")

    for chunk in app.stream(
        initial_state,
        config,
        stream_mode="values",
    ):
        final_state = chunk

        print(
            f"Iteration: {chunk['iteration_count']} | "
            f"Score: {chunk['quality_score']}/10 | "
            f"Sources: {len(chunk['collected_data'])}"
        )

    # 5. Print final report and execution logs
    if final_state:
        print("\n=== FINAL REPORT ===\n")
        print(final_state["final_report"])

        print("\n=== EXPORTED REPORT FILE ===\n")
        print(final_state["report_file_path"])

        print("\n=== EXECUTION LOGS ===\n")

        for log in final_state["execution_logs"]:
            print(f"- {log}")

# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] My nodes return partial dicts, never the whole mutated state
# [ ] execution_logs uses a reducer, and I can explain why
# [ ] My router has BOTH a quality exit AND an iteration cap
# [ ] Retried searches use a different query than the first attempt
# [ ] I saw the Mermaid diagram and it matches the intended flow
# [ ] I know what GraphRecursionError is and how to trigger it
# [ ] The quality score comes from with_structured_output, not int()
#
# Stuck? Debugging order that works:
#   1. print() the raw return of search_tool.invoke — check its shape
#   2. run app.stream(..., stream_mode="updates") — shows exactly
#      which node produced which state update
#   3. compare your edge wiring against the diagram at the top
#   4. only THEN open enterprise_research_agent.py
# ============================================================
