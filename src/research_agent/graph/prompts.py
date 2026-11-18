"""Prompt templates. Kept terse and explicit: small models follow short, concrete rules best.

A ChatPromptTemplate is a Runnable: dict of variables -> list of messages. Piping it into a
model (`prompt | llm`) is LCEL (LangChain Expression Language) composition.
"""

from datetime import date

from langchain_core.prompts import ChatPromptTemplate

# Temporal grounding. Observed failure without it: the 4B model called real, recent sources
# "speculative" and a 2026 URL a "future date", because they postdate its training data.
GROUNDING = (
    "Today's date is {today}. Sources may describe things newer than your training data: "
    "trust the sources and never call them fictional, speculative or 'from the future'."
)


def today() -> str:
    return date.today().isoformat()


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research planner. Break the user's question into at most "
            "{max_subquestions} short, specific sub-questions that can each be researched "
            "independently. Do not answer them. Reply in JSON.",
        ),
        ("human", "Question: {question}\n{feedback}"),
    ]
)

RESEARCHER_SYSTEM = (
    "You are a focused research assistant with tools. Answer ONE sub-question.\n"
    + GROUNDING
    + "\nRules:\n"
    "1. Call search_knowledge_base first.{web_rule}\n"
    "2. Use at most {max_tool_rounds} tool calls, then stop calling tools.\n"
    "3. Final answer: 3-6 sentences of factual notes drawn ONLY from tool results. "
    "Mention which source each fact came from. If the tools found nothing, say so."
)

WEB_RULE = (
    " If it finds no relevant passages or not enough, you MUST call web_search; "
    "optionally fetch_url on one promising result URL."
)

RESEARCH_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write 3-6 sentences of factual notes answering the sub-question using ONLY the "
            "tool results below. If they don't contain the answer, say so.\n" + GROUNDING,
        ),
        ("human", "Sub-question: {sub_question}\n\nTool results:\n{tool_results}"),
    ]
)

WRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write concise, well-structured research reports in Markdown.\n"
            "Rules:\n"
            "- Use ONLY the research notes provided. Do not invent facts.\n"
            "- Cite sources inline with their numbers, like [1] or [2][3]. Only use numbers "
            "from the source list.\n"
            "- Structure: a one-paragraph summary, then 2-4 short sections with ## headings.\n"
            "- Do not write a Sources/References section or list URLs; sources are added "
            "automatically.\n" + GROUNDING,
        ),
        (
            "human",
            "Question: {question}\n\nSource list:\n{source_list}\n\nResearch notes:\n{notes}"
            "\n{revision_block}\nWrite the report now.",
        ),
    ]
)

REVISION_BLOCK = (
    "\nA reviewer read your previous draft and asked for changes:\n{feedback}\n\n"
    "Previous draft:\n{draft}\n"
)

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict reviewer (LLM-as-judge). Score the report from 1 to 5:\n"
            "5 = answers the question, every claim supported by the notes, citations used.\n"
            "3 = partly answers or some unsupported claims.\n"
            "1 = off-topic or mostly unsupported.\n"
            "Give short, actionable feedback. Reply in JSON.\n" + GROUNDING,
        ),
        ("human", "Question: {question}\n\nResearch notes:\n{notes}\n\nReport:\n{draft}"),
    ]
)
