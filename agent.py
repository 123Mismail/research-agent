import arxiv
import json
import os
from typing import List
import asyncio
from agents import Agent ,Runner , function_tool , OpenAIChatCompletionsModel ,AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL")
PAPER_DIR = "papers"

set_tracing_disabled(True)
client = AsyncOpenAI(
    api_key =api_key,
    base_url=base_url
)
model =OpenAIChatCompletionsModel(
    model = model,
    openai_client=client
)
run_config=RunConfig(
    model = model ,
    model_provider=client
)


@function_tool
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """

    # Use arxiv to find the papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids


@function_tool
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."

@function_tool
def summarize_paper(paper_json: str) -> str:
    """
    Provide a structured summary of a paper’s key components for analysis.

    Args:
        paper_json: The JSON string containing paper metadata and summary.

    Returns:
        A human-readable summary of important paper elements.
    """
    try:
        paper = json.loads(paper_json)
        summary = f"Title: {paper.get('title', 'N/A')}\n"
        summary += f"Authors: {', '.join(paper.get('authors', []))}\n"
        summary += f"Published: {paper.get('published', 'N/A')}\n"
        summary += f"PDF Link: {paper.get('pdf_url', 'N/A')}\n"
        summary += "\nKey Points:\n"
        summary += f"- Problem Statement & Objective: [EXTRACT FROM SUMMARY]"\
                   f"\n- Key Techniques/Methods: [EXTRACT FROM SUMMARY]"\
                   f"\n- Positive/Negative Implications: [EXTRACT FROM SUMMARY]"\
                   f"\n- Considerations or Limitations: [EXTRACT FROM SUMMARY]"
        return summary
    except json.JSONDecodeError:
        return "Invalid paper JSON provided."


@function_tool
def extract_title_and_abstract(paper_json: str) -> str:
    """
    Extract only the title and abstract from a paper JSON.

    Args:
        paper_json: The JSON string containing paper metadata.

    Returns:
        A string with title and abstract only.
    """
    try:
        paper = json.loads(paper_json)
        return f"Title: {paper.get('title', 'N/A')}\n\nAbstract:\n{paper.get('summary', 'N/A')}"
    except json.JSONDecodeError:
        return "Invalid paper JSON provided."
    
    
ResearchAssistantAgent = Agent(
    name="Research Agent",
    instructions=(
        "You are a helpful research assistant. Whenever a user asks for a research paper, "
        "automatically search for relevant papers, pick one of the top results, and provide:\n"
        "- The paper's title\n"
        "- Authors\n"
        "- Publication date\n"
        "- Direct URL to the PDF\n"
        "- A brief structured summary that includes the paper's objective, methods, key findings or implications, and any limitations.\n"
        "Do this even if the user doesn't explicitly request all those details."
    ),
    tools=[search_papers, extract_info, summarize_paper, extract_title_and_abstract],
    model=model
)

