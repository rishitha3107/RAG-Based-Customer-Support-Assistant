"""
CLI Interface for the RAG Customer Support Assistant.
Interactive terminal-based chatbot with colored output.

Usage:
    python app_cli.py
"""
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.hitl_manager import HITLManager
from src.graph_workflow import RAGGraphWorkflow

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


# ─── ANSI Color Codes ───────────────────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def print_banner():
    """Print the application banner."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    🤖 TechCorp Customer Support Assistant (RAG)         ║")
    print("║    Powered by LangGraph + Gemini + ChromaDB             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Commands:                                              ║")
    print("║    • Type your question to get support                  ║")
    print("║    • 'status'  — View system status                     ║")
    print("║    • 'help'    — Show this menu                         ║")
    print("║    • 'quit'    — Exit the application                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")


def print_response(result: dict):
    """Format and print the chatbot response."""
    print(f"\n{Colors.GREEN}{'━' * 60}{Colors.RESET}")

    # Intent info
    intent = result.get("intent", "unknown")
    intent_conf = result.get("intent_confidence", 0)
    print(f"  {Colors.DIM}🏷️  Intent: {intent} (confidence: {intent_conf:.2f}){Colors.RESET}")

    # Retrieved docs count
    docs = result.get("retrieved_docs", [])
    print(f"  {Colors.DIM}📄 Retrieved {len(docs)} relevant documents{Colors.RESET}")

    # Main response
    handled_by = result.get("handled_by", "ai")
    icon = "🤖" if handled_by == "ai" else "👤"
    label = "AI Assistant" if handled_by == "ai" else "Human Agent"

    print(f"\n  {Colors.BOLD}{icon} {label}:{Colors.RESET}")
    response = result.get("response", "No response generated")
    # Indent response lines
    for line in response.split("\n"):
        print(f"  {Colors.WHITE}{line}{Colors.RESET}")

    # Confidence
    confidence = result.get("confidence", 0)
    if handled_by == "ai":
        color = Colors.GREEN if confidence >= 0.7 else Colors.YELLOW if confidence >= 0.5 else Colors.RED
        bar_len = int(confidence * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n  {color}📊 Confidence: [{bar}] {confidence:.0%}{Colors.RESET}")

    # Sources
    sources = result.get("source_documents", [])
    if sources:
        print(f"\n  {Colors.DIM}📑 Sources:{Colors.RESET}")
        for src in sources[:3]:
            print(f"     {Colors.DIM}• Page {src.get('page', '?')}: {src.get('excerpt', 'N/A')[:80]}...{Colors.RESET}")

    # Error
    if result.get("error"):
        print(f"\n  {Colors.RED}⚠️  Error: {result['error']}{Colors.RESET}")

    print(f"{Colors.GREEN}{'━' * 60}{Colors.RESET}")


def print_status(embedding_mgr: EmbeddingManager):
    """Print system status information."""
    stats = embedding_mgr.get_collection_stats()

    print(f"\n{Colors.CYAN}{'━' * 60}")
    print(f"  📊 System Status")
    print(f"{'━' * 60}{Colors.RESET}")
    print(f"  Vector Store: {Colors.GREEN}{stats.get('status', 'unknown')}{Colors.RESET}")
    print(f"  Documents:    {stats.get('document_count', 0)}")
    print(f"  Collection:   {stats.get('collection_name', 'N/A')}")
    print(f"  Storage:      {stats.get('persist_directory', 'N/A')}")
    print(f"  Embedding:    {stats.get('embedding_model', 'N/A')}")
    print(f"  LLM:          {config.LLM_MODEL_NAME}")
    print(f"{Colors.CYAN}{'━' * 60}{Colors.RESET}\n")


def main():
    """Main CLI loop."""
    print_banner()

    # ─── Initialize Components ──────────────────────────────
    print(f"{Colors.YELLOW}⏳ Initializing system...{Colors.RESET}")

    try:
        embedding_mgr = EmbeddingManager()
        vectorstore = embedding_mgr.load_vectorstore()

        if vectorstore is None:
            print(f"\n{Colors.RED}❌ No knowledge base found!{Colors.RESET}")
            print(f"   Run 'python ingest.py' first to process your PDF documents.")
            print(f"   Place PDFs in the '{config.PDF_DIRECTORY}/' directory.\n")
            sys.exit(1)

        stats = embedding_mgr.get_collection_stats()
        if stats.get("document_count", 0) == 0:
            print(f"\n{Colors.RED}❌ Knowledge base is empty!{Colors.RESET}")
            print(f"   Run 'python ingest.py' to ingest PDF documents.\n")
            sys.exit(1)

        retriever = DocumentRetriever(vectorstore)
        llm_handler = LLMHandler()
        hitl_manager = HITLManager(mode="cli")
        workflow = RAGGraphWorkflow(retriever, llm_handler, hitl_manager)

        print(f"{Colors.GREEN}✅ System ready! ({stats.get('document_count', 0)} documents loaded){Colors.RESET}\n")

    except Exception as e:
        print(f"\n{Colors.RED}❌ Initialization failed: {e}{Colors.RESET}")
        print(f"   Make sure you have:")
        print(f"   1. Set GOOGLE_API_KEY in your .env file")
        print(f"   2. Run 'pip install -r requirements.txt'")
        print(f"   3. Run 'python ingest.py' to build the knowledge base\n")
        sys.exit(1)

    # ─── Chat Loop ──────────────────────────────────────────
    while True:
        try:
            query = input(f"\n{Colors.BOLD}{Colors.BLUE}You: {Colors.RESET}").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print(f"\n{Colors.CYAN}👋 Thank you for using TechCorp Support. Goodbye!{Colors.RESET}\n")
                break

            if query.lower() == "help":
                print_banner()
                continue

            if query.lower() == "status":
                print_status(embedding_mgr)
                continue

            # Run the RAG workflow
            print(f"{Colors.YELLOW}⏳ Processing your query...{Colors.RESET}")
            result = workflow.run(query)
            print_response(result)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}👋 Goodbye!{Colors.RESET}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
            print(f"   Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
