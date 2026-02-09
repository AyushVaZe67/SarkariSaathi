"""
RAG-based Chatbot for Girls Education Schemes in Maharashtra
Complete pipeline with chunking, vector database, and chatbot functionality

Features:
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Groq LLM (llama-3.1-8b-instant)
- Conversation history
- Source scheme citations
- Persistent vector database
"""

import json
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


VECTOR_DIR = "vectorstore"
os.makedirs(VECTOR_DIR, exist_ok=True)


class DocumentChunker:
    """Handles document chunking from JSON data"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_scheme(self, scheme: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from a scheme document
        Each chunk contains context from the scheme
        """
        chunks = []

        # Create a header with basic info for all chunks
        header = f"Scheme: {scheme.get('scheme_name', 'N/A')}\n"
        header += f"Scheme ID: {scheme.get('scheme_id', 'N/A')}\n"
        header += f"State: {scheme.get('state', 'N/A')}\n"
        if scheme.get('ministry'):
            header += f"Ministry: {scheme['ministry']}\n"

        # Chunk 1: Description and Benefits
        if scheme.get('description') or scheme.get('benefits'):
            chunk_text = header + "\n"
            if scheme.get('description'):
                chunk_text += f"Description: {scheme['description']}\n\n"
            if scheme.get('benefits'):
                chunk_text += f"Benefits: {scheme['benefits']}\n"

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'scheme_id': scheme.get('scheme_id'),
                    'scheme_name': scheme.get('scheme_name'),
                    'state': scheme.get('state'),
                    'type': 'description_benefits',
                    'source_url': scheme.get('source_url', '')
                }
            })

        # Chunk 2: Eligibility and Relaxation/Priority
        if scheme.get('eligibility') or scheme.get('relaxation_priority'):
            chunk_text = header + "\n"
            if scheme.get('eligibility'):
                chunk_text += f"Eligibility: {scheme['eligibility']}\n\n"
            if scheme.get('relaxation_priority'):
                chunk_text += f"Relaxation/Priority: {scheme['relaxation_priority']}\n"

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'scheme_id': scheme.get('scheme_id'),
                    'scheme_name': scheme.get('scheme_name'),
                    'state': scheme.get('state'),
                    'type': 'eligibility',
                    'source_url': scheme.get('source_url', '')
                }
            })

        # Chunk 3: Application Process and Documents
        if scheme.get('application_process') or scheme.get('documents_required'):
            chunk_text = header + "\n"
            if scheme.get('application_process'):
                chunk_text += f"Application Process: {scheme['application_process']}\n\n"
            if scheme.get('documents_required'):
                chunk_text += f"Documents Required: {scheme['documents_required']}\n"

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'scheme_id': scheme.get('scheme_id'),
                    'scheme_name': scheme.get('scheme_name'),
                    'state': scheme.get('state'),
                    'type': 'application',
                    'source_url': scheme.get('source_url', '')
                }
            })

        # Chunk 4: Additional Information (tenure, FAQs, etc.)
        additional_fields = ['tenure', 'faqs', 'objectives', 'ministry', 'implementing_agency', 'source_url']
        additional_info = []
        for field in additional_fields:
            if scheme.get(field):
                additional_info.append(f"{field.replace('_', ' ').title()}: {scheme[field]}")

        if additional_info:
            chunk_text = header + "\n" + "\n\n".join(additional_info)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'scheme_id': scheme.get('scheme_id'),
                    'scheme_name': scheme.get('scheme_name'),
                    'state': scheme.get('state'),
                    'type': 'additional_info',
                    'source_url': scheme.get('source_url', '')
                }
            })

        return chunks

    def process_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Process JSON file and create chunks"""
        with open(filepath, 'r', encoding='utf-8') as f:
            schemes = json.load(f)

        all_chunks = []
        for scheme in schemes:
            chunks = self.chunk_scheme(scheme)
            all_chunks.extend(chunks)

        return all_chunks


class VectorDatabase:
    """Manages vector storage and retrieval using FAISS"""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"📦 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        print(f"✅ Vector database initialized with dimension: {self.dimension}")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector database"""
        print(f"\n📝 Adding {len(chunks)} chunks to vector database...")

        texts = [chunk['text'] for chunk in chunks]

        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))

        # Store chunks for retrieval
        self.chunks.extend(chunks)

        print(f"✅ Successfully added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'),
            k
        )

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'distance': float(distance),
                    'similarity_score': 1 / (1 + float(distance))  # Convert distance to similarity
                })

        return results

    def save(self, index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
        """Save vector database to disk"""
        print(f"\n💾 Saving vector database...")
        faiss.write_index(self.index, os.path.join(VECTOR_DIR, "faiss_index.bin"))
        with open(os.path.join(VECTOR_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Vector database saved to {index_path} and {chunks_path}")

    def load(self, index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
        """Load vector database from disk"""
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"\n📂 Loading existing vector database...")
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"✅ Vector database loaded. Total chunks: {len(self.chunks)}")
            return True
        return False


class RAGChatbot:
    """RAG-based chatbot using Groq LLM with conversation history and source citations"""

    def __init__(
            self,
            vector_db: VectorDatabase,
            groq_api_key: str = None,
            model: str = "llama-3.1-8b-instant"
    ):
        self.vector_db = vector_db
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it as an environment variable:\n"
                "export GROQ_API_KEY='your_api_key_here'"
            )

        self.client = Groq(api_key=self.groq_api_key)
        self.conversation_history = []
        print(f"✅ Chatbot initialized with model: {model}")

    def _create_context(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context from vector database"""
        results = self.vector_db.search(query, k=k)

        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            metadata = chunk['metadata']
            similarity = result['similarity_score']

            context_parts.append(f"--- Context {i} (Relevance: {similarity:.2f}) ---\n{chunk['text']}\n")

            # Collect source information
            sources.append({
                'scheme_name': metadata.get('scheme_name', 'Unknown'),
                'scheme_id': metadata.get('scheme_id', 'N/A'),
                'state': metadata.get('state', 'N/A'),
                'type': metadata.get('type', 'general'),
                'source_url': metadata.get('source_url', ''),
                'relevance': similarity
            })

        context = "\n".join(context_parts)
        return context, sources

    def _create_prompt(self, query: str, context: str) -> str:
        """Create system prompt for the LLM"""
        system_prompt = """You are a knowledgeable and helpful assistant specializing in girls' education schemes in Maharashtra and across India. Your purpose is to help students, parents, and educators find relevant information about scholarships, financial aid, and educational programs for girls.

IMPORTANT GUIDELINES:
1. Use ONLY the information provided in the context to answer questions
2. Be specific about scheme names, eligibility criteria, benefits, and application processes
3. If the context doesn't contain enough information to answer the question completely, clearly state what information is missing
4. Be encouraging and supportive when discussing educational opportunities
5. Provide step-by-step guidance when asked about application processes
6. Mention specific amounts, dates, and requirements when available
7. If multiple schemes are relevant, compare them briefly to help the user choose
8. Always be accurate - don't make up information not present in the context

CONTEXT INFORMATION:
{context}

Based on the above context, please answer the user's question in a helpful and informative manner."""

        return system_prompt.format(context=context)

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source schemes for display"""
        if not sources:
            return ""

        source_text = "\n\n📚 **Source Schemes Referenced:**\n"
        source_text += "=" * 60 + "\n"

        seen_schemes = set()
        for source in sources:
            scheme_id = source['scheme_id']
            if scheme_id not in seen_schemes:
                seen_schemes.add(scheme_id)
                source_text += f"\n📌 {source['scheme_name']}\n"
                source_text += f"   ID: {scheme_id}\n"
                source_text += f"   State: {source['state']}\n"
                source_text += f"   Relevance: {source['relevance']:.2%}\n"
                if source.get('source_url'):
                    source_text += f"   URL: {source['source_url']}\n"

        source_text += "\n" + "=" * 60
        return source_text

    def chat(self, user_query: str, k: int = 3, show_sources: bool = True) -> str:
        """
        Process user query and generate response

        Args:
            user_query: User's question
            k: Number of relevant chunks to retrieve
            show_sources: Whether to show source schemes in response

        Returns:
            Chatbot response with optional source citations
        """
        print(f"\n🔍 Searching for relevant information...")

        # Retrieve relevant context and sources
        context, sources = self._create_context(user_query, k=k)

        # Create system prompt
        system_prompt = self._create_prompt(user_query, context)

        # Prepare messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last 4 exchanges = 8 messages)
        if self.conversation_history:
            recent_history = self.conversation_history[-8:]
            messages.extend(recent_history)

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        print(f"💬 Generating response using {self.model}...")

        # Get response from Groq
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9,
            )

            response = chat_completion.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Add source citations if requested
            if show_sources:
                response += self._format_sources(sources)

            return response

        except Exception as e:
            return f"❌ Error generating response: {str(e)}\n\nPlease check your GROQ_API_KEY and internet connection."

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared.")

    def get_history_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history."

        exchanges = len(self.conversation_history) // 2
        return f"Conversation history: {exchanges} exchange(s)"


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🎓 Girls Education Schemes RAG Chatbot 🎓                          ║
║                                                                              ║
║                    Powered by HuggingFace + Groq                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print help information"""
    help_text = """
📖 Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💬 Ask Questions:
   - Just type your question naturally
   - Examples:
     • "What schemes are available for SC/ST girls?"
     • "Tell me about scholarship for higher education abroad"
     • "What documents do I need for National Scheme of Incentive?"

⚙️  Commands:
   • help     - Show this help message
   • clear    - Clear conversation history
   • history  - Show conversation history summary
   • stats    - Show database statistics
   • quit/exit - Exit the chatbot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(help_text)


def setup_database(json_file: str, force_rebuild: bool = False) -> VectorDatabase:
    """Setup and return vector database"""
    index_path = "/home/claude/faiss_index.bin"
    chunks_path = "/home/claude/chunks.pkl"

    # Initialize vector database
    vector_db = VectorDatabase(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if we should load existing database
    if not force_rebuild and vector_db.load(index_path, chunks_path):
        return vector_db

    # Build new database
    print("\n🔧 Building new vector database...")
    chunker = DocumentChunker(chunk_size=500, overlap=50)

    print(f"📄 Processing JSON file: {json_file}")
    chunks = chunker.process_json_file(json_file)
    print(f"✅ Created {len(chunks)} chunks from the document")

    vector_db.add_documents(chunks)
    vector_db.save(index_path, chunks_path)

    return vector_db


def main():
    """Main function to run the RAG chatbot"""

    print_banner()

    # Configuration
    json_file = "data/girls_education_maharashtra.json"

    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"❌ Error: JSON file not found at {json_file}")
        print("Please ensure the file is uploaded correctly.")
        return

    # Setup vector database
    try:
        vector_db = setup_database(json_file, force_rebuild=False)
    except Exception as e:
        print(f"❌ Error setting up vector database: {e}")
        return

    # Initialize chatbot
    print("\n🤖 Initializing RAG Chatbot...")
    try:
        chatbot = RAGChatbot(vector_db, model="llama-3.1-8b-instant")
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nTo set your GROQ API key, run:")
        print("export GROQ_API_KEY='your_api_key_here'")
        return
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return

    # Print instructions
    print("\n" + "=" * 80)
    print("                          💬 CHAT INTERFACE")
    print("=" * 80)
    print("\nType 'help' for available commands, or start asking questions!")
    print("=" * 80 + "\n")

    # Interactive chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Girls Education Schemes Chatbot!")
                print("   Keep empowering girls through education! 🎓✨\n")
                break

            elif user_input.lower() == 'help':
                print_help()
                continue

            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                continue

            elif user_input.lower() == 'history':
                print(f"\n📊 {chatbot.get_history_summary()}\n")
                continue

            elif user_input.lower() == 'stats':
                print(f"\n📊 Database Statistics:")
                print(f"   Total chunks: {len(vector_db.chunks)}")
                print(f"   Embedding dimension: {vector_db.dimension}")
                print(f"   {chatbot.get_history_summary()}\n")
                continue

            # Get chatbot response
            response = chatbot.chat(user_input, k=3, show_sources=True)

            # Print response
            print(f"\n🤖 Assistant:\n{response}\n")
            print("=" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting... Thank you for using the chatbot!\n")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")
            continue


if __name__ == "__main__":
    print("DEBUG | GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
    main()