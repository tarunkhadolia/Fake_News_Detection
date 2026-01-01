"""
🔍 Fake News Detector
A Streamlit app powered by DuckDuckGo and Gemini AI
"""

import streamlit as st
import json
import time
import warnings
import numpy as np
from typing import List, Dict
from newspaper import Article
from ddgs import DDGS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

warnings.filterwarnings('ignore')

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .valid-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .unverifiable-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .confidence-bar {
        height: 30px;
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# API KEY CONFIGURATION
# ========================================
# For local development, you can hardcode your key here
# For deployment, use Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Fallback for local testing

# ========================================
# INITIALIZE SESSION STATE
# ========================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY

# ========================================
# CLASSES FROM YOUR CODE
# ========================================

class WebSearchEngine:
    """
    This is our internet detective! It searches the web using DuckDuckGo
    and filters results to find trustworthy sources.
    """
    def __init__(self):
        # These are the sources we trust most
        self.trusted_keywords = [
            'reuters', 'apnews', 'bbc', 'cnn', 'npr',  # News outlets
            'snopes', 'factcheck', 'politifact',        # Fact checkers
            'cdc.gov', 'who.int', 'nih.gov'             # Health authorities
        ]
    
    def search(self, query, num_results=10):
        """
        Search the web and return the most relevant, trustworthy results.
        """
        try:
            # Use DuckDuckGo to search
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="wt-wt",        # Search worldwide
                        safesearch="moderate",
                        max_results=num_results
                    )
                )

            if not results:
                return []

            # Filter for trusted sources
            trusted_results = []
            for result in results:
                url = (result.get("href") or "").lower()
                
                # Check if this URL is from a trusted source
                if any(trusted_site in url for trusted_site in self.trusted_keywords):
                    trusted_results.append({
                        "url": result.get("href"),
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "")
                    })

            # If we didn't find any trusted sources, use the top results anyway
            if not trusted_results:
                trusted_results = [{
                    "url": result.get("href"),
                    "title": result.get("title", ""),
                    "snippet": result.get("body", "")
                } for result in results[:3]]

            return trusted_results[:5]  # Return top 5

        except Exception as error:
            st.warning(f"Search error: {error}")
            return []


class ContentScraper:
    """
    This grabs the full text from news articles so we can analyze them properly.
    """
    def scrape_article(self, url):
        """
        Download and extract text from a single article.
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return {
                'url': url,
                'title': article.title,
                'content': article.text,
                'success': True
            }
        except:
            # Sometimes articles can't be scraped (paywalls, errors, etc.)
            return {'url': url, 'success': False}
    
    def scrape_multiple(self, urls):
        """
        Scrape multiple articles and only keep the ones that worked.
        """
        successful_articles = []
        
        for url in urls:
            result = self.scrape_article(url)
            
            # Only keep articles with enough content (200+ characters)
            if result['success'] and len(result.get('content', '')) > 200:
                successful_articles.append(result)
            
            time.sleep(0.5)  # Be polite to websites - don't overwhelm them
        
        return successful_articles


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')


class RAGPipeline:
    """
    RAG = Retrieval Augmented Generation
    This breaks articles into chunks and creates a searchable knowledge base.
    """
    def __init__(self):
        # Load the AI model that understands text meaning
        self.model = load_embedding_model()
        self.dimension = 384  # Size of the AI embeddings
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """
        Break long articles into smaller, overlapping pieces.
        Overlapping helps maintain context between chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - overlap  # Move back a bit to create overlap
        
        return chunks
    
    def create_vectorstore(self, articles):
        """
        Turn all our articles into a searchable knowledge base.
        """
        all_chunks = []
        all_metadata = []
        
        # Break each article into chunks
        for article in articles:
            chunks = self.chunk_text(article['content'])
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': article['url'],
                    'title': article.get('title', '')
                })
        
        # Convert text to AI embeddings (numbers that represent meaning)
        embeddings = self.model.encode(all_chunks)
        
        # Create a FAISS index for super-fast searching
        index = faiss.IndexFlatL2(self.dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, all_chunks, all_metadata
    
    def retrieve_context(self, index, chunks, metadata, claim, k=5):
        """
        Find the most relevant chunks for fact-checking a claim.
        """
        # Convert the claim to an embedding
        query_embedding = self.model.encode([claim])
        
        # Search for the k most similar chunks
        distances, indices = index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Return the matching chunks with their metadata
        relevant_chunks = []
        for idx in indices[0]:
            relevant_chunks.append({
                'content': chunks[idx],
                'metadata': metadata[idx]
            })
        
        return relevant_chunks


class FakeNewsClassifier:
    """
    This uses Google's Gemini AI to analyze claims and determine if they're true.
    """
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.config = {
            'temperature': 0.1,        # Low temperature = more factual
            'max_output_tokens': 2048
        }
    
    def generate_queries(self, claim):
        """
        Ask Gemini to create smart search queries for fact-checking.
        """
        prompt = f"""You're a fact-checker. Generate 3 diverse search queries to verify this claim:

"{claim}"

Return ONLY a Python list format: ["query1", "query2", "query3"]"""
        
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.config
            )
            text = response.text.strip()
            
            # Clean up the response and convert to list
            text = text.replace('```python', '').replace('```', '').strip()
            return eval(text)
        except:
            # Fallback if generation fails
            return [claim, f"{claim} fact check"]
    
    def classify(self, claim, evidence_chunks):
        """
        Analyze a claim against evidence and return a verdict.
        """
        # Combine all evidence into one text
        evidence_text = "\n\n".join([
            f"[Source: {chunk['metadata']['source']}]\n{chunk['content']}"
            for chunk in evidence_chunks
        ])
        
        # Limit evidence length to avoid token limits
        if len(evidence_text) > 15000:
            evidence_text = evidence_text[:15000] + "\n\n[Evidence truncated...]"
        
        prompt = f"""You are an expert fact-checker. Analyze this claim against the evidence provided.

CLAIM TO VERIFY:
{claim}

EVIDENCE FROM TRUSTED SOURCES:
{evidence_text}

YOUR TASK:
Determine if the claim is VALID, FAKE, or NOT_VERIFIABLE based on the evidence.

CRITICAL: Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, no extra text):
{{"classification": "VALID", "confidence": 0.85, "explanation": "Your reasoning here", "key_points": ["Point 1", "Point 2"]}}

Rules:
- classification must be exactly "VALID", "FAKE", or "NOT_VERIFIABLE"
- confidence must be a number between 0.0 and 1.0
- explanation must be a single clear sentence
- key_points must be an array of 2-4 short bullet points
- Use proper JSON escaping for quotes in text"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt, generation_config=self.config
                )
                text = response.text.strip()
                
                # Clean up common formatting issues
                text = text.replace('```json', '').replace('```', '').strip()
                text = text.replace('`', '')
                
                # Try to find JSON object in the response
                if '{' in text and '}' in text:
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    text = text[start:end]
                
                # Parse the JSON
                result = json.loads(text)
                
                # Validate the result has required fields
                if 'classification' in result and 'confidence' in result:
                    return result
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as error:
                if attempt == max_retries - 1:
                    break
                time.sleep(1)
        
        # Fallback: Try to extract information manually from the text
        try:
            response_text = response.text.lower()
            
            # Try to determine classification from keywords
            if 'valid' in response_text and 'fake' not in response_text:
                classification = "VALID"
            elif 'fake' in response_text or 'false' in response_text:
                classification = "FAKE"
            else:
                classification = "NOT_VERIFIABLE"
            
            return {
                "classification": classification,
                "confidence": 0.5,
                "explanation": "AI provided analysis but in non-standard format. Manual review recommended.",
                "key_points": ["Response format was non-standard", "Classification based on keyword analysis"]
            }
        except:
            # Ultimate fallback
            return {
                "classification": "NOT_VERIFIABLE",
                "confidence": 0.0,
                "explanation": "Unable to parse AI response after multiple attempts",
                "key_points": ["Technical error occurred", "Please try again"]
            }


class FakeNewsDetector:
    """
    This brings everything together! It searches, scrapes, analyzes,
    and gives you a verdict on any claim.
    """
    def __init__(self, api_key):
        self.searcher = WebSearchEngine()
        self.scraper = ContentScraper()
        self.rag = RAGPipeline()
        self.classifier = FakeNewsClassifier(api_key)
    
    def verify(self, claim, progress_callback=None):
        """
        The main method - fact-check any claim!
        """
        # Step 1: Generate smart search queries
        if progress_callback:
            progress_callback(0.1, "🤔 Creating search queries...")
        queries = self.classifier.generate_queries(claim)
        
        # Step 2: Search the web
        if progress_callback:
            progress_callback(0.2, "🌐 Searching the internet...")
        all_search_results = []
        for query in queries[:3]:  # Use top 3 queries
            results = self.searcher.search(query)
            all_search_results.extend(results)
        
        # Remove duplicate URLs
        unique_urls = list(set([r['url'] for r in all_search_results]))[:8]
        
        # Step 3: Scrape articles
        if progress_callback:
            progress_callback(0.4, f"📄 Downloading {len(unique_urls)} articles...")
        articles = self.scraper.scrape_multiple(unique_urls)
        
        if not articles:
            return {
                "classification": "NOT_VERIFIABLE", 
                "confidence": 0.0,
                "explanation": "Unable to find sufficient evidence online",
                "key_points": [],
                "sources": [],
                "num_sources": 0
            }
        
        # Step 4: Build knowledge base
        if progress_callback:
            progress_callback(0.6, "🧠 Building knowledge base...")
        index, chunks, metadata = self.rag.create_vectorstore(articles)
        
        # Step 5: Find relevant evidence
        if progress_callback:
            progress_callback(0.7, "🔎 Extracting relevant evidence...")
        relevant_evidence = self.rag.retrieve_context(index, chunks, metadata, claim)
        
        # Step 6: AI analysis
        if progress_callback:
            progress_callback(0.9, "⚖️ AI is analyzing the evidence...")
        verdict = self.classifier.classify(claim, relevant_evidence)
        
        # Add extra info to the result
        verdict['claim'] = claim
        verdict['sources'] = [
            {'url': article['url'], 'title': article['title']} 
            for article in articles
        ]
        verdict['num_sources'] = len(articles)
        
        if progress_callback:
            progress_callback(1.0, "✅ Analysis complete!")
        
        return verdict


# ========================================
# UI FUNCTIONS
# ========================================

def display_result(result):
    """Display the fact-check results"""
    classification = result.get('classification', 'UNKNOWN')
    confidence = result.get('confidence', 0.0)
    
    # Determine box style
    if classification == 'VALID':
        box_class = "valid-box"
        icon = "✅"
        verdict = "CLAIM IS VALID"
    elif classification == 'FAKE':
        box_class = "fake-box"
        icon = "❌"
        verdict = "CLAIM IS FAKE"
    else:
        box_class = "unverifiable-box"
        icon = "⚠️"
        verdict = "CANNOT BE VERIFIED"
    
    # Display result box
    st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
    
    st.markdown(f"## {icon} {verdict}")
    
    # Confidence bar
    st.markdown("### 📊 Confidence Level")
    confidence_percent = int(confidence * 100)
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_percent}%">
            {confidence_percent}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("### 💡 Explanation")
    st.write(result.get('explanation', 'No explanation available'))
    
    # Key points
    key_points = result.get('key_points', [])
    if key_points:
        st.markdown("### 🔑 Key Findings")
        for i, point in enumerate(key_points, 1):
            st.write(f"{i}. {point}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sources
    num_sources = result.get('num_sources', 0)
    st.markdown(f"### 📚 Sources Analyzed: {num_sources}")
    
    sources = result.get('sources', [])
    if sources:
        with st.expander("🔗 View All Sources"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**{i}. {source.get('title', 'Untitled')}**")
                st.markdown(f"🔗 [{source.get('url', '')}]({source.get('url', '')})")
                st.markdown("---")


# ========================================
# MAIN APP
# ========================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by DuckDuckGo & Gemini AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Show API key status
        if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            st.success("✅ API Key configured!")
        else:
            st.error("⚠️ Please set your API key in the code")
            st.code('GEMINI_API_KEY = "your-actual-key-here"')
        
        st.markdown("---")
        
        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. Enter your Gemini API key above
        2. Type or paste a claim to verify
        3. Click "Fact-Check This Claim"
        4. Wait 2-3 minutes for analysis
        5. Review the results
        """)
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses:
        - **DuckDuckGo** for web search
        - **Gemini AI** for analysis
        - **RAG** for evidence retrieval
        - Trusted sources for verification
        """)
        
        # History
        if st.session_state.history:
            st.markdown("---")
            st.header("📜 Recent Checks")
            for item in st.session_state.history[-5:]:
                with st.expander(f"{item['icon']} {item['claim'][:50]}..."):
                    st.write(f"**Result:** {item['classification']}")
                    st.write(f"**Confidence:** {item['confidence']:.0%}")
    
    # Main content area
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Claim input
        claim = st.text_area(
            "Enter a claim to fact-check:",
            height=100,
            placeholder="e.g., Vaccines cause autism, The Earth is flat, Coffee prevents cancer..."
        )
        
        # Fact-check button
        if st.button("🔍 Fact-Check This Claim", type="primary", use_container_width=True):
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
                st.error("❌ Please set your Gemini API key in the code at the top of the file!")
                st.code('GEMINI_API_KEY = "your-actual-key-here"')
            elif not claim.strip():
                st.warning("⚠️ Please enter a claim to verify")
            else:
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    # Run the detector
                    detector = FakeNewsDetector(st.session_state.api_key)
                    result = detector.verify(claim, progress_callback=update_progress)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    display_result(result)
                    
                    # Add to history
                    icon_map = {
                        'VALID': '✅',
                        'FAKE': '❌',
                        'NOT_VERIFIABLE': '⚠️'
                    }
                    st.session_state.history.append({
                        'claim': claim,
                        'classification': result['classification'],
                        'confidence': result['confidence'],
                        'icon': icon_map.get(result['classification'], '❓')
                    })
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name=f"fact_check_{int(time.time())}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("💡 Tip: Check your API key and internet connection")

if __name__ == "__main__":
    main()