import os
import pandas as pd
import tiktoken
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
    read_indexer_relationships,
    read_indexer_covariates,
    read_indexer_text_units
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
import asyncio
import concurrent.futures
# todo
# wyczyszczenie i ujednolicenie kodu - local i global search są napisane trochę inaczej
# poszukaj co robią dane paramery i wyczyść kod.
# dodatkowo pomyśl czy da się lepiej zrobić async. Czy można usunąć nest_asyncio.apply(), jeżeli nie ma notebooka?
#
# dodaj parametr do wyboru modelu dla embeddingu

class SearchQuery(BaseModel):
    """Schema for search queries"""
    query: str = Field(description="The search query to execute")

class GraphRAG:
    def __init__(self, config: Dict[str, Any]):
        if not self._validate_config(config):
            raise ValueError("Invalid configuration provided")
            
        self.config = config
        self._initialize_components()

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> bool:
        required_keys = ['openai_api_key', 'small_llm_model']
        return all(key in config for key in required_keys)

    def _setup_llm_and_embeding(self) -> ChatOpenAI:
        """Initialize LLM with configuration"""
        self.llm = ChatOpenAI(
            api_key=self.config['openai_api_key'],
            model=self.config['small_llm_model'],
            api_base=self.config.get('api_base_url', "https://api.openai.com/v1"),
            api_type=OpenaiApiType.OpenAI,
            max_retries=3
        )
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.text_embedder = OpenAIEmbedding(
            api_key=self.config['openai_api_key'],
            api_base=None,
            api_type=OpenaiApiType.OpenAI,
            model=self.config.get('embedding_model', "text-embedding-3-small"),
            deployment_name=self.config.get('embedding_model', "text-embedding-3-small"),
            max_retries=20,
        )
 

    def _initialize_components(self) -> None:
            """Initialize all RAG components"""
            try:
                self._setup_llm_and_embeding()

                input_dir = self.config.get('input_dir', 'rag')
                community_level = self.config.get('community_level', 0)
                
                # Load dataframes
                dfs = self._load_dataframes(input_dir)
                
                # Setup RAG components
                self._setup_rag_components(dfs, community_level, input_dir)
                
                # Initialize search engines
                self.search_engines = self._setup_search_engine()
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def _load_dataframes(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        return {
            'community': pd.read_parquet(f"{input_dir}/output/create_final_communities.parquet"),
            'entity': pd.read_parquet(f"{input_dir}/output/create_final_nodes.parquet"),
            'report': pd.read_parquet(f"{input_dir}/output/create_final_community_reports.parquet"),
            'embedding': pd.read_parquet(f"{input_dir}/output/create_final_entities.parquet"),
            'relationship': pd.read_parquet(f"{input_dir}/output/create_final_relationships.parquet"),
            'text_unit': pd.read_parquet(f"{input_dir}/output/create_final_text_units.parquet")
        }

    def _setup_rag_components(self, dfs: Dict[str, pd.DataFrame], community_level: int, input_dir: str) -> None:
        self.communities = read_indexer_communities(dfs['community'], dfs['entity'], dfs['report'])
        self.reports = read_indexer_reports(dfs['report'], dfs['entity'], community_level)
        self.entities = read_indexer_entities(dfs['entity'], dfs['embedding'], community_level)
        self.relationships = read_indexer_relationships(dfs['relationship'])
        self.text_units = read_indexer_text_units(dfs['text_unit'])
        
        vector_store_path = os.path.join(input_dir, 'output', 'lancedb')
        os.makedirs(vector_store_path, exist_ok=True)

        # Setup vector store
        self.description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description"
        )
        self.description_embedding_store.connect(db_uri=vector_store_path)

    def _setup_search_engine(self) -> Dict[str, Any]:
        global_context = GlobalCommunityContext(
            community_reports=self.reports,
            communities=self.communities,
            entities=self.entities,
            token_encoder=self.token_encoder,
        )

        local_context = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
            covariates = None
        )

        return {
            'global': GlobalSearch(
                llm=self.llm,
                context_builder=global_context,
                token_encoder=self.token_encoder,
                max_data_tokens=6000,
            ),
            'local': LocalSearch(
                llm=self.llm,
                context_builder=local_context,
                token_encoder=self.token_encoder,
                llm_params={"max_tokens": 2000, "temperature": 0.0},
                context_builder_params={
                    "text_unit_prop": 0.5,
                    "community_prop": 0.1,
                    "top_k_mapped_entities": 5,
                    "top_k_relationships": 5,
                    "max_tokens": 6000,
                },
                response_type="multiple paragraphs",
            )
        }

    def global_search(self, query: str) -> str:
        """Execute global search synchronously by running the coroutine in a separate thread."""
        engine = self.search_engines['global']
        result = run_async_in_thread(engine.asearch(query))
        return result.response

    def local_search(self, query: str, entity_name: Optional[str] = None, entity_id: Optional[str] = None) -> str:
        """Execute local search synchronously by running the coroutine in a separate thread."""
        engine = self.search_engines['local']
        result = run_async_in_thread(engine.asearch(query))
        return result.response

    def get_search_tools(self) -> List[StructuredTool]:
        """Returns separate tools for global and local search"""
        return [
            StructuredTool.from_function(
                func=self.global_search,
                name="global_search",
                description="Search across entire knowledge base",
                args_schema=SearchQuery,
                return_direct=False
            ),
            StructuredTool.from_function(
                func=self.local_search,
                name="local_search",
                description="Search detailed information about specific entity.",
                args_schema=SearchQuery,
                return_direct=False
            )
        ]
    
def run_async_in_thread(coro):
    """Run an async coroutine in a new thread and wait for the result."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
    
