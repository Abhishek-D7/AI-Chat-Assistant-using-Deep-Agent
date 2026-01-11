from app.config import Config
import uuid
import asyncio
from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec

# We need both Async (for data ops) and Sync (for inference, as plugin might be sync only or we wrap it)
# Actually, the Pinecone client unified structure allows us to use 'pc.inference.embed'
# But 'PineconeAsyncio' might not have 'inference' plugin support directly out of the box in the same way?
# Let's use the standard Pinecone client for inference generation, then async for upsert if needed, 
# or just use sync for inference -> async for upsert.
# Since we are in an async function, we can await loop.run_in_executor for the sync inference call if needed.

class PineconeMemory:
    def __init__(self):
        self.api_key = Config.PINECONE_API_KEY
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not set in Config")
        
        self.index_name = Config.PINECONE_INDEX_NAME
        # Use simple synchronous client for inference operations
        self.pc_sync = Pinecone(api_key=self.api_key)
        
        # Model to use
        self.model = "multilingual-e5-large" 

    async def _ensure_index(self):
        """Checks if index exists, create if not (using integrated model)."""
        # Checks using sync client
        existing = await asyncio.to_thread(self.pc_sync.list_indexes)
        existing_names = [i.name for i in existing]
        
        if self.index_name not in existing_names:
            print(f"Creating index {self.index_name} with integrated embedding model...")
            await asyncio.to_thread(
                 self.pc_sync.create_index_for_model,
                 name=self.index_name,
                 cloud="aws",
                 region="us-east-1",
                 embed={
                     "model": "llama-text-embed-v2",
                     "field_map": {"text": "chunk_text"}
                 }
            )

    async def add_memory(self, user_id: str, text: str):
        """Adds memory using text-based upsert (Pinecone generates embedding)."""
        memory_id = str(uuid.uuid4())
        
        await self._ensure_index()
        
        # Use upsert_records
        index = self.pc_sync.Index(self.index_name)
        
        record = {
            "id": memory_id,
            "chunk_text": text,
            "user_id": user_id,
            "text": text
        }
        
        await asyncio.to_thread(
            index.upsert_records,
            namespace="default", 
            records=[record]
        )
            
        print(f"Memory added for user {user_id}: {text}")

    async def search_memory(self, user_id: str, query: str, k: int = 3):
        """Searches memory using text query (Pinecone generates embedding)."""
        
        # Ensure index exists before searching to avoid 404 on first run
        await self._ensure_index()
        
        index = self.pc_sync.Index(self.index_name)
        
        try:
            # Use search_records
            resp = await asyncio.to_thread(
                index.search_records,
                namespace="default",
                query={
                    "inputs": {"text": query},
                    "top_k": k,
                    "filter": {"user_id": user_id}
                },
                fields=["text", "chunk_text", "user_id"] # Return fields
            )
            
            # Extract text from response
            # Response format: {'result': {'hits': [...]}} or similar
            # Iterate hits
            hits = resp.get('result', {}).get('hits', [])
            memories = []
            for hit in hits:
                fields = hit.get('fields', {})
                if 'text' in fields:
                    memories.append(fields['text'])
                elif 'chunk_text' in fields:
                    memories.append(fields['chunk_text'])
            
            return memories
            
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []

if __name__ == "__main__":
    # Test (Async)
    import asyncio
    async def main():
        try:
            mem = PineconeMemory()
            # await mem.add_memory("user123", "I love coffee.")
            # print(await mem.search_memory("user123", "What do I like?"))
            print("Pinecone Memory Initialized Successfully (Mock)")
        except Exception as e:
            print(f"Initialization Failed: {e}")
    asyncio.run(main())
