"""SQLite-based caching for LLM responses."""

import sqlite3
import json
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path

class LLMCache:
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store the SQLite database
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    system_prompt TEXT,
                    prompt_template TEXT,
                    context_hash TEXT,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Add index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                ON llm_cache(model_name, system_prompt, prompt_template, context_hash)
            """)
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_cache_key(self, 
                      model_name: str,
                      system_prompt: str,
                      prompt_template: str,
                      context: str) -> str:
        """Generate a cache key from the input parameters."""
        components = [
            model_name,
            system_prompt,
            prompt_template,
            self._compute_hash(context)
        ]
        return self._compute_hash("|".join(components))
    
    def get(self,
            model_name: str,
            system_prompt: str,
            prompt_template: str,
            context: str) -> Optional[str]:
        """Get cached response if it exists.
        
        Args:
            model_name: Name of the LLM model
            system_prompt: System prompt used
            prompt_template: Template used for the prompt
            context: Context provided to the model
            
        Returns:
            Cached response if found, None otherwise
        """
        cache_key = self._get_cache_key(
            model_name, system_prompt, prompt_template, context
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()
            
        return result[0] if result else None
    
    def set(self,
            model_name: str,
            system_prompt: str,
            prompt_template: str,
            context: str,
            response: str):
        """Cache a response.
        
        Args:
            model_name: Name of the LLM model
            system_prompt: System prompt used
            prompt_template: Template used for the prompt
            context: Context provided to the model
            response: Response to cache
        """
        cache_key = self._get_cache_key(
            model_name, system_prompt, prompt_template, context
        )
        context_hash = self._compute_hash(context)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO llm_cache 
                (cache_key, model_name, system_prompt, prompt_template, context_hash, response)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cache_key, model_name, system_prompt, prompt_template, context_hash, response))
    
    def clear(self):
        """Clear all cached responses."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM llm_cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) as count 
                FROM llm_cache 
                GROUP BY model_name
            """)
            model_counts = dict(cursor.fetchall())
            
            cursor = conn.execute("""
                SELECT MIN(created_at) as oldest, MAX(created_at) as newest 
                FROM llm_cache
            """)
            oldest, newest = cursor.fetchone()
        
        return {
            "total_entries": total_entries,
            "model_counts": model_counts,
            "oldest_entry": oldest,
            "newest_entry": newest
        } 