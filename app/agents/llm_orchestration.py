

import os
import json
import time
import requests
import concurrent.futures
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from Utils.logger import setup_logger
from Utils.token_counter import count_tokens  
from agents.vector_db import VectorDB  # ✅ Now using ChromaDB instead of Pinecone

# ✅ Logger Setup
logger = setup_logger()
logger.info("✅ Logger setup completed successfully.")

class LLMOrchestrator:
    def __init__(self, CONFIG_PATH="config/Configration.json"):
        """Load API keys, model selection, and initialize the Orchestrator with VectorDB."""
        self.logger = setup_logger()

        # ✅ Load Configuration
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)

        self.api_keys = self.config["api_keys"]
        self.models = self.config["models"]
        self.generation_config = self.config["generation_config"]
        self.fallback_models = ["claude-2", "mixtral-8x7b-32768"]

        # ✅ Initialize ChromaDB for memory storage (Hugging Face embeddings)
        self.memory = VectorDB(persist_directory="db_memory")

    def select_model(self, query: str) -> str:
        """Dynamically selects the best LLM based on query intent."""
        q_lower = query.lower()

        if any(word in q_lower for word in ['summarize', 'key points', 'tl;dr']):
            return "llama3-8b-8192"
        elif any(word in q_lower for word in ['code', 'script', 'program', 'scrape', 'scraping']):
            return "qwen-2.5-coder-32b"
        else:
            return "mixtral-8x7b-32768"

    def call_model(self, model_name: str, query: str) -> str:
        """Calls the selected LLM API dynamically with fallback handling."""
        retries = 2  # Maximum retry attempts
        for attempt in range(retries + 1):
            start_time = time.time()
            api_key = self.api_keys["groq"] if "claude" not in model_name else self.api_keys["anthropic"]

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # ✅ API URL and Payload Setup
            if "claude" in model_name:
                url = "https://api.anthropic.com/v1/messages"
                payload = {
                    "model": model_name,
                    "max_tokens": 512,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": query}]
                }
            else:
                url = "https://api.groq.com/openai/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 512,
                    "temperature": 0.1
                }

            try:
                response = requests.post(url, json=payload, headers=headers)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    response_data = response.json()
                    output_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response generated.")

                    if not output_text.strip():
                        output_text = "⚠️ No meaningful response. Try rephrasing your query."

                    # ✅ Token calculations
                    input_tokens = count_tokens(query, model_name)
                    output_tokens = count_tokens(output_text, model_name)
                    total_tokens = input_tokens + output_tokens
                    cost_per_1000_tokens = self.config["pricing_per_1000_tokens"].get(model_name, 0.01)
                    total_cost = (total_tokens / 1000) * cost_per_1000_tokens

                    self.logger.info(
                        f"[Success] Model: {model_name}, Response Time: {response_time:.2f}s\n"
                        f"• Input Tokens: {input_tokens}\n"
                        f"• Output Tokens: {output_tokens}\n"
                        f"• Total Tokens: {total_tokens}\n"
                        f"• Estimated Cost: ${total_cost:.4f}\n"
                    )

                    # ✅ Store query-response pair in memory
                    self.memory.store_interaction(query, output_text)

                    return output_text

                else:
                    self.logger.warning(f"[Failure] Model: {model_name}, Status: {response.status_code}, Response: {response.text}")

            except Exception as e:
                self.logger.error(f"Error calling {model_name}: {str(e)}")

            # ✅ Switch to fallback model if the primary model fails
            if attempt < len(self.fallback_models):
                model_name = self.fallback_models[attempt]
                self.logger.info(f"[Retry] Switching to fallback model: {model_name}")

        return "⚠️ API request failed. Please try again later."

    def generate_response(self, query: str) -> str:
        """Selects the best model and generates response while retrieving memory for context."""
        
        selected_model = self.select_model(query)
        self.logger.info(f"[Routing] Query: {query} → Selected Model: {selected_model}")

        # ✅ Retrieve similar past interactions
        past_responses = self.memory.retrieve_similar(query)
        if past_responses:
            self.logger.info(f"[Memory] Found similar past responses: {past_responses}")
            query = f"Past responses: {past_responses} \n\n Current Query: {query}"  # Enhance query with memory context

        # ✅ Generate new response
        response = self.call_model(selected_model, query)

        # ✅ Store new interaction in memory
        self.memory.store_interaction(query, response)

        return response  # ✅ Ensure function returns response correctly

    def batch_process_queries(self, queries: list) -> dict:
        """Handles multiple queries in parallel for efficiency."""
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {executor.submit(self.generate_response, query): query for query in queries}

            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results[query] = future.result()
                except Exception as e:
                    results[query] = f"Error: {str(e)}"

        return results
