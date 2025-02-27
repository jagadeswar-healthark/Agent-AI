# # app/agent/research_agent.py

# # app/agents/research_agent.py

# import os
# import json
# import time
# import requests
# import re  # For extracting category from response
# from langchain.prompts import PromptTemplate
# from Utils.logger import setup_logger
# from Utils.token_counter import count_tokens


# def extract_category(response: str) -> str:
#     """Extracts only the category name from the API response."""
#     match = re.search(r"(Tree[-\s]of[-\s]Thought|Self[-\s]Consistency|Chain[-\s]of[-\s]Thought)", response, re.IGNORECASE)
#     if match:
#         return match.group(1).lower().replace("-", "_").replace(" ", "_")
#     return "chain_of_thought"  # Default fallback if classification fails


# class ResearchAgent:
#     def __init__(self, CONFIG_PATH=r"C:\Users\HP\Desktop\Agent-AI\config\Configration.json"):
#         """Initialize Research Agent with Groq API for structured research-based LLM calls."""
#         self.logger = setup_logger()

#         with open(CONFIG_PATH, "r") as f:
#             self.config = json.load(f)

#         self.api_key = self.config["api_keys"]["groq"]  # ✅ Use Groq API Key
#         self.model_name = "mixtral-8x7b-32768"  # ✅ Default Model
#         self.base_url = "https://api.groq.com/openai/v1/chat/completions"

#         # ✅ Define Prompt Templates
#         self.chain_of_thought_template = PromptTemplate(
#             input_variables=["query"],
#             template="Let's think step by step to answer:\n{query}\n\nStep 1:"
#         )
#         self.self_consistency_template = PromptTemplate(
#             input_variables=["query"],
#             template="Provide multiple perspectives for:\n{query}\n\nAnswer 1:"
#         )
#         self.tree_of_thought_template = PromptTemplate(
#             input_variables=["query"],
#             template="Break down into sub-questions and synthesize response:\n{query}\n\nSub-question 1:"
#         )

#         self.refinement_template = PromptTemplate(
#             input_variables=["query", "initial_response"],
#             template=(
#                 "Based on previous answer, refine response:\n\n"
#                 "Query: {query}\n\n"
#                 "Initial Answer: {initial_response}\n\n"
#                 "Refined Answer:"
#             )
#         )

#     def choose_prompt_style(self, query: str) -> str:
#         """Classify query to select best prompting style dynamically. Uses Groq API first, then falls back to manual classification."""
    
#         classification_prompt = f"""
#         You are an expert classifier for reasoning styles. Choose the best option:

#         1. **Tree-of-Thought** – When the question requires breaking down into **sub-questions** before answering.
#         2. **Self-Consistency** – When the question requires weighing **multiple perspectives** before making a decision.
#         3. **Chain-of-Thought** – When the question requires **step-by-step reasoning** to reach an answer.

#         **Response should contain ONLY one of these categories:**
#         - Tree-of-Thought
#         - Self-Consistency
#         - Chain-of-Thought

#         Query: "{query}"

#         Category:
#         """

#         # ✅ Call Groq API
#         raw_response = self.call_groq_api(classification_prompt)
#         self.logger.info(f"[DEBUG] Full API Response from Groq: {raw_response}")

#         # ✅ Extract only the category name
#         style_response = extract_category(raw_response)

#         # ✅ If Groq returns chain_of_thought too often, apply manual classification
#         if style_response == "chain_of_thought":
#             style_response = self.manual_classification(query)
#             self.logger.info(f"[Fallback] Applying manual classification: '{style_response}'")

#         self.logger.info(f"[Classification] Final Selected Style: '{style_response}'")
        
#         return style_response


#     def manual_classification(self, query: str) -> str:
#         """Manually classifies query if Groq fails or returns 'Chain-of-Thought' too often."""

#         query_lower = query.lower()

#         # ✅ **Tree-of-Thought** (Complex Multi-Part Queries)
#         if any(keyword in query_lower for keyword in ["compare", "contrast", "differences", "break down", "steps"]):
#             return "tree_of_thought"

#         # ✅ **Self-Consistency** (Decision-Making, Evaluations, Multiple Perspectives)
#         elif any(keyword in query_lower for keyword in ["best option", "should I choose", "pros and cons", "evaluate", "compare opinions"]):
#             return "self_consistency"

#         # ✅ Default to **Chain-of-Thought** if nothing else matches
#         return "chain_of_thought"

#     def call_groq_api(self, query: str) -> str:
#         """Calls the Groq API and returns response."""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": query}],
#             "max_tokens": 512,
#             "temperature": 0.1
#         }

#         try:
#             response = requests.post(self.base_url, json=payload, headers=headers)

#             if response.status_code == 200:
#                 return response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response generated.")
#             else:
#                 self.logger.error(f"Groq API Failure: {response.status_code} {response.text}")
#                 return "⚠️ API request failed. Try again later."

#         except Exception as e:
#             self.logger.error(f"Groq API Error: {str(e)}")
#             return "⚠️ API request error."

#     def generate_response(self, query: str):
#         """Generate structured response using selected prompt style with token tracking."""

#         style = self.choose_prompt_style(query)
#         formatted_query = getattr(self, f"{style}_template").format(query=query)

#         # ✅ Count input tokens for initial request
#         input_tokens_initial = count_tokens(formatted_query, self.model_name)

#         # ✅ Get Initial Response
#         initial_response = self.call_groq_api(formatted_query)

#         # ✅ Count output tokens for initial response
#         output_tokens_initial = count_tokens(initial_response, self.model_name)

#         # ✅ Prepare Refined Query
#         refined_query = self.refinement_template.format(query=query, initial_response=initial_response)

#         # ✅ Count input tokens for refined request
#         input_tokens_refined = count_tokens(refined_query, self.model_name)

#         # ✅ Get Refined Response
#         refined_response = self.call_groq_api(refined_query)

#         # ✅ Count output tokens for refined response
#         output_tokens_refined = count_tokens(refined_response, self.model_name)

#         # ✅ Compute Total Tokens
#         total_input_tokens = input_tokens_initial + input_tokens_refined
#         total_output_tokens = output_tokens_initial + output_tokens_refined
#         total_tokens = total_input_tokens + total_output_tokens

#         model_info = {
#             "model": style,
#             "total_input_tokens": total_input_tokens,
#             "total_output_tokens": total_output_tokens,
#             "total_tokens": total_tokens
#         }

#         # ✅ Log Token Usage
#         self.logger.info(
#             f"[Token Usage]\n"
#             f"• Prompting Style: {style}\n"
#             f"• Input Tokens (Initial + Refined): {total_input_tokens}\n"
#             f"• Output Tokens (Initial + Refined): {total_output_tokens}\n"
#             f"• Total Tokens: {total_tokens}\n"
#         )

#         return refined_response, model_info

# app/agent/research_agent.py

# app/agents/research_agent.py


import os
import json
import time
import requests
from langchain.prompts import PromptTemplate
from Utils.logger import setup_logger
from Utils.token_counter import count_tokens

class ResearchAgent:
    def __init__(self, CONFIG_PATH=r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"):
        """Initialize Research Agent with Groq API for structured research-based LLM calls."""
        self.logger = setup_logger()

        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)

        self.api_key = self.config["api_keys"]["groq"]  # ✅ Use Groq API Key
        self.model_name = "mixtral-8x7b-32768"  # ✅ Default Model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

        # ✅ Define Prompt Templates
        self.chain_of_thought_template = PromptTemplate(
            input_variables=["query"],
            template="Let's think step by step to answer:\n{query}\n\nStep 1:"
        )
        self.self_consistency_template = PromptTemplate(
            input_variables=["query"],
            template="Provide multiple perspectives for:\n{query}\n\nAnswer 1:"
        )
        self.tree_of_thought_template = PromptTemplate(
            input_variables=["query"],
            template="Break down into sub-questions and synthesize response:\n{query}\n\nSub-question 1:"
        )

        self.refinement_template = PromptTemplate(
            input_variables=["query", "initial_response"],
            template=(
                "Based on previous answer, refine response:\n\n"
                "Query: {query}\n\n"
                "Initial Answer: {initial_response}\n\n"
                "Refined Answer:"
            )
        )

    def choose_prompt_style(self, query: str) -> str:
        """Classify query to select best prompting style."""
        classification_prompt = f"""
        Classify the following query into one of the three categories:

        1. Tree-of-Thought
        2. Self-Consistency
        3. Chain-of-Thought

        Clearly respond with ONLY the category name.

        Query: "{query}"

        Category:
        """

        style_response = self.call_groq_api(classification_prompt).strip().lower()

        if "tree-of-thought" in style_response:
            return "tree_of_thought"
        elif "self-consistency" in style_response:
            return "self_consistency"
        elif "chain-of-thought" in style_response:
            return "chain_of_thought"
        else:
            self.logger.warning(f"Unknown Style: '{style_response}'. Defaulting to Chain-of-Thought.")
            return "chain_of_thought"

    def call_groq_api(self, query: str) -> str:
        """Calls the Groq API and returns response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 512,
            "temperature": 0.1
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)

            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response generated.")
            else:
                self.logger.error(f"Groq API Failure: {response.status_code} {response.text}")
                return "⚠️ API request failed. Try again later."

        except Exception as e:
            self.logger.error(f"Groq API Error: {str(e)}")
            return "⚠️ API request error."

    def generate_response(self, query: str) -> str:
        """Generate structured response using selected prompt style with token tracking."""

        style = self.choose_prompt_style(query)
        formatted_query = getattr(self, f"{style}_template").format(query=query)

        # ✅ Count input tokens for initial request
        input_tokens_initial = count_tokens(formatted_query, self.model_name)

        # ✅ Get Initial Response
        initial_response = self.call_groq_api(formatted_query)

        # ✅ Count output tokens for initial response
        output_tokens_initial = count_tokens(initial_response, self.model_name)

        # ✅ Prepare Refined Query
        refined_query = self.refinement_template.format(query=query, initial_response=initial_response)

        # ✅ Count input tokens for refined request
        input_tokens_refined = count_tokens(refined_query, self.model_name)

        # ✅ Get Refined Response
        refined_response = self.call_groq_api(refined_query)

        # ✅ Count output tokens for refined response
        output_tokens_refined = count_tokens(refined_response, self.model_name)

        # ✅ Compute Total Tokens
        total_input_tokens = input_tokens_initial + input_tokens_refined
        total_output_tokens = output_tokens_initial + output_tokens_refined
        total_tokens = total_input_tokens + total_output_tokens

        # ✅ Log Token Usage
        self.logger.info(
            f"[Token Usage]\n"
            f"• Prompting Style: {style}\n"
            f"• Input Tokens (Initial + Refined): {total_input_tokens}\n"
            f"• Output Tokens (Initial + Refined): {total_output_tokens}\n"
            f"• Total Tokens: {total_tokens}\n"
        )

        return refined_response
