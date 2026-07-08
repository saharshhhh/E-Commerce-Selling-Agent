import os
import json
import re
import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from rag_system import RAGSystem
from order_lookup import OrderLookup
from return_policy import check_return_eligibility
from datetime import date

load_dotenv()

class SupportAgent:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.rag = RAGSystem("sample_data/docs/")
        self.orders = OrderLookup("sample_data/orders.csv")
        self.model = "llama-3.3-70b-versatile"
        # Fallback models available on Groq
        self._fallback_models = [
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ]
        self.messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant."
        }
    ]
    
        self.last_order_id = None

    def _chat_completion(self, max_retries_per_model: int = 2, **kwargs):
        """Wrap client.chat.completions.create with retry + model fallback."""
        requested_model = kwargs.pop("model", self.model)
        models_to_try = [requested_model] + [
            m for m in self._fallback_models if m != requested_model
        ]

        last_error = None
        for model_name in models_to_try:
            for attempt in range(max_retries_per_model):
                try:
                    return self.client.chat.completions.create(model=model_name, **kwargs)
                except RateLimitError as e:
                    last_error = e
                    print(f"Rate limit on {model_name}: {e}")
                    wait_seconds = self._parse_retry_wait(e)
                    if wait_seconds <= 10:
                        print(f"Waiting {wait_seconds:.0f}s, retrying {model_name}...")
                        time.sleep(wait_seconds)
                    else:
                        print(f"{model_name} rate limited ({wait_seconds:.0f}s) - trying next model")
                        break  # skip remaining attempts on this model, move on

        if last_error is None:
            raise RuntimeError("No models were attempted in _chat_completion")
        raise last_error

    @staticmethod
    def _parse_retry_wait(error: RateLimitError) -> float:
        """Extract the wait time from a Groq rate-limit error."""
        match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2))
            return minutes * 60 + seconds + 1

        return 5.0
        
    def route_question(self, query: str) -> dict:
        prompt = f"""
        Analyze the current user query and determine the appropriate route:
        1. "knowledge": For general information about company policies (shipping, return conditions, COD limits, payment methods, etc.) that do not require looking up details of a specific order.
        2. "order_data": For questions about a specific, existing order (status, tracking, plain details, etc.) using an order ID, where NO policy judgment is needed.
        3. "both": If the query needs both general policy rules and specific order data.

        Rules:
        - If the user query contains a specific order ID (e.g., ORD1001), extract it into the "order_id" field.
        - If the user refers to a previous order from history using a pronoun or reference (e.g. "it", "that order", "same order", "status of that"), infer the order ID from the conversation history and place it in the "order_id" field.
        - If the query is about a general policy or rule (like "COD limits" or "shipping to USA"), even if it mentions the word "order", route it to "knowledge" and set "order_id" to null.
        - Any question asking whether an order CAN be returned, exchanged, refunded, or cancelled, or whether it is still ELIGIBLE/within a WINDOW, always needs "both" — it requires the order's category/date AND the policy rule for that category. Never route these as "order_data" alone, even though they name a specific order ID.

        Return ONLY a JSON object with this exact structure:
        {{
            "route": "knowledge" | "order_data" | "both",
            "order_id": string | null
        }}

        Current query:
        {query}
        """ 
        messages = self.messages + [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Note: Groq supports response_format but it must be enabled specifically for some models.
        # Llama 3 models support it.
        response = self._chat_completion(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        routing = json.loads(response.choices[0].message.content)

        order_id = routing.get("order_id")

        if order_id:
            self.last_order_id = order_id
        elif routing.get("route") in ["order_data", "both"]:
            routing["order_id"] = self.last_order_id
        else:
            routing["order_id"] = None

        return routing

    def handle_knowledge(self, query: str) -> str:
        results = self.rag.search(query, top_k=5)
        context = "\n\n".join([r['chunk']['text'] for r in results])

        prompt = f"""
        Answer the user's question using only the provided context. 
        Be direct and helpful. You are allowed to make simple, direct logical inferences (e.g. if the context says "We currently do not ship internationally" and the question is "Do you ship to USA?", you can infer that we do not ship to the USA since the USA is international, or if COD is only available under 5000 INR and the order is 6000 INR, COD is not available).

        If the provided context does not contain enough information to answer the question, say "I'm sorry, I don't have that information."

        Context:
        {context}

        Question: {query}
        """

        response = self._chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        reply = response.choices[0].message.content
        return reply

    def handle_order_data(self, query: str, order_id: str) -> str:
        if not order_id:
            return "Please provide an order ID so I can look that up for you."

        order_details = self.orders.get_order_details(order_id)
        if not order_details:
            return f"I'm sorry, I couldn't find any order with ID {order_id}."

        prompt = f"""
        The user is asking: {query}
        Here are the order details I found:
        {json.dumps(order_details, indent=2)}

        Provide a concise and helpful answer to the user.
        """

        response = self._chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        reply = response.choices[0].message.content
        return reply

    def handle_both(self, query: str, order_id: str) -> str:
        if not order_id:
            return self.handle_knowledge(query)

        order_details = self.orders.get_order_details(order_id)
        if not order_details:
             return f"I found some policy information but I couldn't find order {order_id}."

        category = order_details.get("category", "")
        # Search with the category injected so the right policy chunk ranks highly,
        # not just the raw user phrasing (which may not mention "Electronics" at all).
        rag_results = self.rag.search(f"{category} return policy window {query}", top_k=3)
        context = "\n\n".join([r['chunk']['text'] for r in rag_results])

        today = date.today()
        elig = check_return_eligibility(order_details, today)

        facts = f"""
        - Product category: {elig['category']}
        - Order status: {elig['status']}
        - Order date: {elig['order_date']}
        - Return window for this category: {elig['window_days']} days
        - Return deadline: {elig['deadline']}
        - Eligible for return today ({today}): {elig['eligible']}
        """
        if elig["reason"]:
            facts += f"        - Note: {elig['reason']}\n"

        prompt = f"""
        You are a support agent. A user has a question that involves both their order and company policy.

        Order Details:
        {json.dumps(order_details, indent=2)}

        Relevant Policy Information (for wording/context only):
        {context}

        Pre-computed facts (ground truth — already correctly calculated, do NOT recompute,
        override, or invent any different window length, date, or eligibility result):
        {facts}

        User Question: {query}

        Write a clear, direct answer using ONLY the pre-computed facts above for the window
        length, dates, and eligibility. Do not use outside knowledge or typical industry
        return windows. Mention:
        1. The product name and its category.
        2. The return window that applies to this category.
        3. Whether they are eligible now, referencing the actual deadline date.
        """

        response = self._chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        reply = response.choices[0].message.content
        return reply

    def add_user_message(self, query: str):
        self.messages.append({
            "role": "user",
            "content": query
        })

        # Keep  only recent history
        if len(self.messages) > 12:
            self.messages = [self.messages[0]] + self.messages[-11:]


    def add_assistant_message(self, reply: str):
        self.messages.append({
            "role": "assistant",
            "content": reply
        })
    
        if len(self.messages) > 12:
            self.messages = [self.messages[0]] + self.messages[-11:]

    # Keywords that always require the policy-grounded eligibility path,
    # even if the router misclassifies the query as plain order_data.
    _ELIGIBILITY_KEYWORDS = ("return", "refund", "exchange", "cancel")

    def ask(self, query: str) -> str:
        # Route first, so routing LLM only sees true conversation history
        routing = self.route_question(query)

        route = routing.get("route")
        order_id = routing.get("order_id")

        # Deterministic backstop: never let a misrouted "order_data" classification
        # skip policy grounding for eligibility-type questions. Cheap insurance
        # against router mistakes, doesn't depend on the classifier being perfect.
        if order_id and route == "order_data" and any(
            kw in query.lower() for kw in self._ELIGIBILITY_KEYWORDS
        ):
            route = "both"

        if route == "order_data":
            answer = self.handle_order_data(query, order_id)

        elif route == "both":
            answer = self.handle_both(query, order_id)

        else:
            answer = self.handle_knowledge(query)

        # Now append to history
        self.add_user_message(query)
        self.add_assistant_message(answer)

        return answer

if __name__ == "__main__":
    agent = SupportAgent()
    print("Agent: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # print(f"today's date:{date.today()}")
        print(f"Agent: {agent.ask(user_input)}")
