import os
import json
# pyrefly: ignore [missing-import]
from groq import Groq
from dotenv import load_dotenv
from rag_system import RAGSystem
from order_lookup import OrderLookup

load_dotenv()

class SupportAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.rag = RAGSystem("sample_data/docs/")
        self.orders = OrderLookup("sample_data/orders.csv")
        self.model = "llama-3.3-70b-versatile"

    def route_question(self, query: str) -> dict:
        """Decide whether to use RAG, Order Lookup, or both"""
        prompt = f"""
        Analyze the following user query and decide if it needs:
        1. "knowledge": Information about company policies (shipping, returns, etc.)
        2. "order_data": Information about a specific order (status, details, etc.)
        3. "both": Needs both policy info and order data.

        If it's about an order, extract the order_id (e.g., ORD1001).

        Return a JSON object with:
        "route": "knowledge" | "order_data" | "both",
        "order_id": string | null

        Query: {query}
        """

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def handle_knowledge(self, query: str) -> str:
        results = self.rag.search(query, top_k=5)
        context = "\n\n".join([r['chunk']['text'] for r in results])

        prompt = f"""
        Answer the following question using the provided context.
        If the answer is not in the context, say "I'm sorry, I don't have that information."

        Context:
        {context}

        Question: {query}
        """

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return response.choices[0].message.content

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

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return response.choices[0].message.content

    def handle_both(self, query: str, order_id: str) -> str:
        if not order_id:
            return self.handle_knowledge(query)

        order_details = self.orders.get_order_details(order_id)
        if not order_details:
             return f"I found some policy information but I couldn't find order {order_id}."

        rag_results = self.rag.search(query)
        context = "\n\n".join([r['chunk']['text'] for r in rag_results])

        prompt = f"""
        You are a support agent. A user has a question that involves both their order and company policy.

        Order Details:
        {json.dumps(order_details, indent=2)}

        Relevant Policy Information:
        {context}

        User Question: {query}

        Reason through the situation and provide a complete answer.
        For example, if they want to return an item, check if the delivery date and category comply with the return policy.
        Today's date is 2026-07-05.
        """

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return response.choices[0].message.content

    def ask(self, query: str) -> str:
        routing = self.route_question(query)
        route = routing.get("route")
        order_id = routing.get("order_id")

        if route == "order_data":
            return self.handle_order_data(query, order_id)
        elif route == "both":
            return self.handle_both(query, order_id)
        else:
            return self.handle_knowledge(query)

if __name__ == "__main__":
    agent = SupportAgent()
    print("Agent: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(f"Agent: {agent.ask(user_input)}")
