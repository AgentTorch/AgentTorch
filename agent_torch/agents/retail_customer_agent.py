import random
import torch
import openai  # Import OpenAI library
from agent_torch.agents.base_agent import BaseAgent
import os

# Set OpenAI API key from environment variable
openai.api_key = "Add your API key here"

class RetailCustomerAgent(BaseAgent):
    def __init__(self, agent_id, preferences, budget):
        super().__init__(agent_id)
        self.preferences = preferences  # Dictionary of product preferences
        self.budget = budget  # Available spending budget
        self.cart = {}  # Items the agent intends to purchase
        self.reorders = 0

    @staticmethod
    def generate_budget():
        """Generate a random budget for the agent."""
        return random.uniform(50, 150)

    @staticmethod
    def generate_preferences(products):
        """Generate a random preference for each product."""
        preferences = {}
        for product in products:
            preferences[product.product_id] = random.uniform(0, 1)  # Random preference between 0 and 1
        return preferences

    def query_llm_for_decision(self, product_info):
        """Use OpenAI API to make a decision based on product information."""
        prompt = f"""
        You are a customer trying to decide whether to buy the following product:
        Product: {product_info['product_category']}
        Price: {product_info['price']}
        Preference: {product_info['preference']}
        Budget: {product_info['budget']}
        Promotion: {product_info['promotion']}
        
        Should you buy this product? Respond with 'yes' or 'no' and explain briefly.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            decision_text = response['choices'][0]['message']['content'].strip()
            return decision_text.lower().startswith('yes')  # Return True for 'yes', False otherwise
        except Exception as e:
            print(f"Error querying OpenAI API: {e}")
            return False  # Default to 'no' if there's an error


    def decide_purchases(self, products, promotions, current_step):
        """Decide what the agent purchases using LLM."""
        for product in products:
            product_id = product.product_id  # Assuming product has a product_id attribute
            preference = self.preferences.get(product_id, 0)
            promotion = promotions.get(product_id, None)
            price = product.price

            # Apply promotion if available
            promotion_info = "none"
            if promotion and promotion.is_active(current_step):
                price *= (1 - promotion.discount_rate)
                promotion_info = f"{promotion.discount_rate * 100}% discount applied"

            # Prepare product information for LLM decision
            product_info = {
                'product_category': product.category,  # Assuming product has just a category no name
                'price': price,
                'preference': preference,
                'budget': self.budget,
                'promotion': promotion_info
            }

            # Ask the LLM to decide if the agent should buy the product
            should_buy = self.query_llm_for_decision(product_info)

            # Purchase the product if the LLM says 'yes' and budget allows
            if should_buy and self.budget >= price:
                self.cart[product_id] = 1  # Purchase one unit
                self.budget -= price
                print(f"Product {product.category} added to cart. Budget remaining: {self.budget}")
            else:
                print(f"Product {product.category} not purchased.")
