from agent_torch.agents.retail_customer_agent import RetailCustomerAgent
from agent_torch.forecasting.demand_forecasting import forecast_demand

def run_retail_simulation(num_agents, num_steps, products, promotions):
    agents = [
        RetailCustomerAgent(
            agent_id=i,
            preferences=RetailCustomerAgent.generate_preferences(products),
            budget=RetailCustomerAgent.generate_budget()
        )
        for i in range(num_agents)
    ]

    sales = {product.product_id: 0 for product in products}  # Initialize sales

    for step in range(num_steps):
        active_promotions = {promo.product_id: promo for promo in promotions if promo.is_active(step)}
        
        for agent in agents:
            agent.decide_purchases(products, active_promotions, step)
        
        # Track sales (how much each product is sold)
        for agent in agents:
            for product_id in agent.cart:
                sales[product_id] += agent.cart[product_id]

        # Reset the cart for the next step
        for agent in agents:
            agent.cart = {}

    return sales  # Return sales data
