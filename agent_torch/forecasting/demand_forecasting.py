def forecast_demand(agents, products):
    """
    Forecast demand based on agent carts.
    Works with a list of product objects.
    """
    # Initialize demand for each product using product_id from the list of products
    demand = {product.product_id: 0 for product in products}
    
    # Calculate demand based on what agents have in their carts
    for agent in agents:
        for product_id in agent.cart.keys():
            if product_id in demand:
                demand[product_id] += agent.cart[product_id]
    
    return demand

