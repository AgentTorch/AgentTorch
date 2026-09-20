# Import necessary libraries
from agent_torch.simulations.retail_simulation import run_retail_simulation
from agent_torch.agents.retail_customer_agent import RetailCustomerAgent
import matplotlib.pyplot as plt
import random

# Define a simple Product class
class Product:
    def __init__(self, product_id, price, category):
        self.product_id = product_id
        self.price = price
        self.category = category

# Step 1: Initialize Products and Promotions
products = [
    Product(product_id=1, price=10.0, category='frozen'),
    Product(product_id=2, price=15.0, category='produce'),
    Product(product_id=3, price=5.0, category='bakery')
]

class Promotion:
    def __init__(self, product_id, discount_rate):
        self.product_id = product_id
        self.discount_rate = discount_rate
    
    def is_active(self, step):
        return True  # Promotions active for all steps

promotions = [
    Promotion(product_id=1, discount_rate=0.20)  # 20% off frozen products
]

# Step 2: Initialize RetailCustomerAgents
agents = [
    RetailCustomerAgent(
        agent_id=i, 
        preferences=RetailCustomerAgent.generate_preferences(products),  # Call the static method
        budget=RetailCustomerAgent.generate_budget()  # Call the static method
    )
    for i in range(100)
]

# Randomize reorders for demonstration
for agent in agents:
    agent.reorders = random.randint(0, 5)

# Step 3: Run Simulation Without Promotion
sales_without_promo = run_retail_simulation(num_agents=100, num_steps=10, products=products, promotions=[])

# Step 4: Run Simulation With Promotion
sales_with_promo = run_retail_simulation(num_agents=100, num_steps=10, products=products, promotions=promotions)

# Step 5: Visualize Sales Before and After Promotion
categories = ['frozen', 'produce', 'bakery']
sales_before = {cat: 0 for cat in categories}
sales_after = {cat: 0 for cat in categories}

# Calculate sales before promotion
for product_id in sales_without_promo.keys():
    for product in products:
        if product.product_id == product_id:
            sales_before[product.category] += sales_without_promo[product_id]

# Calculate sales after promotion
for product_id in sales_with_promo.keys():
    for product in products:
        if product.product_id == product_id:
            sales_after[product.category] += sales_with_promo[product_id]

# Check for empty sales data before plotting
if all(v == 0 for v in sales_before.values()) and all(v == 0 for v in sales_after.values()):
    print("No sales data to plot. Please check agent behavior or product promotions.")
else:
# Plot sales comparison
    plt.figure(figsize=(10, 6))
    labels = ['Frozen', 'Produce', 'Bakery']
    x = range(len(categories))  # positions for the bars
    sales_before_vals = [sales_before[cat] for cat in categories]
    sales_after_vals = [sales_after[cat] for cat in categories]
    bar_width = 0.35  # width of the bars
    # Adjust positions for "Before Promotion" bars
    plt.bar([p - bar_width/2 for p in x], sales_before_vals, width=bar_width, label='Before Promotion')
    # Adjust positions for "After Promotion" bars
    plt.bar([p + bar_width/2 for p in x], sales_after_vals, width=bar_width, label='After Promotion')
    # Setting up labels and ticks
    plt.xticks(x, labels)
    plt.title('Sales Volume by Category Before and After Promotion')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.show()  # Show sales comparison plot


# Step 6: Visualize Reorder Rates Before and After Promotion
reorder_before = [agent.reorders for agent in agents]
reorder_after = [agent.reorders * 1.2 for agent in agents]  # Assuming reorder rate improves post-promotion

# Plot reorder behavior
plt.figure(figsize=(10, 6))
plt.hist([reorder_before, reorder_after], bins=20, label=['Before Promo', 'After Promo'], alpha=0.7)
plt.title('Reorder Behavior Before and After Promotion')
plt.xlabel('Number of Reorders')
plt.ylabel('Frequency')
plt.legend()
plt.show()  # Show reorder behavior plot
