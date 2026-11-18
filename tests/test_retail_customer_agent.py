import unittest
from agent_torch.agents.retail_customer_agent import RetailCustomerAgent
from agent_torch.promotions.promotion import Promotion

class Product:
    def __init__(self, id, price):
        self.id = id
        self.price = price

class TestRetailCustomerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RetailCustomerAgent(
            agent_id=1,
            preferences={1: 0.8, 2: 0.3},
            budget=100
        )
        self.products = {
            1: Product(id=1, price=50),
            2: Product(id=2, price=30)
        }
        self.promotions = {
            1: Promotion(product_id=1, discount_rate=0.2, start_step=0, end_step=5)
        }

    def test_decide_purchases(self):
        current_step = 1  # Example current step
        self.agent.decide_purchases(self.products, self.promotions, current_step)
        self.assertIn(1, self.agent.cart)
        self.assertNotIn(2, self.agent.cart)
        self.assertEqual(self.agent.budget, 60)  # 50 * 0.8 = 40 spent

if __name__ == '__main__':
    # Create a test suite and run it manually
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRetailCustomerAgent)
    result = unittest.TextTestRunner().run(suite)
    
    if result.wasSuccessful():
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. ❌")


