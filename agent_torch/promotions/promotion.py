# agent_torch/promotions/promotion.py

class Promotion:
    def __init__(self, product_id, discount_rate, start_step, end_step):
        self.product_id = product_id
        self.discount_rate = discount_rate  # e.g., 0.2 for 20% off
        self.start_step = start_step
        self.end_step = end_step

    def is_active(self, current_step):
        """
        Check if the promotion is active based on the current step in the simulation.
        """
        return self.start_step <= current_step <= self.end_step

