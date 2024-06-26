At any given step:

1. Agents receive test result
    - Input: agents_awaiting_test_result, agents_test_result_date, current stages
    - Argument: false_positive_prob; true_positive_prob
    - Process:
        - tested_positive_agents = agents_awaiting_test_result + false_positive_rate and true_positive_rate
        - reset agents_awaiting_test_result
    - Output: tested_positive_agents; agents_awaiting_test_result;

2. Agents take test:
    - Input: current_stages, is_quarantined, agents_awaiting_test_result, agents_test_result_date
    - Argument: eligibility_compliance_prob, result_delay
    - Process:
        - check eligibility: exposed_infected + not is_quarantined + not awaiting_test_result
        - enrolled = eligibility*compliance_mask
        - agent_awaiting_test_result is true for enrolled agents
        - agents_test_result_date = t + result_delay for enrolled agents
    - Output: agents_awaiting_test_result; agents_test_result_date
