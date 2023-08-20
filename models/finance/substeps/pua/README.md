At each step of the simulation, the following are happening:

PUA Object has two components:
- pua_enrollment (num_agents,)
- pua_payment (num_agents, num_steps)

- administrative_delay (~ discrete_distribution)
- start_date
- end_date

1. Agent's receive PUA
    - transiton:
        - update: current_assets
        - current_assets += enrolled_agents*pua_payments[t].to_dense()

2. Agents request enrollment
    - observation: null
    - policy:
        - request_enrollment = give_me_pua(occupation_status)
        - action: requesting_enrollment
    - transition:
        - update_enrollment_status: 
            - approve_prob = 0.7
            - update: pua_enrolled_agents
        - update_pua_payment
            - compute_pua_amount, start_date, end_date, delay
            - update: pua_payment

3. Agents end PUA enrollment
    - transition:
        - update_enrollment_status
            - if t >= end_date
            - update: pua_enrolled_agents