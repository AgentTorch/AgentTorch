The base consumption equation:
- A_{t+1} = (1+r)*A_{t} - C_{t} + Emp_Status*E_{t} + I_{t}

1. We consider a single asset expense model. (liquid asset only)
2. Time step is one week -> because of recurring weekly earnings + UA and stimulus are paid weekly.
3. A_{t}: Cash assets
4. E_{t}: Income / 52-weeks
5. C_{t}: Consumption split into durable and non-durable expenses.
    - Parameters: r -> interest rate of assets; \delta -> deprecation rate on durable expenses.
6. I_{t}: Positive income shocks

What parameters do we calibrate?
What is the ground truth?

https://www.mdrc.org/podcast/how-can-behavioral-science-help-programs-better-serve-clients-during-pandemic 

# regular execution
Assets = (1+r)Assets + Earnings - Expenses

1. Earnings:
    - Recurring income of the household

2. Expenses:
    - Discretionary and Non-discretionary


A. Outflows: (decrease assets or increase liabilities)
1. Non-discretionary - Regular: Rent, Utilities, Mortgage
2. Non-discretionary - Irregular: Medical Expenses
3. Discretionary - Regular: Subscriptions
4. Discretionary - Irregular: Entertainment

B. Inflows: (increase assets)
1. Regular income - if agent is currently employed 
    - (assets += employed*recurring_income)

Constraints:
1. Outflows prioritized in a specific order.