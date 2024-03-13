## Sample simulation experience - target Sept 1 2024

sim = Simulator()

agent.transmit()
agent.progress()
agent.quarantine()
agent.test()
agent.vaccinate()

virus.evolve()

agent.work_and_earn() # variable to update: assets
agent.spend() # variable to update: assets and debt
agent.pay_tax() # action - will you skim on tax?; transition - how much tax am i paying -> variable: assets
agent.save() # variable: assets and debt

market.evolve() # change interest rate, compute unemployment rate

sim.dispatch(agent, virus, market, population='') # vmap internally
