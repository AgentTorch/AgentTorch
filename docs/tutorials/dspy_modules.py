import dspy


class BasicQAWillToWork(dspy.Signature):
    """
    You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC.
    """

    history = dspy.InputField(
        desc="may contain your decision in the previous months", format=list
    )
    question = dspy.InputField(
        desc="will contain the number of COVID cases in NYC, your age and other information about the economy and your identity, to help you decide your willingness to work and consumption demands"
    )
    answer = dspy.OutputField(
        desc="will contain single float value, between 0 and 1, representing realistic probability of your willingness to work. No other information should be there."
    )


class BasicQACovid(dspy.Signature):
    """Consider a random person with the following attributes:
    * age: {age}
    * location: {location}

    There is a novel disease. It spreads through contact. It is more dangerous to older people.
    People have the option to isolate at home or continue their usual recreational activities outside.
    Given this scenario, you must estimate the person's actions based on
        1) the information you are given,
        2) what you know about the general population with these attributes.

    "There isn't enough information" and "It is unclear" are not acceptable answers.
    Give a "Yes" or "No" answer, followed by a period. Give one sentence explaining your choice.
    """

    history = dspy.InputField(
        desc="may contain your decision in the previous months", format=list
    )
    question = dspy.InputField(
        desc="will contain the number of weeks since a disease started (if specified), the number of new cases this week, the percentage change from the past month's average, and asks if the person chooses to isolate at home. It may have other information also."
    )
    answer = dspy.OutputField(
        desc="Give a 'Yes' or 'No' answer, followed by a period. No other information should be there in the answer"
    )


class Reflect(dspy.Signature):
    """
    You are an Economic Analyst.
    """

    history = dspy.InputField(desc="may contain data on previous months", format=list)
    question = dspy.InputField(desc="may contain the question you are being asked")
    answer = dspy.OutputField(
        desc="may contain your analysis of the question asked based on the data in the history"
    )


class COT(dspy.Module):
    def __init__(self, qa):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(qa)

    def forward(self, question, history):
        prediction = self.generate_answer(question=question, history=history)
        return dspy.Prediction(answer=prediction)
