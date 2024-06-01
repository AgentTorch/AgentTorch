# Contributing Guide

Thanks for your interest in contributing to Agent Torch! This guide will show
you how to set up your environment and contribute to this library.

## Prerequisites

You must have the following software installed:

1. [`git`](https://github.com/git-guides/install-git) (latest)
2. [`python`](https://wiki.python.org/moin/BeginnersGuide/Download) (>= 3.10)

Once you have installed the above, follow
[these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to
[`fork`](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)
and [`clone`](https://github.com/git-guides/git-clone) the repository
(`AgentTorch/AgentTorch`).

Once you have forked and cloned the repository, you can
[pick out an issue](https://github.com/AgentTorch/AgentTorch/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)
you want to fix/implement!

## Making Changes

Once you have cloned the repository to your computer (say, in
`~/Code/AgentTorch`) and picked the issue you want to tackle, create a virtual
environment, install all dependencies, and create a new branch to hold all your
work.

```sh
# create a virtual environment
> python -m venv .venv/

# set it up
> . .venv/bin/activate
> pip install -r requirements.txt
> pip install -e .

# create a new branch
> git switch master
> git switch --create branch-name
```

While naming your branch, make sure the name is short and self explanatory.

Once you have created a branch, you can start coding!

## Project Structure

```sh
.
├── agent_torch/
│  ├── helpers/ # defines helper functions used to initialize or work with the state of the simulation.
│  ├── llm/ # contains all the code related to using llms as agents in the simulation
│  ├── __init__.py # exports everything to the world
│  ├── config.py # handles reading and processing the simulation's configuration
│  ├── controller.py # executes the substeps for each episode
│  ├── initializer.py # creates a simulation from a configuration and registry
│  ├── registry.py # registry that stores references to the implementations of the substeps and helper functions
│  ├── runner.py # executes the episodes of the simulation, and handles its state
│  ├── substep.py # contains base classes for the substep observations, actions and transitions
│  └── utils.py # utility functions used throughout the project
├── docs/
│  ├── media/ # assets like screenshots or diagrams inserted in .md files
│  ├── tutorials/ # jupyter notebooks with tutorials and their explanations
│  ├── architecture.md # the framework's architecture
│  └── install.md # instructions on installing the framework
├── models/
│  ├── covid/ # a model simulating disease spread, using the example of covid 19
│  └── predator_prey/ # a simple model used to showcase the features of the framework
├── citation.bib # contains the latex code to use to cite this project
├── contributing.md # this file, helps onboard contributors
├── license.md # contains the license for this project (MIT)
├── readme.md # contains details on the what, why, and how
├── requirements.txt # lists the dependencies of the framework
└── setup.py # defines metadata for the project
```

## Saving Changes

After you have made changes to the code, you will want to
[`commit`](https://github.com/git-guides/git-commit) (basically, Git's version
of save) the changes. To commit the changes you have made locally:

```sh
> git add this/folder that-file.js
> git commit --message 'commit-message'
```

While writing the `commit-message`, try to follow the below guidelines:

1. Prefix the message with `type:`, where `type` is one of the following
   dependending on what the commit does:
   - `fix`: Introduces a bug fix.
   - `feat`: Adds a new feature.
   - `test`: Any change related to tests.
   - `perf`: Any performance related change.
   - `meta`: Any change related to the build process, workflows, issue
     templates, etc.
   - `refc`: Any refactoring work.
   - `docs`: Any documentation related changes.
2. Keep the first line brief, and less than 60 characters.
3. Try describing the change in detail in a new paragraph (double newline after
   the first line).

## Contributing Changes

Once you have committed your changes, you will want to
[`push`](https://github.com/git-guides/git-push) (basically, publish your
changes to GitHub) your commits. To push your changes to your fork:

```sh
> git push origin branch-name
```

If there are changes made to the `master` branch of the `AgentTorch/AgentTorch`
repository, you may wish to merge those changes into your branch. To do so, you
can run the following commands:

```
> git fetch upstream master
> git merge upstream/master
```

This will automatically add the changes from `master` branch of the
`AgentTorch/AgentTorch` repository to the current branch. If you encounter any
merge conflicts, follow
[this guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line)
to resolve them.

Once you have pushed your changes to your fork, follow
[these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a
[`pull request`](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests):

Once you have submitted a pull request, the maintainers of the repository will
review your pull requests and provide feedback. If they find the work to be
satisfactory, they will merge the pull request.

#### Thanks for contributing!

<!-- This contributing guide was inspired by the Electron project's contributing guide. -->
