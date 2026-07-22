# Day 4 – AI Agent Security Lab

This lab demonstrates how a prompt injection attack can change the intended behavior of a LangGraph research agent.

The project includes:

- `multiagent_lab_vulnerable.py` – the agent before adding security protection.
- `multiagent_lab_hardened.py` – the protected agent with input guardrails and secure routing.
- `attack_output_before_guardrail.md` – the attack result before protection.
- `attack_output_after_guardrail.txt` – the blocked attack result after protection.
- `Dockerfile` – used to run the project inside a Docker container.
- `requirements.txt` – required Python libraries.

The security guard detects task-hijacking instructions and stops the workflow before the malicious prompt reaches the research agents.

**Author:** Muneera Alsaeed
