"""
Evaluation test prompts – 10 Healthcare & Life Sciences questions
designed to exercise all agent capabilities.
"""

TEST_PROMPTS: list[dict[str, str]] = [
    {
        "task": "Analyze the impact of recent FDA adaptive trial design guidelines on Phase III oncology trial timelines and recommend process improvements.",
        "goal": "Produce actionable recommendations for the clinical operations leadership team.",
        "expected_topics": ["adaptive trial designs", "FDA guidelines", "oncology", "timelines"],
    },
    {
        "task": "What are the best practices for implementing decentralized clinical trials in oncology, and what cost savings can be expected?",
        "goal": "Build a business case for DCT adoption in our oncology portfolio.",
        "expected_topics": ["decentralized trials", "cost-benefit", "oncology", "patient engagement"],
    },
    {
        "task": "How can real-world evidence be used to support regulatory submissions for label expansion of approved oncology drugs?",
        "goal": "Advise the regulatory affairs team on RWE strategy for a post-approval submission.",
        "expected_topics": ["real-world evidence", "regulatory submissions", "label expansion"],
    },
    {
        "task": "Evaluate the current state of AI and machine learning applications in drug discovery and recommend investment priorities.",
        "goal": "Inform R&D leadership on where to allocate AI/ML investment for maximum impact.",
        "expected_topics": ["AI drug discovery", "target identification", "molecular design", "ADMET"],
    },
    {
        "task": "What patient data privacy safeguards are required for a multi-site international clinical trial using decentralized elements?",
        "goal": "Ensure compliance for a planned 15-country Phase III trial with telemedicine components.",
        "expected_topics": ["HIPAA", "data privacy", "cross-border", "decentralized trials"],
    },
    {
        "task": "Assess pharmacovigilance requirements for a newly approved oncology biologic and recommend a post-market safety monitoring plan.",
        "goal": "Prepare the safety team for launch readiness.",
        "expected_topics": ["pharmacovigilance", "adverse events", "signal detection", "REMS"],
    },
    {
        "task": "How can healthcare organizations build supply chain resilience for temperature-sensitive biologic drugs?",
        "goal": "Develop a cold chain risk mitigation strategy for our biologics portfolio.",
        "expected_topics": ["cold chain", "supply chain resilience", "biologics", "IoT monitoring"],
    },
    {
        "task": "What strategies should a mid-size health system use to transition from fee-for-service to value-based care models?",
        "goal": "Develop a 3-year transition roadmap for the executive team.",
        "expected_topics": ["value-based care", "fee-for-service", "financial modeling", "care coordination"],
    },
    {
        "task": "Compare the clinical operations efficiency of top-quartile pharmaceutical sponsors versus the industry average and identify key differentiators.",
        "goal": "Benchmark our clinical ops against best-in-class and identify improvement opportunities.",
        "expected_topics": ["site activation", "enrollment rates", "data management", "operational efficiency"],
    },
    {
        "task": "What are the regulatory and operational considerations for using AI-assisted signal detection in pharmacovigilance?",
        "goal": "Evaluate whether to implement AI-based PV automation for our safety database.",
        "expected_topics": ["AI pharmacovigilance", "signal detection", "regulatory", "case processing"],
    },
]
