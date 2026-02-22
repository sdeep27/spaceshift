# tests/test_evaluate.py

import pytest
from cruise_llm import LLM, pairwise_evaluate


class TestEvaluateWithRealLLMOutputs:
    """Tests using actual LLM-generated results - the real use case."""

    def test_compare_different_system_prompts(self):
        """Compare outputs from same model with different system prompts."""
        user_question = "Explain what a neural network is"

        # Generate responses with different personas
        technical = LLM(model=1, v=False).sys(
            "You are a machine learning researcher. Be technical and precise."
        ).user(user_question).res()

        simple = LLM(model=1, v=False).sys(
            "You are a patient teacher for beginners. Use simple analogies."
        ).user(user_question).res()

        brief = LLM(model=1, v=False).sys(
            "Be extremely concise. One sentence max."
        ).user(user_question).res()

        eval_result = pairwise_evaluate(
            results=[technical, simple, brief],
            additional_information="Evaluating explanations for a general audience",
            metrics=["How accessible is this for someone with no ML background?"],
            v=False
        )

        assert len(eval_result["rankings"]) == 3
        assert all(0 <= s <= 1 for s in eval_result["scores"].values())

    def test_compare_different_models(self):
        """Compare outputs from different models on same prompt."""
        prompt = "Write a haiku about programming"

        results = []
        for model in [1, 2, 3]:  # top 3 models
            response = LLM(model=model, v=False).user(prompt).res()
            results.append(response)

        eval_result = pairwise_evaluate(
            results=results,
            metrics=[
                "Does this follow haiku structure (5-7-5 syllables)?",
                "How creative and evocative is this haiku?"
            ],
            v=False
        )

        assert len(eval_result["rankings"]) == 3
        assert "comparison_matrix" in eval_result["raw"]

    def test_evaluate_generated_json_outputs(self):
        """Evaluate structured JSON outputs from LLM."""
        prompt_template = LLM(model=1, v=False).sys(
            "Extract entities from text. Return JSON: {entities: [{name, type}]}"
        ).user("{text}")

        text_samples = [
            "Apple CEO Tim Cook announced the new iPhone in Cupertino.",
            "Microsoft and Google are competing in the AI space.",
        ]

        results = []
        for text in text_samples:
            output = prompt_template.run_json(text=text)
            results.append(str(output))  # Convert to string for evaluation

        eval_result = pairwise_evaluate(
            results=results,
            metrics=["How complete is the entity extraction?"],
            v=False
        )

        assert len(eval_result["rankings"]) == 2

    def test_prompt_variation_evaluation(self):
        """Generate prompt variations and evaluate which works best."""
        base_task = "Summarize a news article"

        # Use LLM to generate prompt variations
        variations = LLM(model=1, v=False).sys(
            "Generate 3 different system prompts for the given task. "
            "Return JSON: {prompts: [str, str, str]}"
        ).user(f"Task: {base_task}").res_json()

        # Test each prompt variation on same input
        test_article = "The stock market rose 2% today amid positive jobs data..."

        results = []
        for sys_prompt in variations.get("prompts", [])[:3]:
            response = LLM(model=1, v=False).sys(sys_prompt).user(test_article).res()
            results.append(response)

        if len(results) >= 2:
            eval_result = pairwise_evaluate(
                prompts=variations.get("prompts", [])[:len(results)],
                results=results,
                additional_information="Evaluating prompt effectiveness for summarization",
                v=False
            )
            assert "rankings" in eval_result

    def test_iterative_refinement_evaluation(self):
        """Evaluate responses across refinement iterations."""
        topic = "Explain quantum computing"

        # First attempt
        v1 = LLM(model=1, v=False).user(topic).res()

        # Refined with feedback
        v2 = LLM(model=1, v=False).sys(
            "Improve upon previous explanation. Make it clearer."
        ).user(f"Original: {v1}\n\nPlease improve this explanation.").res()

        # Further refined
        v3 = LLM(model=1, v=False).sys(
            "Add a concrete real-world example to this explanation."
        ).user(f"Explanation: {v2}\n\nAdd a practical example.").res()

        eval_result = pairwise_evaluate(
            results=[v1, v2, v3],
            metrics=["How clear and complete is this explanation?"],
            v=False
        )

        # Later iterations should generally rank higher
        assert len(eval_result["rankings"]) == 3


class TestEvaluateLastRealWorkflows:
    """Tests for evaluate_last with real LLM workflows."""

    def test_score_single_response(self):
        """Basic scoring of a single LLM response."""
        llm = LLM(model=1, v=False).sys(
            "You are a helpful coding assistant."
        ).user(
            "Write a Python function to check if a number is prime"
        ).chat()

        score = llm.evaluate_last(
            metrics={
                "Is the code correct and handles edge cases?": "1-10",
                "Is the code clean and well-structured?": "1-10"
            },
            v=False
        )

        assert 0 <= score["score"] <= 1
        assert len(score["metric_scores"]) == 2

    def test_compare_models_same_prompt(self):
        """Primary use case: compare different models on same task."""
        system = "You are a creative writing assistant."
        prompt = "Write a compelling opening line for a mystery novel."

        metrics = {
            "How intriguing and hook-worthy is this opening?": "1-10",
            "How well does it set a mysterious tone?": "1-10"
        }

        llm_fast = LLM(model="fast", v=False).sys(system).user(prompt).chat()
        llm_best = LLM(model="best", v=False).sys(system).user(prompt).chat()
        llm_optimal = LLM(model="optimal", v=False).sys(system).user(prompt).chat()

        scores = {
            "fast": llm_fast.evaluate_last(metrics=metrics, v=False),
            "best": llm_best.evaluate_last(metrics=metrics, v=False),
            "optimal": llm_optimal.evaluate_last(metrics=metrics, v=False)
        }

        # All should return valid scores
        for name, score_data in scores.items():
            assert 0 <= score_data["score"] <= 1
            assert "metric_scores" in score_data

    def test_evaluate_with_prompt_context(self):
        """Include the prompt in evaluation for instruction-following check."""
        llm = LLM(model=1, v=False).sys(
            "Always respond in exactly 3 bullet points."
        ).user(
            "What are the benefits of exercise?"
        ).chat()

        score = llm.evaluate_last(
            include_prompt=True,
            metrics={"Did the response follow the instruction format?": "0-1"},
            v=False
        )

        assert "score" in score

    def test_evaluate_json_response(self):
        """Evaluate a JSON-mode response."""
        llm = LLM(model=1, v=False).sys(
            "Extract sentiment. Return: {sentiment: positive|negative|neutral, confidence: 0-1}"
        ).user("I love this product! Best purchase ever!").chat_json()

        score = llm.evaluate_last(
            metrics={
                "Is the sentiment classification correct?": "0-1",
                "Is the confidence level appropriate?": "1-10"
            },
            v=False
        )

        assert 0 <= score["score"] <= 1

    def test_multi_turn_conversation_evaluation(self):
        """Evaluate response quality in a multi-turn conversation."""
        llm = LLM(model=1, v=False).sys("You are a math tutor.")

        # Multi-turn conversation
        llm.user("What is calculus?").chat()
        llm.user("Can you give me a simple example?").chat()
        llm.user("How is this used in real life?").chat()

        # Evaluate the final response in context
        score = llm.evaluate_last(
            include_prompt=True,
            additional_information="This is the 3rd message in a tutoring conversation",
            metrics={"How well does this build on the conversation context?": "1-10"},
            v=False
        )

        assert "score" in score


class TestAutoGeneratedMetrics:
    """Tests for automatic metric generation."""

    def test_auto_metrics_for_code(self):
        """Auto-generate appropriate metrics for code evaluation."""
        code_responses = [
            LLM(model=1, v=False).user("Write fizzbuzz in Python").res(),
            LLM(model=2, v=False).user("Write fizzbuzz in Python").res()
        ]

        eval_result = pairwise_evaluate(
            results=code_responses,
            metrics=[],  # Auto-generate
            additional_information="These are code solutions",
            v=False
        )

        assert eval_result["raw"]["auto_generated_metrics"] == True
        assert len(eval_result["raw"]["metrics_used"]) == 3

    def test_auto_metrics_for_creative_writing(self):
        """Auto-generate appropriate metrics for creative content."""
        stories = [
            LLM(model=1, v=False).user("Write a 2-sentence horror story").res(),
            LLM(model=1, v=False).user("Write a 2-sentence horror story").res()
        ]

        eval_result = pairwise_evaluate(
            results=stories,
            metrics=[],
            additional_information="Flash fiction horror stories",
            v=False
        )

        # Metrics should be generated
        assert len(eval_result["raw"]["metrics_used"]) == 3


class TestValidationAndErrors:
    """Test error handling and validation."""

    def test_mismatched_prompts_results_length(self):
        """Error when prompts and results arrays have different lengths."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = [
            LLM(model=1, v=False).user("test").res()
        ]  # Only 1 result

        with pytest.raises(ValueError, match="(?i)length"):
            pairwise_evaluate(prompts=prompts, results=results)

    def test_empty_inputs_error(self):
        """Error when nothing to evaluate."""
        with pytest.raises(ValueError):
            pairwise_evaluate(prompts=[], results=[])

    def test_evaluate_last_no_response(self):
        """Error when evaluate_last called before any response."""
        llm = LLM(model=1, v=False).user("Hello")  # No chat() call

        with pytest.raises(ValueError, match="(?i)no.*response"):
            llm.evaluate_last()

    def test_invalid_weights_keys(self):
        """Error when weight keys don't match metric keys."""
        results = [
            LLM(model=1, v=False).user("test").res(),
            LLM(model=1, v=False).user("test").res()
        ]

        with pytest.raises(ValueError, match="(?i)weight"):
            pairwise_evaluate(
                results=results,
                metrics=["How good?"],
                weights={"Wrong key": 1.0}  # Doesn't match
            )


class TestBradleyTerryForManyItems:
    """Test Bradley-Terry sampling for >5 items."""

    def test_many_results_uses_sampling(self):
        """More than 5 items should use Bradley-Terry sampling."""
        # Generate 8 different responses
        results = []
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]

        for temp in temperatures:
            response = LLM(model=1, temperature=temp, v=False).user(
                "Write a creative tagline for a coffee shop"
            ).res()
            results.append(response)

        eval_result = pairwise_evaluate(
            results=results,
            metrics=["How creative and memorable?"],
            v=False
        )

        assert len(eval_result["rankings"]) == 8
        # Verify rankings are valid permutation
        assert sorted(eval_result["rankings"]) == list(range(8))


class TestPositionBiasMitigation:
    """Test position swap functionality."""

    def test_position_swap_on(self):
        """Position swap doubles comparisons for bias mitigation."""
        results = [
            LLM(model=1, v=False).user("Explain REST APIs").res(),
            LLM(model=1, v=False).user("Explain REST APIs").res()
        ]

        eval_result = pairwise_evaluate(
            results=results,
            position_swap=True,
            v=False
        )

        assert "comparison_matrix" in eval_result["raw"]

    def test_position_swap_off(self):
        """Can disable position swap for faster evaluation."""
        results = [
            LLM(model=1, v=False).user("Hello").res(),
            LLM(model=1, v=False).user("Hello").res()
        ]

        eval_result = pairwise_evaluate(
            results=results,
            position_swap=False,
            v=False
        )

        assert "rankings" in eval_result


class TestRealWorldScenarios:
    """End-to-end tests for realistic use cases."""

    def test_ab_test_prompt_variants(self):
        """A/B test two prompt variants for a chatbot."""
        user_query = "How do I reset my password?"

        # Variant A: Formal
        response_a = LLM(model=1, v=False).sys(
            "You are a professional customer service agent. Be formal and thorough."
        ).user(user_query).res()

        # Variant B: Friendly
        response_b = LLM(model=1, v=False).sys(
            "You are a friendly helper. Be warm and conversational."
        ).user(user_query).res()

        eval_result = pairwise_evaluate(
            results=[response_a, response_b],
            additional_information="Customer support chatbot for a tech company",
            metrics=[
                "How helpful is this response?",
                "How appropriate is the tone for customer service?"
            ],
            v=False
        )

        # Get winner
        winner_idx = eval_result["rankings"][0]
        assert winner_idx in [0, 1]

    def test_evaluate_rag_responses(self):
        """Evaluate RAG-style responses with context."""
        context = "The Eiffel Tower was built in 1889 and is 330 meters tall."
        question = "When was the Eiffel Tower built and how tall is it?"

        # Response that uses context well
        good_response = LLM(model=1, v=False).sys(
            f"Answer based on this context: {context}"
        ).user(question).res()

        # Response without context (may hallucinate)
        risky_response = LLM(model=1, v=False).user(question).res()

        eval_result = pairwise_evaluate(
            results=[good_response, risky_response],
            additional_information=f"Ground truth context: {context}",
            metrics=[
                "How factually accurate based on the provided context?",
                "Does the response stick to the given information?"
            ],
            v=False
        )

        # Context-aware response should generally win
        assert eval_result["rankings"][0] in [0, 1]

    def test_evaluate_translation_quality(self):
        """Evaluate translation outputs."""
        english = "The quick brown fox jumps over the lazy dog."

        translations = []
        for model in [1, 2]:
            translation = LLM(model=model, v=False).sys(
                "You are an expert translator."
            ).user(f"Translate to French: {english}").res()
            translations.append(translation)

        eval_result = pairwise_evaluate(
            results=translations,
            additional_information=f"Original English: {english}",
            metrics=[
                "How accurate is this translation?",
                "How natural does the French sound?"
            ],
            v=False
        )

        assert len(eval_result["rankings"]) == 2
