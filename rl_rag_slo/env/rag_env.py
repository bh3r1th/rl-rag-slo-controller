from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import BaseLLMClient
from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score
from rl_rag_slo.controller.actions import ACTIONS, RagActionConfig


@dataclass
class RagStepResult:
    """
    Result of a single RAG environment step.
    """
    answer: str
    reward: float
    cost_tokens: int
    meta: Dict[str, Any]


class RagEnvironment:
    """
    Single-step environment for RL control of a RAG pipeline.
    Each call to `step` corresponds to one (question, action) evaluation.

    This implementation assumes:
    - A real LLM backend (e.g., gpt-4.1-nano via OpenAILLMClient).
    - Reward is based on QA quality (EM), cost, hallucinations, and refusal behavior.
    """

    def __init__(self, retriever: BM25Retriever, llm_client: BaseLLMClient, slo_weights: Dict[str, float]):
        """
        Args:
            retriever: Object with retrieve(query, top_k) -> list of docs.
            llm_client: LLM backend used to generate answers.
            slo_weights: Dict with SLO weights and scaling parameters:
                - "w_quality"
                - "w_cost"
                - "w_refusal"
                - "w_halluc"
                - "lambda_cost"
                - "lambda_halluc"
                - "lambda_wrong_ref"
                - "lambda_correct_ref"
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.slo = slo_weights

    def step(
        self,
        question: str,
        ground_truth: Optional[str],
        action_id: int,
        extra_eval: Optional[Dict[str, Any]] = None,
    ) -> RagStepResult:
        """
        Run a single RAG step for a given question and action.

        - Look up RagActionConfig from ACTIONS[action_id].
        - If answer_mode == "refuse":
            - Return a fixed refusal string; no retrieval, no LLM call.
        - Otherwise:
            - Retrieve documents using the configured k.
            - Call llm_client.answer_with_context(...) with guarded flag.
        - Compute scalar reward using _compute_reward.

        Returns:
            RagStepResult with answer, reward, cost_tokens, and a meta dict.
        """
        if action_id not in ACTIONS:
            raise KeyError(f"Unknown action_id: {action_id}")

        cfg: RagActionConfig = ACTIONS[action_id]

        if extra_eval is None:
            extra_eval = {}

        # Explicit refusal: no retrieval, no LLM call
        if cfg.answer_mode == "refuse":
            answer = "I cannot safely answer this question based on the provided constraints."
            cost_tokens = len(answer.split())
            # For refusals, we still want scoring logic to identify correct vs wrong refusals.
            reward, cost_tokens = self._compute_reward(
                answer=answer,
                ground_truth=ground_truth,
                cost_tokens=cost_tokens,
                extra_eval=extra_eval,
            )
            return RagStepResult(
                answer=answer,
                reward=reward,
                cost_tokens=cost_tokens,
                meta={
                    "k": cfg.k,
                    "answer_mode": cfg.answer_mode,
                    "n_tokens_generated": len(answer.split()),
                    "n_tokens_context": 0,
                },
            )

        # Normal RAG action: retrieval + LLM call
        docs = self.retriever.retrieve(question, top_k=cfg.k)

        # LLM call: guarded vs auto controls safety instructions internally.
        answer, n_tokens_generated, n_tokens_context = self.llm_client.answer_with_context(
            question=question,
            docs=docs,
            model_size="base",  # OpenAILLMClient ignores this and always uses gpt-4.1-nano in our setup.
            guarded=(cfg.answer_mode == "guarded"),
        )
        cost_tokens = n_tokens_generated + n_tokens_context

        reward, cost_tokens = self._compute_reward(
            answer=answer,
            ground_truth=ground_truth,
            cost_tokens=cost_tokens,
            extra_eval=extra_eval,
        )

        meta: Dict[str, Any] = {
            "k": cfg.k,
            "answer_mode": cfg.answer_mode,
            "n_tokens_generated": n_tokens_generated,
            "n_tokens_context": n_tokens_context,
        }

        return RagStepResult(
            answer=answer,
            reward=reward,
            cost_tokens=cost_tokens,
            meta=meta,
        )

    def _compute_reward(
        self,
        answer: str,
        ground_truth: Optional[str],
        cost_tokens: int,
        extra_eval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int]:
        """
        Compute scalar reward using compute_qa_score and SLO weights.

        Reward (per query) is:

            R = w_quality * accuracy
                - w_cost * lambda_cost * cost_tokens
                - w_halluc * lambda_halluc * hallucination
                + w_refusal * (lambda_correct_ref * correct_refusal
                               - lambda_wrong_ref * wrong_refusal)

        Where:
            - accuracy ∈ [0, 1] (EM or F1 from compute_qa_score)
            - hallucination ∈ {0, 1}
            - correct_refusal ∈ {0, 1}
            - wrong_refusal ∈ {0, 1}
        """
        metrics = compute_qa_score(answer, ground_truth, extra_eval)

        accuracy = float(metrics.get("accuracy", 0.0))
        hallucination = int(metrics.get("hallucination", 0))
        correct_refusal = int(metrics.get("correct_refusal", 0))
        wrong_refusal = int(metrics.get("wrong_refusal", 0))

        wq = float(self.slo.get("w_quality", 0.7))
        wc = float(self.slo.get("w_cost", 0.2))
        wr = float(self.slo.get("w_refusal", 0.1))
        wh = float(self.slo.get("w_halluc", 0.3))

        lambda_cost = float(self.slo.get("lambda_cost", 1e-3))
        lambda_h = float(self.slo.get("lambda_halluc", 1.0))
        lambda_wr = float(self.slo.get("lambda_wrong_ref", 1.0))
        lambda_cr = float(self.slo.get("lambda_correct_ref", 1.0))

        reward = (
            wq * accuracy
            - wc * lambda_cost * float(cost_tokens)
            - wh * lambda_h * float(hallucination)
            + wr * (lambda_cr * float(correct_refusal) - lambda_wr * float(wrong_refusal))
        )

        return float(reward), cost_tokens
