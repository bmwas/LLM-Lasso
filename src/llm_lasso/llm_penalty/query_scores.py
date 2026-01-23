from pydantic import BaseModel
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.utils.score_collection import extract_scores_from_responses
import logging

class Score(BaseModel):
    """Score for a single feature/factor."""
    gene: str  # Keep as 'gene' for backward compatibility with structured output schema
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    """Collection of feature scores (named GeneScores for backward compatibility)."""
    scores: list[Score]


def query_scores_with_retries(
    model: LLMQueryWrapperWithMemory,
    system_message: str,
    full_prompt: str,
    batch_features: list[str],
    retry_limit=50
) -> tuple[list[int], str]:
    """
    Query an LLM for feature/factor scores, with automatic retries.
    
    Args:
        model: LLM wrapper with query capabilities
        system_message: System prompt for the LLM
        full_prompt: User prompt containing the features to score
        batch_features: List of feature names to score
        retry_limit: Maximum number of retries
        
    Returns:
        Tuple of (list of scores, raw LLM output)
    """
    upper_batch_names = [n.upper() for n in batch_features]

    if model.has_structured_output():
        feature_scores: GeneScores = model.structured_query(
            system_message=system_message,
            full_prompt=full_prompt,
            response_format_class=GeneScores,
            sleep_time=1,
        )
        scores_list = [score for score in feature_scores.scores if score.gene.upper() in upper_batch_names]
        features_retrieved = set([score.gene.upper() for score in scores_list])
        missing = set(upper_batch_names).difference(features_retrieved)

        # Retry logic for score validation
        n_retries = 0
        while len(missing) > 0:
            logging.warning(f"Missing features: {missing}. Retrying...")
            assert n_retries < retry_limit, f"Exceeded retry limit ({retry_limit}) for missing features: {missing}"
            n_retries += 1

            feature_scores: GeneScores = model.retry_last(sleep_time=1)
            scores_list = [score for score in feature_scores.scores if score.gene.upper() in upper_batch_names]
            features_retrieved = set([score.gene.upper() for score in scores_list])
            missing = set(upper_batch_names).difference(features_retrieved)
        
        feature_to_scores = {
            score.gene: score.penalty_factor for score in feature_scores.scores
        }
        batch_scores_partial = [feature_to_scores[feature] for feature in batch_features]
        output = feature_scores.model_dump_json()
    else:
        output = model.query(
            system_message=system_message,
            full_prompt=full_prompt,
            sleep_time=1,
        )

        batch_scores_partial = extract_scores_from_responses(
            output if isinstance(output, list) else [output],
            batch_features
        )

        # Retry logic for score validation
        n_retries = 0
        while len([score for score in batch_scores_partial if score is not None]) != len(batch_features):
            logging.info(output)
            assert n_retries < retry_limit, f"Exceeded retry limit ({retry_limit}) for batch {batch_features}"
            n_retries += 1
            try:
                valid_count = len([s for s in batch_scores_partial if s is not None])
                logging.warning(
                    f"Batch scores count mismatch: got {valid_count}/{len(batch_features)} scores "
                    f"for features {batch_features}. Retrying..."
                )
                output = model.retry_last(sleep_time=1)
                batch_scores_partial = extract_scores_from_responses(
                    output if isinstance(output, list) else [output],
                    batch_features
                )
            except Exception as e:
                logging.error(f"Error during retry: {str(e)}. Continuing retry...")
        # end retry while loop
    # end structured output if/else
    return batch_scores_partial, output