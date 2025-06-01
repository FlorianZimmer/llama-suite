import logging
from typing import List, Dict, Tuple, Union, Optional
from operator import itemgetter
import dataclasses # For Qwen class

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI # Base for our LlamaCpp compatible one
from lm_eval.models.utils import handle_stop_sequences # For generate=True logic
from lm_eval.api.instance import Instance # For type hinting in Qwen class

# Logger for this custom module
custom_api_logger = logging.getLogger("lm_eval.custom_local_api") # Used by both classes


@register_model("llama_cpp_compatible_api") # For general llama.cpp server use
class LlamaCppCompatibleCompletionsAPI(LocalCompletionsAPI):
    """
    Overrides LocalCompletionsAPI for better compatibility with llama.cpp server:
    - Removes "echo:true" from loglikelihood payloads.
    - Provides a custom logprob parser tailored for llama.cpp server output.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.AUTO_TOKENIZE = False # Explicitly tell lm-eval not to auto-tokenize for this API
        custom_api_logger.info(
            f"Initialized LlamaCppCompatibleCompletionsAPI for model/engine: {self.model}"
        )

    def _create_payload(
            self,
            messages: Union[List[List[int]], List[dict], List[str], str], 
            generate=False,
            gen_kwargs: Optional[dict] = None,
            seed: int = 1234,
            eos=None, 
            **kwargs, 
        ) -> dict:
        
        custom_api_logger.debug( 
            f"LlamaCppCompatibleAPI._create_payload IN. generate={generate}. "
            f"Type of messages: {type(messages)}. "
            f"Value (first 200 chars/items): {str(messages)[:200]}"
        )
        
        prompt_to_send: str

        if isinstance(messages, str):
            prompt_to_send = messages
        elif isinstance(messages, list) and messages:
            if all(isinstance(item, int) for item in messages): 
                # This case handles messages being List[int] (a single tokenized prompt)
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    custom_api_logger.debug("Messages is List[int], attempting to decode.")
                    try:
                        prompt_to_send = self.tokenizer.decode(messages)
                    except Exception as e:
                        custom_api_logger.error(f"Failed to decode List[int] prompt: {e}. Original: {str(messages)[:100]}")
                        raise ValueError(f"Cannot decode token list for prompt: {str(messages)[:100]}") from e
                else:
                    custom_api_logger.error("Received List[int] prompt but no tokenizer available to decode.")
                    raise ValueError("Cannot decode token list for prompt: No tokenizer available on API model object.")
            
            elif isinstance(messages[0], list) and all(isinstance(item, int) for item in messages[0]):
                custom_api_logger.debug( # CHANGED TO DEBUG
                    f"Unexpected payload format: _create_payload received List[List[int]]. "
                    f"This typically means the prompt was tokenized and then wrapped in an outer list. "
                    f"Using the first element of the outer list as the tokenized prompt. "
                    f"Data (first element shown): {str(messages[0])[:100]}."
                )
                first_prompt_tokens = messages[0] # This is List[int]
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    try:
                        prompt_to_send = self.tokenizer.decode(first_prompt_tokens)
                    except Exception as e:
                        custom_api_logger.error(f"Failed to decode tokenized prompt (from List[List[int]] payload): {e}. Original tokens: {str(first_prompt_tokens)[:100]}")
                        # Re-raise as ValueError to halt if decoding fails, as it's crucial.
                        raise ValueError(f"Cannot decode tokenized prompt from List[List[int]] payload: {str(first_prompt_tokens)[:100]}") from e
                else:
                    custom_api_logger.error("Cannot process List[List[int]] payload: No tokenizer available on API model object for decoding.")
                    raise ValueError("Cannot process List[List[int]] payload: No tokenizer available.")
            # END OF MODIFIED BRANCH

            else: 
                custom_api_logger.error(f"Messages is a list with unexpected element types: {[type(m) for m in messages[:2]]}. Cannot form prompt.")
                raise ValueError(f"Unsupported 'messages' list content for prompt: {[type(m) for m in messages[:2]]}")
        else: 
            custom_api_logger.error(f"Messages is an unexpected type or empty, cannot form prompt: {type(messages)}.")
            raise ValueError(f"Invalid 'messages' type for prompt: {type(messages)}")

        custom_api_logger.debug(f"Final prompt_to_send (type: {type(prompt_to_send)}): {str(prompt_to_send)[:200]}")

        gen_kwargs = gen_kwargs or {} 
        if generate:
            gen_kwargs.pop("do_sample", None) 
            max_tokens = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_gen_toks", self._max_gen_toks))
            temperature = gen_kwargs.pop("temperature", 0.0) 
            
            raw_stop_sequences = gen_kwargs.pop("until", None) 
            final_stop_sequences = None # Initialize

            try:
                # Primary attempt: pass both, assuming signature like handle_stop_sequences(stop_list, eos_string)
                # or handle_stop_sequences(stop_list, eos_token=eos_string)
                # The error "missing 1 required positional argument: 'eos'" from before suggests it wants 'eos' positionally.
                final_stop_sequences = handle_stop_sequences(raw_stop_sequences, eos)
            except TypeError as te:
                custom_api_logger.warning(
                    f"TypeError calling handle_stop_sequences(raw_stop_sequences, eos): {te}. "
                    f"raw_stop_sequences: {raw_stop_sequences}, eos: {eos}. "
                    "This might indicate an older lm-eval version or an unexpected signature. "
                    "Attempting fallback to single-argument call if applicable."
                )
                # Check if the error is about too many arguments (meaning it only wants one)
                if "takes 1 positional argument but 2 were given" in str(te) or \
                   "takes at most 1 positional argument" in str(te) or \
                   "handle_stop_sequences() takes 1 positional argument but" in str(te).lower() : # More checks
                    custom_api_logger.info("Falling back to single-argument call for handle_stop_sequences, manually combining stops.")
                    combined_stop_for_fallback = raw_stop_sequences
                    if isinstance(combined_stop_for_fallback, str) and eos and combined_stop_for_fallback != eos:
                        combined_stop_for_fallback = [combined_stop_for_fallback, eos]
                    elif isinstance(combined_stop_for_fallback, list) and eos and eos not in combined_stop_for_fallback:
                        combined_stop_for_fallback.append(eos) # Modify list in place or create new
                    elif not combined_stop_for_fallback and eos: # raw_stop_sequences was None but eos exists
                        combined_stop_for_fallback = [eos]
                    
                    if combined_stop_for_fallback is not None:
                        final_stop_sequences = handle_stop_sequences(combined_stop_for_fallback)
                    else: # Both raw_stop_sequences and eos were None
                        final_stop_sequences = None
                else: # Different TypeError, re-raise it as we don't know how to handle it
                    raise te
            
            payload = {
                "prompt": prompt_to_send, 
                "model": self.model, 
                "max_tokens": max_tokens, 
                "temperature": temperature, 
                "seed": seed,
                **({"stop": final_stop_sequences} if final_stop_sequences else {}),
                **gen_kwargs, 
            }
            # custom_api_logger.debug(f"LlamaCppCompatibleAPI: Generate payload for '{self.model}': {str(payload)[:500]}...")
            return payload
        else: # Loglikelihood
            payload = {
                "model": self.model, 
                "prompt": prompt_to_send, 
                "temperature": 0,    
                "max_tokens": 1,     
                "logprobs": 1,       
                "seed": seed,
            }
            # custom_api_logger.debug(f"LlamaCppCompatibleAPI: Loglikelihood payload for '{self.model}': {payload}")
            return payload
        
    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None, 
        ctxlens: List[int] = None,   
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        for out_idx, out in enumerate(outputs):
            # Assuming n=1, so one choice per 'out' (response object)
            if not out.get("choices"):
                custom_api_logger.warning(f"No 'choices' in output {out_idx}: {str(out)[:200]}")
                res.append((-float('inf'), False)) # Penalize if no choices
                continue

            for choice_idx, choice in enumerate(sorted(out.get("choices", []), key=itemgetter("index"))):
                logprobs_obj = choice.get("logprobs")
                summed_token_logprobs_for_this_choice = 0.0
                is_greedy_for_this_choice = True # Assume greedy by default

                if logprobs_obj and isinstance(logprobs_obj, dict) and \
                   "content" in logprobs_obj and isinstance(logprobs_obj["content"], list):
                    
                    token_logprobs_from_server = []
                    top_logprobs_for_each_token = [] 

                    # logprobs_obj["content"] is a list of dicts, one for each generated token.
                    # For loglikelihood with max_tokens=1, this list should have 0 or 1 item.
                    for content_item_idx, content_item in enumerate(logprobs_obj["content"]):
                        if isinstance(content_item, dict) and "logprob" in content_item:
                            token_logprobs_from_server.append(content_item["logprob"])
                            # Store top_logprobs if available (list of dicts)
                            if "top_logprobs" in content_item and isinstance(content_item["top_logprobs"], list):
                                top_logprobs_for_each_token.append(content_item["top_logprobs"])
                            else:
                                top_logprobs_for_each_token.append(None) # Mark as missing for this token
                        else:
                            custom_api_logger.warning(f"Malformed content_item {content_item_idx} in logprobs for choice {choice_idx}: {content_item}")
                            token_logprobs_from_server.append(-float('inf')) # Penalize
                            top_logprobs_for_each_token.append(None)

                    if not token_logprobs_from_server: # No tokens generated or no logprobs for them
                        # custom_api_logger.debug(f"No token logprobs found in logprobs.content for choice {choice_idx}.")
                        summed_token_logprobs_for_this_choice = -float('inf') 
                        is_greedy_for_this_choice = False
                    else:
                        # For loglikelihood, we are typically interested in the logprob of the *continuation*.
                        # If max_tokens=1, this sums the logprob of the single generated token.
                        summed_token_logprobs_for_this_choice = sum(token_logprobs_from_server)
                        
                        # Greediness check for the *first* generated token
                        if top_logprobs_for_each_token and top_logprobs_for_each_token[0] is not None:
                            first_token_actual_logprob = token_logprobs_from_server[0]
                            # top_logprobs_for_each_token[0] is a list of dicts: [{"token": T, "logprob": L}, ...]
                            all_top_values_for_first_token = [
                                item.get("logprob", -float('inf')) 
                                for item in top_logprobs_for_each_token[0] 
                                if isinstance(item, dict)
                            ]
                            if not all_top_values_for_first_token: # If top_logprobs list is empty
                                is_greedy_for_this_choice = False # Cannot confirm
                            # Check if the actual logprob is "close enough" to the max of top logprobs
                            # Accounts for potential floating point inaccuracies.
                            elif not any(abs(first_token_actual_logprob - top_val) < 1e-6 for top_val in all_top_values_for_first_token if first_token_actual_logprob >= top_val - 1e-6):
                                # More robust check: is our token's logprob the maximum among the top candidates?
                                if first_token_actual_logprob < (max(all_top_values_for_first_token) - 1e-6):
                                     is_greedy_for_this_choice = False
                        else: 
                            # custom_api_logger.debug(f"Cannot determine greediness for choice {choice_idx}: top_logprobs missing or malformed for first token.")
                            is_greedy_for_this_choice = True # Default to True if cannot verify, or False if strictness needed
                else: 
                    custom_api_logger.warning(f"Choice {choice_idx} has no valid 'logprobs' object or 'content' list. Choice: {str(choice)[:200]}")
                    summed_token_logprobs_for_this_choice = -float('inf')
                    is_greedy_for_this_choice = False
                
                res.append((summed_token_logprobs_for_this_choice, is_greedy_for_this_choice))
                # custom_api_logger.debug(f"LlamaCppCompatibleAPI: Parsed logprobs for choice {choice_idx}: sum={summed_token_logprobs_for_this_choice}, greedy={is_greedy_for_this_choice}")
        return res

    # parse_generations can be inherited if LocalCompletionsAPI's version is fine.
    # If llama.cpp server's "text" field for completions is standard, no override needed.
    # @staticmethod
    # def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
    #     return LocalCompletionsAPI.parse_generations(outputs, **kwargs)


# --- Qwen3 specific class, now inheriting from LlamaCppCompatibleCompletionsAPI ---
@register_model("custom_qwen3_local_api") # Renamed for clarity, was "custom-local-completions-qwen"
class Qwen3LocalCompletionsAPI(LlamaCppCompatibleCompletionsAPI): # Inherits from our patched class
    """
    Extends LlamaCppCompatibleCompletionsAPI to add Qwen3-specific '/no_think'
    prefixing logic for certain tasks.
    """
    def __init__(self, 
                 model_alias_for_qwen_check: Optional[str] = None, 
                 no_prefix_tasks_str: str = "humaneval;mbpp", # Default tasks to EXCLUDE prefix
                 **kwargs):
        # Call the __init__ of LlamaCppCompatibleCompletionsAPI
        super().__init__(**kwargs) 
        
        self.effective_model_name_for_log = model_alias_for_qwen_check if model_alias_for_qwen_check else self.model
        self.is_qwen3_model_for_no_think = False

        if self.effective_model_name_for_log and "qwen3" in self.effective_model_name_for_log.lower():
            self.is_qwen3_model_for_no_think = True
            custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Qwen3LocalCompletionsAPI Initialized. Conditional '/no_think' prefixing enabled."
            )
        else:
            custom_api_logger.debug(
                f"({self.effective_model_name_for_log}): Qwen3LocalCompletionsAPI Initialized. NOT Qwen3. '/no_think' prefixing disabled."
            )

        self.no_prefix_tasks_set = set()
        if self.is_qwen3_model_for_no_think and no_prefix_tasks_str:
            self.no_prefix_tasks_set = {task.strip().lower() for task in no_prefix_tasks_str.split(';') if task.strip()} 
            if self.no_prefix_tasks_set:
                 custom_api_logger.info(
                    f"({self.effective_model_name_for_log}): Tasks excluded from '/no_think' prefix: {self.no_prefix_tasks_set}"
                )
        elif self.is_qwen3_model_for_no_think: # Qwen3 model but no exclusion string
             custom_api_logger.info(
                f"({self.effective_model_name_for_log}): No tasks specified for '/no_think' exclusion. Prefix may apply to all relevant tasks."
            )


    def _should_prefix_prompt_for_task(self, task_name: Optional[str]) -> bool:
        if not self.is_qwen3_model_for_no_think or not task_name:
            return False 
        
        # Normalize task_name for comparison
        return task_name.lower() not in self.no_prefix_tasks_set

    def _conditionally_prefix_requests(self, requests: List[Instance], is_generate_until: bool) -> List[Instance]:
        if not self.is_qwen3_model_for_no_think:
            return requests # No prefixing if not a Qwen3 model

        processed_requests: List[Instance] = []
        prefix_applied_count = 0
        prefix_skipped_due_to_tokenization = 0

        for i, instance in enumerate(requests):
            task_name = instance.task_name 
            current_context = instance.arguments[0] # First element is the prompt/context
            new_context = current_context
            
            if self._should_prefix_prompt_for_task(task_name):
                if isinstance(current_context, str):
                    if not current_context.startswith("/no_think"): # Avoid double-prefixing
                        new_context = "/no_think " + current_context
                        prefix_applied_count += 1
                elif isinstance(current_context, list) and getattr(self, 'tokenized_requests', False):
                    # This warning is important if tokenized_requests=True is ever used with this model type
                    custom_api_logger.warning(
                        f"({self.effective_model_name_for_log}): Task '{task_name}' (Request {i+1}) is Qwen3 and requires prefix, "
                        "but is using 'tokenized_requests=True'. Conditional prepending of '/no_think' to tokenized "
                        "prompts is not automatically handled by this class."
                    )
                    prefix_skipped_due_to_tokenization += 1
            
            if new_context is not current_context:
                # arguments is a tuple: (context_str, gen_kwargs_dict_or_continuation_str)
                # The second element of the tuple instance.arguments[1] contains the rest.
                new_arguments_tuple = (new_context, instance.arguments[1]) 
                processed_requests.append(dataclasses.replace(instance, arguments=new_arguments_tuple))
            else:
                processed_requests.append(instance)
        
        if prefix_applied_count > 0 or prefix_skipped_due_to_tokenization > 0 :
            custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Processed {len(requests)} requests for {'generate_until' if is_generate_until else 'loglikelihood'}. "
                f"'/no_think' prefix applied to {prefix_applied_count} requests. "
                f"Skipped for tokenized: {prefix_skipped_due_to_tokenization}."
            )
        return processed_requests

    # Override loglikelihood and generate_until to apply the conditional prefixing
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prefixed_requests = self._conditionally_prefix_requests(requests, is_generate_until=False)
        # Call the loglikelihood from the parent class (LlamaCppCompatibleCompletionsAPI)
        return super().loglikelihood(prefixed_requests)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prefixed_requests = self._conditionally_prefix_requests(requests, is_generate_until=True)
        # Call generate_until from the parent class (LlamaCppCompatibleCompletionsAPI)
        return super().generate_until(prefixed_requests)