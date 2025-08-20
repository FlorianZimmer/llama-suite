#!/usr/bin/env python3
from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Union, Optional, Any
from operator import itemgetter
import dataclasses

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.api.instance import Instance

from typing import Any, Dict, List, Optional, Union, cast

# Shared logger
custom_api_logger = logging.getLogger("lm_eval.custom_local_api")


# ------------------------------ helpers ------------------------------

def _combine_stops_with_eos(
    raw_stops: Optional[Union[str, List[str]]],
    eos: Optional[str],
) -> Optional[Union[str, List[str]]]:
    """
    Combine a raw stop spec (str|list|None) with an EOS string (optional).
    Returns None if both are None.
    """
    if raw_stops is None and not eos:
        return None

    # Normalize to list
    stops_list: List[str] = []
    if isinstance(raw_stops, str):
        stops_list = [raw_stops]
    elif isinstance(raw_stops, list):
        stops_list = list(raw_stops)

    if eos:
        if eos not in stops_list:
            stops_list.append(eos)

    if not stops_list:
        return None
    if len(stops_list) == 1:
        return stops_list[0]
    return stops_list


# ------------------------------ base model ------------------------------

@register_model("llama_cpp_compatible_api")
class LlamaCppCompatibleCompletionsAPI(LocalCompletionsAPI):
    def __init__(self,
        *,
        num_concurrent: int = 1,
        timeout: int = 60,
        max_retries: int = 3,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            num_concurrent=int(num_concurrent),
            timeout=int(timeout),
            max_retries=int(max_retries),
            batch_size=int(batch_size),
            **kwargs,
        )
        self.AUTO_TOKENIZE = False
        custom_api_logger.info(
            "Initialized LlamaCppCompatibleCompletionsAPI (model=%s) "
            "num_concurrent=%s timeout=%s max_retries=%s batch_size=%s",
            self.model, num_concurrent, timeout, max_retries, batch_size
        )

    def _create_payload(  # type: ignore[override]
        self,
        messages: Union[List[int], List[dict], List[str], str],
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        custom_api_logger.debug(
            "LlamaCppCompatibleAPI._create_payload IN. generate=%s, type(messages)=%s, preview=%s",
            generate,
            type(messages),
            (str(messages)[:200] if not isinstance(messages, list) else str(messages[:1])[:200]),
        )

        # ---- normalize prompt to a *str* in all cases ----
        prompt_to_send: str

        if isinstance(messages, str):
            prompt_to_send = messages

        elif isinstance(messages, list) and messages:
            # List[int] — tokenized prompt
            if all(isinstance(it, int) for it in messages):
                if getattr(self, "tokenizer", None) is not None:
                    try:
                        prompt_to_send = self.tokenizer.decode(messages)  # type: ignore[attr-defined]
                    except Exception as e:
                        custom_api_logger.warning("Decode failed for List[int]; fallback to str(...): %s", e)
                        prompt_to_send = str(messages)
                else:
                    custom_api_logger.warning("List[int] without tokenizer; fallback to str(...).")
                    prompt_to_send = str(messages)

            # List[List[int]] — decode first sub-list if possible
            elif isinstance(messages[0], list) and all(isinstance(x, int) for x in messages[0]):
                first_tokens = cast(List[int], messages[0])
                if getattr(self, "tokenizer", None) is not None:
                    try:
                        prompt_to_send = self.tokenizer.decode(first_tokens)  # type: ignore[attr-defined]
                    except Exception as e:
                        custom_api_logger.warning("Decode failed for List[List[int]]; fallback to str(...): %s", e)
                        prompt_to_send = str(first_tokens)
                else:
                    custom_api_logger.warning("List[List[int]] without tokenizer; fallback to str(...).")
                    prompt_to_send = str(first_tokens)

            # List[str] — common wrapper for a single prompt
            elif all(isinstance(s, str) for s in messages):
                prompt_to_send = cast(List[str], messages)[0]

            elif all(isinstance(m, dict) for m in messages):
                ms = cast(List[Dict[str, Any]], messages)
                contents = [m.get("content", "") for m in ms]
                prompt_to_send = "\n".join(c for c in contents if isinstance(c, str) and c) or str(ms)


            # List[dict] — light chat-ish inputs
            elif all(isinstance(m, dict) for m in messages):
                contents = [cast(dict, m).get("content", "") for m in messages]
                prompt_to_send = "\n".join(c for c in contents if isinstance(c, str) and c) or str(messages)

            else:
                custom_api_logger.warning(
                    "Unsupported messages shape %s; using str(...) fallback.",
                    [type(m) for m in messages[:2]]
                )
                prompt_to_send = str(messages)

        else:
            custom_api_logger.warning("Unexpected/empty messages type %s; using str(...) fallback.", type(messages))
            prompt_to_send = str(messages)

        # ---- build payloads as before (unchanged) ----
        gen_kwargs = dict(gen_kwargs or {})
        if generate:
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_gen_toks", self._max_gen_toks))
            temperature = gen_kwargs.pop("temperature", 0.0)

            raw_stop_sequences = gen_kwargs.pop("until", None)
            combined_stops = _combine_stops_with_eos(raw_stop_sequences, eos)
            final_stop_sequences = (
                handle_stop_sequences(combined_stops, eos)
                if (combined_stops is not None or eos is not None)
                else None
            )

            payload = {
                "prompt": prompt_to_send,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                **({"stop": final_stop_sequences} if final_stop_sequences else {}),
                **gen_kwargs,
            }
            return payload

        return {
            "model": self.model,
            "prompt": prompt_to_send,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 1,
            "seed": seed,
        }


    # -------- llama.cpp logprob parsing --------
    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        tokens: Optional[List[List[int]]] = None,
        ctxlens: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[Tuple[float, bool]]:
        """
        Parse llama.cpp style logprobs for loglikelihood evaluation.
        Returns a list of (sum_logprob, is_greedy) per request.
        """
        res: List[Tuple[float, bool]] = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out_idx, out in enumerate(outputs):
            choices = out.get("choices") if isinstance(out, dict) else None
            if not choices:
                custom_api_logger.warning("No 'choices' in output %d: %s", out_idx, str(out)[:200])
                res.append((-float("inf"), False))
                continue

            for choice_idx, choice in enumerate(sorted(choices, key=itemgetter("index"))):
                logprobs_obj = choice.get("logprobs")
                sum_lp = 0.0
                greedy = True

                if isinstance(logprobs_obj, dict) and isinstance(logprobs_obj.get("content"), list):
                    token_logprobs: List[float] = []
                    top_for_each: List[Optional[List[Dict[str, Any]]]] = []

                    for content_item_idx, content_item in enumerate(logprobs_obj["content"]):
                        if isinstance(content_item, dict) and "logprob" in content_item:
                            token_logprobs.append(content_item["logprob"])
                            top = content_item.get("top_logprobs")
                            top_for_each.append(top if isinstance(top, list) else None)
                        else:
                            custom_api_logger.warning(
                                "Malformed content_item %d in choice %d: %s",
                                content_item_idx,
                                choice_idx,
                                content_item,
                            )
                            token_logprobs.append(-float("inf"))
                            top_for_each.append(None)

                    if not token_logprobs:
                        sum_lp = -float("inf")
                        greedy = False
                    else:
                        sum_lp = sum(token_logprobs)
                        # Greedy check for first token using top_logprobs
                        if top_for_each and top_for_each[0]:
                            first_lp = token_logprobs[0]
                            top_vals = [
                                d.get("logprob", -float("inf"))
                                for d in top_for_each[0]  # type: ignore[index]
                                if isinstance(d, dict)
                            ]
                            if top_vals and first_lp < (max(top_vals) - 1e-6):
                                greedy = False
                        else:
                            greedy = True
                else:
                    custom_api_logger.warning(
                        "Choice %d has no valid 'logprobs' or 'content'. Choice: %s",
                        choice_idx,
                        str(choice)[:200],
                    )
                    sum_lp = -float("inf")
                    greedy = False

                res.append((sum_lp, greedy))

        return res


# ------------------------------ Qwen3 adapter ------------------------------

@register_model("custom_qwen3_local_api")
class Qwen3LocalCompletionsAPI(LlamaCppCompatibleCompletionsAPI):
    def __init__(self,
        *,
        model_alias_for_qwen_check: Optional[str] = None,
        no_prefix_tasks_str: str = "humaneval;mbpp",
        num_concurrent: int = 1,
        timeout: int = 60,
        max_retries: int = 3,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            num_concurrent=num_concurrent,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            **kwargs,
        )

        self.effective_model_name_for_log: str = model_alias_for_qwen_check or self.model
        self.is_qwen3_model_for_no_think: bool = "qwen3" in self.effective_model_name_for_log.lower()

        if self.is_qwen3_model_for_no_think:
            custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Qwen3LocalCompletionsAPI initialized. "
                "Conditional '/no_think' prefixing enabled."
            )
        else:
            custom_api_logger.debug(
                f"({self.effective_model_name_for_log}): Qwen3LocalCompletionsAPI initialized. "
                "Not Qwen3; '/no_think' prefixing disabled."
            )

        # Normalize task names to lowercase for comparison
        self.no_prefix_tasks_set = {
            t.strip().lower() for t in (no_prefix_tasks_str or "").split(";") if t.strip()
        }
        if self.is_qwen3_model_for_no_think:
            if self.no_prefix_tasks_set:
                custom_api_logger.info(
                    f"({self.effective_model_name_for_log}): Tasks excluded from '/no_think' prefix: "
                    f"{self.no_prefix_tasks_set}"
                )
            else:
                custom_api_logger.info(
                    f"({self.effective_model_name_for_log}): No tasks specified for '/no_think' exclusion."
                )
        custom_api_logger.info(
            "(%s): Qwen3 adapter ready (num_concurrent=%s, timeout=%s, max_retries=%s)",
            self.effective_model_name_for_log, num_concurrent, timeout, max_retries
        )

    # ---- internal helpers ----
    def _should_prefix_prompt_for_task(self, task_name: Optional[str]) -> bool:
        if not self.is_qwen3_model_for_no_think or not task_name:
            return False
        return task_name.lower() not in self.no_prefix_tasks_set

    from typing import Any, List, Tuple

    def _conditionally_prefix_requests(
        self, requests: List[Instance], *, is_generate_until: bool
    ) -> List[Instance]:
        if not self.is_qwen3_model_for_no_think:
            return requests

        processed: List[Instance] = []
        applied = 0
        skipped_tok = 0

        for i, inst in enumerate(requests):
            task_name = getattr(inst, "task_name", None)

            # keep the original tuple shape; don't assume 2 elements
            args: List[Any] = list(inst.arguments)
            if args:
                current_context: Any = args[0]

                if self._should_prefix_prompt_for_task(task_name):
                    if isinstance(current_context, str):
                        if not current_context.startswith("/no_think"):
                            args[0] = "/no_think " + current_context
                            applied += 1
                    elif isinstance(current_context, list) and getattr(self, "tokenized_requests", False):
                        # tokenized prompt; don't try to string-prepend
                        skipped_tok += 1
                    # elif isinstance(current_context, dict):  # optional: extract "content"
                    #     content = current_context.get("content")
                    #     if isinstance(content, str) and not content.startswith("/no_think"):
                    #         args[0] = "/no_think " + content
                    #         applied += 1

            processed.append(dataclasses.replace(inst, arguments=tuple(args)))

        if applied or skipped_tok:
            custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Processed {len(requests)} for "
                f"{'generate_until' if is_generate_until else 'loglikelihood'}; "
                f"applied '/no_think' to {applied}, skipped (tokenized) {skipped_tok}."
            )
        return processed


    # ---- overrides with matching base signatures ----
    def loglikelihood(self, requests: List[Instance], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:  # type: ignore[override]
        prefixed = self._conditionally_prefix_requests(requests, is_generate_until=False)
        return super().loglikelihood(prefixed, disable_tqdm=disable_tqdm)

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:  # type: ignore[override]
        prefixed = self._conditionally_prefix_requests(requests, is_generate_until=True)
        return super().generate_until(prefixed, disable_tqdm=disable_tqdm)
