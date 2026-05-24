from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    condition: str
    target: str | None = None
    topic: str | None = None


FACTUAL_CASES: list[PromptCase] = [
    PromptCase("The capital of France is [MASK].", "factual", "paris", "capital"),
    PromptCase("The capital of Germany is [MASK].", "factual", "berlin", "capital"),
    PromptCase("The capital of Italy is [MASK].", "factual", "rome", "capital"),
    PromptCase("The capital of Spain is [MASK].", "factual", "madrid", "capital"),
    PromptCase("The capital of Japan is [MASK].", "factual", "tokyo", "capital"),
    PromptCase("The capital of China is [MASK].", "factual", "beijing", "capital"),
    PromptCase("The capital of Russia is [MASK].", "factual", "moscow", "capital"),
    PromptCase("The capital of Canada is [MASK].", "factual", "ottawa", "capital"),
    PromptCase("The capital of Brazil is [MASK].", "factual", "brasilia", "capital"),
    PromptCase("The capital of Egypt is [MASK].", "factual", "cairo", "capital"),
    PromptCase("The capital of Greece is [MASK].", "factual", "athens", "capital"),
    PromptCase("The capital of India is [MASK].", "factual", "delhi", "capital"),
    PromptCase("Water freezes at zero degrees [MASK].", "factual", "celsius", "science"),
    PromptCase("The chemical symbol for water is [MASK].", "factual", "h2o", "science"),
    PromptCase("The largest planet is [MASK].", "factual", "jupiter", "science"),
    PromptCase("The nearest star to Earth is the [MASK].", "factual", "sun", "science"),
    PromptCase("A triangle has three [MASK].", "factual", "sides", "math"),
    PromptCase("A square has four [MASK].", "factual", "sides", "math"),
    PromptCase("Two plus two equals [MASK].", "factual", "four", "math"),
    PromptCase("Five minus two equals [MASK].", "factual", "three", "math"),
    PromptCase("The opposite of hot is [MASK].", "factual", "cold", "lexical"),
    PromptCase("The opposite of up is [MASK].", "factual", "down", "lexical"),
    PromptCase("The opposite of left is [MASK].", "factual", "right", "lexical"),
    PromptCase("The plural of child is [MASK].", "factual", "children", "lexical"),
    PromptCase("The plural of mouse is [MASK].", "factual", "mice", "lexical"),
    PromptCase("A person who teaches is a [MASK].", "factual", "teacher", "lexical"),
    PromptCase("A person who drives a bus is a [MASK].", "factual", "driver", "lexical"),
    PromptCase("The color of grass is usually [MASK].", "factual", "green", "common"),
    PromptCase("The color of the sky is often [MASK].", "factual", "blue", "common"),
    PromptCase("People usually drink coffee from a [MASK].", "factual", "cup", "common"),
]


AMBIGUOUS_CASES: list[PromptCase] = [
    PromptCase("The person went to the [MASK].", "ambiguous", None, "open"),
    PromptCase("The meeting was held in the [MASK].", "ambiguous", None, "open"),
    PromptCase("The child looked at the [MASK].", "ambiguous", None, "open"),
    PromptCase("She put the book on the [MASK].", "ambiguous", None, "open"),
    PromptCase("They walked toward the [MASK].", "ambiguous", None, "open"),
    PromptCase("The artist painted the [MASK].", "ambiguous", None, "open"),
    PromptCase("The team discussed the [MASK].", "ambiguous", None, "open"),
    PromptCase("He waited near the [MASK].", "ambiguous", None, "open"),
    PromptCase("The story was about a [MASK].", "ambiguous", None, "open"),
    PromptCase("The old building had a [MASK].", "ambiguous", None, "open"),
    PromptCase("The bank was beside the [MASK].", "ambiguous", None, "polysemy"),
    PromptCase("The bat flew out of the [MASK].", "ambiguous", None, "polysemy"),
    PromptCase("The coach talked to the [MASK].", "ambiguous", None, "polysemy"),
    PromptCase("The match ended near the [MASK].", "ambiguous", None, "polysemy"),
    PromptCase("The patient waited for the [MASK].", "ambiguous", None, "role"),
    PromptCase("The customer asked for a [MASK].", "ambiguous", None, "open"),
    PromptCase("The student opened the [MASK].", "ambiguous", None, "open"),
    PromptCase("The city is known for its [MASK].", "ambiguous", None, "open"),
    PromptCase("The company announced a new [MASK].", "ambiguous", None, "open"),
    PromptCase("The scientist studied the [MASK].", "ambiguous", None, "open"),
]


def all_prompt_cases() -> list[PromptCase]:
    return FACTUAL_CASES + AMBIGUOUS_CASES


def extended_prompt_cases() -> list[PromptCase]:
    """A larger controlled cloze suite.

    The suite is generated from explicit fact/open-template lists and kept in
    code so the exact prompts are reproducible. Factual targets that are not a
    single tokenizer token are filtered later by `token_id_for_word`.
    """
    capital_pairs = [
        ("France", "paris"),
        ("Italy", "rome"),
        ("Spain", "madrid"),
        ("Japan", "tokyo"),
        ("China", "beijing"),
        ("Russia", "moscow"),
        ("Egypt", "cairo"),
        ("Greece", "athens"),
        ("Portugal", "lisbon"),
        ("Austria", "vienna"),
        ("Belgium", "brussels"),
        ("Denmark", "copenhagen"),
        ("Finland", "helsinki"),
        ("Norway", "oslo"),
        ("Sweden", "stockholm"),
        ("Poland", "warsaw"),
        ("Hungary", "budapest"),
        ("Romania", "bucharest"),
        ("Bulgaria", "sofia"),
        ("Serbia", "belgrade"),
        ("Croatia", "zagreb"),
        ("Turkey", "ankara"),
        ("Iran", "tehran"),
        ("Iraq", "baghdad"),
        ("Syria", "damascus"),
        ("Jordan", "amman"),
        ("Lebanon", "beirut"),
        ("Thailand", "bangkok"),
        ("Vietnam", "hanoi"),
        ("Peru", "lima"),
        ("Chile", "santiago"),
        ("Cuba", "havana"),
        ("Jamaica", "kingston"),
        ("Morocco", "rabat"),
        ("Tunisia", "tunis"),
        ("Ghana", "accra"),
        ("Nigeria", "abuja"),
        ("Senegal", "dakar"),
        ("Uganda", "kampala"),
        ("Rwanda", "kigali"),
        ("Nepal", "kathmandu"),
        ("Pakistan", "islamabad"),
    ]
    fact_cases = [
        PromptCase(f"The capital of {country} is [MASK].", "factual", capital, "capital")
        for country, capital in capital_pairs
    ]
    fact_specs = [
        ("Water freezes at zero degrees [MASK].", "celsius", "science"),
        ("The chemical symbol for water is [MASK].", "h2o", "science"),
        ("The largest planet is [MASK].", "jupiter", "science"),
        ("The nearest star to Earth is the [MASK].", "sun", "science"),
        ("The red planet is [MASK].", "mars", "science"),
        ("Humans breathe [MASK].", "oxygen", "science"),
        ("Plants use sunlight for [MASK].", "photosynthesis", "science"),
        ("A triangle has three [MASK].", "sides", "math"),
        ("A square has four [MASK].", "sides", "math"),
        ("Two plus two equals [MASK].", "four", "math"),
        ("Five minus two equals [MASK].", "three", "math"),
        ("Ten divided by two equals [MASK].", "five", "math"),
        ("The opposite of hot is [MASK].", "cold", "lexical"),
        ("The opposite of up is [MASK].", "down", "lexical"),
        ("The opposite of left is [MASK].", "right", "lexical"),
        ("The opposite of day is [MASK].", "night", "lexical"),
        ("The opposite of yes is [MASK].", "no", "lexical"),
        ("The plural of child is [MASK].", "children", "lexical"),
        ("The plural of mouse is [MASK].", "mice", "lexical"),
        ("A person who teaches is a [MASK].", "teacher", "role"),
        ("A person who drives a bus is a [MASK].", "driver", "role"),
        ("A person who writes books is an [MASK].", "author", "role"),
        ("A person who studies science is a [MASK].", "scientist", "role"),
        ("A person who cooks food is a [MASK].", "chef", "role"),
        ("The color of grass is usually [MASK].", "green", "common"),
        ("The color of the sky is often [MASK].", "blue", "common"),
        ("People usually drink coffee from a [MASK].", "cup", "common"),
        ("People sleep in a [MASK].", "bed", "common"),
        ("Birds usually fly in the [MASK].", "sky", "common"),
        ("Fish usually swim in [MASK].", "water", "common"),
        ("A dog is an [MASK].", "animal", "common"),
        ("A rose is a [MASK].", "flower", "common"),
        ("Bread is made from [MASK].", "flour", "common"),
        ("Milk comes from a [MASK].", "cow", "common"),
        ("The language of France is [MASK].", "french", "language"),
        ("The language of Spain is [MASK].", "spanish", "language"),
        ("The language of Germany is [MASK].", "german", "language"),
        ("The language of Italy is [MASK].", "italian", "language"),
    ]
    fact_cases.extend(
        PromptCase(prompt, "factual", target, topic) for prompt, target, topic in fact_specs
    )

    subjects = [
        "person",
        "student",
        "teacher",
        "artist",
        "scientist",
        "doctor",
        "child",
        "customer",
        "coach",
        "driver",
        "writer",
        "worker",
        "visitor",
        "patient",
        "engineer",
        "musician",
        "chef",
        "farmer",
        "pilot",
        "lawyer",
    ]
    actions = [
        "looked at",
        "walked toward",
        "waited near",
        "talked about",
        "opened",
        "closed",
        "painted",
        "studied",
        "discussed",
        "carried",
        "found",
        "lost",
        "visited",
        "remembered",
        "watched",
        "heard",
        "needed",
        "bought",
        "fixed",
        "changed",
    ]
    places = [
        "school",
        "office",
        "station",
        "garden",
        "market",
        "hospital",
        "museum",
        "theater",
        "airport",
        "library",
    ]
    ambiguous_cases: list[PromptCase] = []
    for idx, subject in enumerate(subjects):
        ambiguous_cases.append(
            PromptCase(f"The {subject} {actions[idx % len(actions)]} the [MASK].", "ambiguous", None, "open")
        )
    for idx, action in enumerate(actions):
        ambiguous_cases.append(
            PromptCase(f"Yesterday they {action} the [MASK].", "ambiguous", None, "open")
        )
    for place in places:
        ambiguous_cases.extend(
            [
                PromptCase(f"The meeting happened near the [MASK] in the {place}.", "ambiguous", None, "open"),
                PromptCase(f"The story about the {place} mentioned a [MASK].", "ambiguous", None, "open"),
                PromptCase(f"People at the {place} asked for a [MASK].", "ambiguous", None, "open"),
            ]
        )
    ambiguous_cases.extend(AMBIGUOUS_CASES)

    seen: set[str] = set()
    result: list[PromptCase] = []
    for case in fact_cases + ambiguous_cases:
        if case.prompt not in seen:
            seen.add(case.prompt)
            result.append(case)
    return result


def token_id_for_word(tokenizer, word: str) -> int | None:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def encode_prompt(tokenizer, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if len(mask_positions) != 1:
        raise ValueError(f"Expected exactly one mask token in prompt: {prompt}")
    return encoded, int(mask_positions[0, 0].item())


def hidden_states_and_logits(model, encoded: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)
    return output.hidden_states, output.logits


def _extended_attention_mask(model, attention_mask: torch.Tensor) -> torch.Tensor:
    return model.bert.get_extended_attention_mask(
        attention_mask,
        attention_mask.shape,
        dtype=next(model.parameters()).dtype,
    )


def logits_from_layer_state(model, state: torch.Tensor, attention_mask: torch.Tensor, layer: int) -> torch.Tensor:
    h = state
    extended_attention_mask = _extended_attention_mask(model, attention_mask)
    for block in model.bert.encoder.layer[layer:]:
        output = block(h, attention_mask=extended_attention_mask)
        h = output[0] if isinstance(output, tuple) else output
    return model.cls(h)


def jacobian_topk_logits_wrt_mask(
    model,
    state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer: int,
    mask_index: int,
    top_token_ids: np.ndarray,
) -> np.ndarray:
    """True Jacobian of selected MLM logits wrt the [MASK] hidden vector."""
    model.eval()
    state = state.detach()
    mask_vector = state[0, mask_index, :].detach().clone().requires_grad_(True)
    fixed_state = state.clone()
    top_ids = torch.as_tensor(top_token_ids, dtype=torch.long, device=state.device)

    def selected_logits(mask_hidden: torch.Tensor) -> torch.Tensor:
        local_state = fixed_state.clone()
        local_state[0, mask_index, :] = mask_hidden
        logits = logits_from_layer_state(model, local_state, attention_mask, layer)
        return logits[0, mask_index, top_ids]

    jac = torch.autograd.functional.jacobian(
        selected_logits,
        mask_vector,
        create_graph=False,
        strict=False,
        vectorize=False,
    )
    return jac.detach().cpu().numpy().astype(np.float64)
