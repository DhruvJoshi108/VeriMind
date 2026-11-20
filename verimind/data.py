"""Synthetic dataset generation for offline NLI training and evaluation."""

from __future__ import annotations


def build_synthetic_dataset() -> dict[str, list]:
    """Create balanced, varied labeled claim-evidence pairs for the local baseline."""
    examples = [
        ("Paris is the capital of France.", "Paris is the capital city of France.", "entailment"),
        ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France.", "entailment"),
        ("Berlin is Germany's capital.", "The capital of Germany is Berlin.", "entailment"),
        ("London is the capital of the United Kingdom.", "London is the capital city of the UK.", "entailment"),
        ("France is in Europe.", "France is a European country.", "entailment"),
        ("Water freezes at 0 degrees Celsius.", "At sea level, water freezes at zero degrees Celsius.", "entailment"),
        ("The human heart pumps blood.", "The heart pumps blood around the body.", "entailment"),
        ("The Moon orbits Earth.", "Earth is orbited by the Moon.", "entailment"),
        ("Paris is the capital of France.", "Paris is the capital of Germany.", "contradiction"),
        ("The Eiffel Tower is in Berlin.", "The Eiffel Tower is in Paris, France.", "contradiction"),
        ("Berlin is Germany's capital.", "Berlin is the capital of France.", "contradiction"),
        ("Water freezes at 100 degrees Celsius.", "Water freezes at 0 degrees Celsius.", "contradiction"),
        ("The Moon does not orbit Earth.", "The Moon orbits Earth.", "contradiction"),
        ("France is in Asia.", "France is a country in Europe.", "contradiction"),
        ("London is the capital of France.", "London is the capital of the United Kingdom.", "contradiction"),
        ("Humans have three hearts.", "Humans have one heart.", "contradiction"),
        ("Paris is the capital of France.", "The Eiffel Tower is in Paris.", "neutral"),
        ("The Eiffel Tower is in Paris.", "France is a country in Europe.", "neutral"),
        ("Water freezes at 0 degrees Celsius.", "The Moon orbits Earth.", "neutral"),
        ("Berlin is Germany's capital.", "London is in the United Kingdom.", "neutral"),
        ("The human heart pumps blood.", "Paris has the Louvre Museum.", "neutral"),
        ("France is in Europe.", "Water is made of hydrogen and oxygen.", "neutral"),
        ("The Moon orbits Earth.", "The Eiffel Tower was built in France.", "neutral"),
        ("London is the capital of the United Kingdom.", "Berlin is a city in Germany.", "neutral"),
        ("Sicily is the largest island in the Mediterranean Sea.", "Sicily is an island in the Mediterranean.", "entailment"),
        ("Sicily is the largest island in the Mediterranean Sea.", "Crete is the largest island in the Mediterranean.", "contradiction"),
        ("The Moon does not orbit Earth.", "The Moon orbits the Earth.", "contradiction"),
        ("A Perfect Circle is a country supergroup.", "A Perfect Circle is an American rock supergroup.", "contradiction"),
        ("Ringo Starr experienced two life-threatening illnesses during a Beatles tour.", "Ringo Starr toured with the Beatles.", "neutral"),
        ("Clint Eastwood is an American singer.", "Clint Eastwood is an American actor and filmmaker.", "contradiction"),
        ("Katie Price is currently married to Kieran Hayler.", "Katie Price was married to Kieran Hayler.", "entailment"),
        ("Loving was directed by Nick Kroll.", "Loving was directed by Jeff Nichols.", "contradiction"),
        ("Black Sabbath formed in Dublin.", "Black Sabbath formed in Birmingham, England.", "contradiction"),
        ("YouTube is headquartered in San Bruno, California.", "YouTube is headquartered in San Bruno, California.", "entailment"),
        ("Barbara Stanwyck starred in Ball of Fire.", "Barbara Stanwyck appeared in the film Ball of Fire.", "entailment"),
        ("Eva Longoria is a Bostonian.", "Eva Longoria is an American actress.", "neutral"),
        ("Rabat is in eastern Morocco.", "Rabat is the capital of Morocco.", "neutral"),
        ("The Wolf of Wall Street (2013 film) is a film.", "The Wolf of Wall Street is a 2013 film directed by Martin Scorsese.", "entailment"),
        ("Peyton Manning's middle name is Williams.", "Peyton Manning is an American former football quarterback.", "neutral"),
        ("Vera Farmiga is only an actress.", "Vera Farmiga is an actress and director.", "contradiction"),
        ("Calvin Harris is from India.", "Calvin Harris is from Scotland.", "contradiction"),
        ("Grace Jones inspired an Irish restaurant.", "Grace Jones is a Jamaican-American model and singer.", "neutral"),
    ]
    return {
        "pairs": [(claim, evidence) for claim, evidence, _ in examples],
        "labels": [label for _, _, label in examples],
    }
