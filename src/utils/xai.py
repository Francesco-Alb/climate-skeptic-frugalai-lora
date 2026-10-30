import re
import numpy as np
import torch
import transformers
import shap
import warnings
from typing import Any
from pathlib import Path


def build_explainer(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: str | torch.device,
    padding: bool | str = "max_length",
    max_length: int = 500,
    truncation: bool | str = True,
) -> shap.Explainer:
    """
    Build a SHAP explainer for the given model and tokenizer.

    Args:
        model: The trained model to explain.
        tokenizer: The tokenizer used for the model.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
        padding: The padding strategy to use.
        max_length: The maximum length of the input sequences.
        truncation: Whether to truncate sequences longer than max_length.

    Returns:
        explainer: A SHAP explainer object.
    """ 

    # define a prediction function
    def f(x: list[str] | np.ndarray) -> np.ndarray:
        tv = torch.tensor([tokenizer.encode(v, padding=padding, max_length=max_length, truncation=truncation) for v in x]).to(device)
        with torch.no_grad():
            outputs = model(tv)
            return outputs.logits.detach().cpu().numpy()

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)
    
    return explainer


def save_shap_text_plot(
    shap_values: shap.Explanation,
    filename: str | Path = Path("shap_text_plot.html"),
    sample_index: int | str | None = 0,
    class_id: int | None = None
) -> None:
    """
    Save a SHAP text plot to an HTML file.

    Args:
        shap_values: The SHAP Explanation object.
        filename: Path to save the HTML output.
        sample_index: Index of the sample to plot (e.g. 0, 1, -1 for last).
                    * Set to 'all' or None to plot all samples in shap_values. Default is 0.
                    * MAX_SAFE_SAMPLES is set to 10 to avoid generating large files by accident.
        class_id: The class ID to isolate. If None, plots all classes for the sample(s).
    """

    MAX_SAFE_SAMPLES = 10

    if isinstance(filename, str):
        filename = Path(filename)
        
    is_all_samples = sample_index in ("all", None) or (
        isinstance(sample_index, str) and sample_index.lower() == "all"
    )

    if is_all_samples:
        num_samples = len(shap_values)
        if num_samples > MAX_SAFE_SAMPLES:
            raise ValueError(
                f"Cannot plot all {num_samples} samples at once (limit is {MAX_SAFE_SAMPLES}). "
                f"Please specify a specific `sample_index` (e.g., sample_index=0) to avoid giant HTML files."
            )
        explanation = shap_values
    else:
        explanation = shap_values[sample_index]

    if class_id is not None:
        explanation = explanation[..., class_id]

    # Generate the raw HTML text from SHAP
    raw_html = shap.plots.text(explanation, display=False)

    # Fix NumPy 2.0 type injections using a Regular Expression
    cleaned_html = re.sub(r"np\.float\d+\(([^)]+)\)", r"\1", raw_html)

    # Explicitly bundle the required SHAP JavaScript headers
    final_html = f"<head>{shap.getjs()}</head><body>{cleaned_html}</body>"

    if not filename.parent.exists():
        warnings.warn(f"Directory {filename.parent} does not exist. Creating it.")
        filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(final_html)