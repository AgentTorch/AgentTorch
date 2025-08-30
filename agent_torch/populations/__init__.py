"""
Population utilities
====================

Top-level helpers to view and edit population mapping.json files without
per-population wrappers. Pure population-module utilities (no Template dependency).
"""

import os
import json
import pickle
from typing import Any, List, Optional, Dict, Tuple
from collections import Counter
try:
    import pandas as _pd  # Optional; used only for nicer previews
except Exception:  # pragma: no cover
    _pd = None
import sys
import importlib
import pkgutil
from functools import partial
import shutil


def _truncate_text(value: Any, *, max_chars: int = 120) -> str:
    try:
        s = str(value)
    except Exception:
        return "<unprintable>"
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _truncate_list(values: List[Any], *, max_items: int = 3, max_chars: int = 120) -> List[str]:
    out: List[str] = []
    for v in values[:max_items]:
        out.append(_truncate_text(v, max_chars=max_chars))
    if len(values) > max_items:
        out.append("…")
    return out


def _truncate_dict(d: Dict[str, Any], *, max_keys: int = 6, max_chars: int = 120) -> Dict[str, str]:
    items = list(d.items())[:max_keys]
    out: Dict[str, str] = {}
    for k, v in items:
        out[str(k)] = _truncate_text(v, max_chars=max_chars)
    if len(d) > max_keys:
        out["…"] = "…"
    return out
    

def _resolve_population_dir(population_or_dir: Any) -> str:
    """Resolve a population module/loader/path into a directory path string."""
    # population module: has __path__
    if hasattr(population_or_dir, '__path__'):
        try:
            return population_or_dir.__path__[0]
        except Exception:
            return str(population_or_dir)
    # loader instance with population_folder_path
    if hasattr(population_or_dir, 'population_folder_path'):
        return str(population_or_dir.population_folder_path)
    # already a path
    return str(population_or_dir)


def view_mappings(
    population_or_dir: Any,
    *,
    fields: Optional[List[str]] = None,
    max_values: int = 10,
    markdown: bool = False,
    sample_n: int = 5,
    show_pkls: bool = True,
    pkl_filter: Optional[List[str]] = None,
) -> str:
    """
    View mapping.json for a population (module, loader, or directory path).

    Args:
        population_or_dir: population module, loader with population_folder_path, or directory path
        fields: optional subset of fields to show
        max_values: number of sample values to display per field
        markdown: render as markdown table if True; plain text otherwise

    Returns:
        Rendered string (also printed).
    """
    base_dir = _resolve_population_dir(population_or_dir)
    mapping_path = os.path.join(base_dir, 'mapping.json')

    if not os.path.exists(mapping_path):
        msg = f"No mapping.json found at: {mapping_path}"
        print(msg)
        return msg

    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping: Dict[str, Any] = json.load(f)
    except Exception as e:
        msg = f"Failed to read mapping.json: {e}"
        print(msg)
        return msg

    # Optionally filter fields
    items = list(mapping.items())
    if fields:
        want = set(fields)
        items = [(k, v) for k, v in items if k in want]

    # Build rows: Field | Count | Sample Values
    rows: List[List[str]] = []
    for field, values in items:
        try:
            count = len(values) if hasattr(values, '__len__') else 0
        except Exception:
            count = 0
        sample_list: List[str] = []
        if isinstance(values, list):
            for label in values[:max_values]:
                sample_list.append(str(label))
            if count > max_values:
                sample_list.append('…')
        sample_str = ", ".join(sample_list)
        rows.append([str(field), str(count), sample_str])

    if markdown:
        output_lines = ["| Field | Count | Sample Values |", "|-------|-------:|---------------|"]
        for field, count, sample in rows:
            sample_safe = sample.replace('|', '\\|')
            output_lines.append(f"| {field} | {count} | {sample_safe} |")
        result = "\n".join(output_lines)
        print(result)
    else:
        # Plain aligned text
        field_w = max([len(r[0]) for r in rows] + [5])
        count_w = max([len(r[1]) for r in rows] + [5])
        header = f"{'Field'.ljust(field_w)}  {'Count'.rjust(count_w)}  Sample Values"
        sep = f"{'-'*field_w}  {'-'*count_w}  {'-'*12}"
        lines = [header, sep]
        for field, count, sample in rows:
            lines.append(f"{field.ljust(field_w)}  {count.rjust(count_w)}  {sample}")
        result = "\n".join(lines)
        print(result)

    # Optionally sample PKL files
    detected_sizes: List[Tuple[str, int]] = []
    if show_pkls:
        try:
            print("\nPKL samples (first", sample_n, "):")
            for fname in sorted(os.listdir(base_dir)):
                if not (fname.lower().endswith('.pkl') or fname.lower().endswith('.pickle')):
                    continue
                if pkl_filter and fname not in pkl_filter:
                    continue
                fpath = os.path.join(base_dir, fname)
                try:
                    with open(fpath, 'rb') as pf:
                        obj = pickle.load(pf)
                    size = len(obj) if hasattr(obj, '__len__') else None

                    # Build a compact, truncated sample preview string
                    if isinstance(obj, (list, tuple)):
                        sample_seq = list(obj)[:sample_n]
                        sample_preview_str = str(_truncate_list(sample_seq, max_items=sample_n, max_chars=120))
                    elif _pd is not None and isinstance(obj, _pd.Series):
                        sample_seq = obj.head(sample_n).tolist()
                        sample_preview_str = str(_truncate_list(sample_seq, max_items=sample_n, max_chars=120))
                    elif _pd is not None and isinstance(obj, _pd.DataFrame):
                        head_df = obj.head(sample_n)
                        records = head_df.to_dict(orient='records')
                        truncated_records = [_truncate_dict(r, max_keys=min(6, len(r)), max_chars=80) for r in records]
                        sample_seq = records
                        sample_preview_str = json.dumps(truncated_records, ensure_ascii=False)
                    elif hasattr(obj, '__getitem__') and isinstance(size, int):
                        try:
                            sample_seq = [obj[i] for i in range(min(sample_n, size))]
                            sample_preview_str = str(_truncate_list(sample_seq, max_items=sample_n, max_chars=120))
                        except Exception:
                            sample_seq = []
                            sample_preview_str = _truncate_text(type(obj), max_chars=120)
                    else:
                        sample_seq = []
                        sample_preview_str = _truncate_text(type(obj), max_chars=120)

                    print(f"- {fname}: size={size if size is not None else 'N/A'}")
                    print("  sample:", sample_preview_str)
                    # Note if first N values are identical (common with encoded categories)
                    try:
                        if isinstance(sample_seq, list) and len(sample_seq) > 1:
                            if all(v == sample_seq[0] for v in sample_seq[1:]):
                                print("  note: first", len(sample_seq), "values identical; this may indicate encoded categories")
                    except Exception:
                        pass
                    print("")
                    if isinstance(size, int):
                        detected_sizes.append((fname, size))
                except Exception as e:
                    print(f"- {fname}: <failed to load> ({e})\n")
        except Exception as e:
            print(f"Failed PKL sampling: {e}")

    # Detect and print inferred population size
    pop_size_msg = ""
    if detected_sizes:
        counts = Counter(size for _, size in detected_sizes)
        most_common_size, _ = counts.most_common(1)[0]
        # Find representative file
        rep = next((fn for fn, sz in detected_sizes if sz == most_common_size), detected_sizes[0][0])
        pop_size_msg = f"Inferred population size: {most_common_size} (from {rep})"
        print(pop_size_msg)

    # Return only the mapping table string; extras are printed
    return result



def add_mapping_field(population_or_dir: Any, key: str, values: List[str]) -> str:
    """
    Add or update a field → labels entry in mapping.json.

    Args:
        population_or_dir: population module, loader with population_folder_path, or directory path
        key: field name to add/update
        values: list of labels in index order

    Returns:
        Status message string
    """
    base_dir = _resolve_population_dir(population_or_dir)
    mapping_path = os.path.join(base_dir, 'mapping.json')

    mapping: Dict[str, Any] = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f) or {}
        except Exception:
            mapping = {}

    # Coerce values to list[str]
    if not isinstance(values, list):
        values = list(values)
    values = [str(v) for v in values]

    mapping[str(key)] = values
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        msg = f"Added mapping field '{key}' with {len(values)} labels to {mapping_path}"
    except Exception as e:
        msg = f"Failed to write mapping.json at {mapping_path}: {e}"
    print(msg)
    return msg


def set_mappings_from_json(population_or_dir: Any, json_path: str) -> str:
    """
    Replace the entire mapping.json with the contents of a provided JSON file.
    """
    base_dir = _resolve_population_dir(population_or_dir)
    mapping_path = os.path.join(base_dir, 'mapping.json')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            new_mapping = json.load(f)
        if not isinstance(new_mapping, dict):
            raise ValueError("Provided JSON must be an object/dict at top level")
    except Exception as e:
        msg = f"Failed to read provided JSON '{json_path}': {e}"
        print(msg)
        return msg

    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(new_mapping, f, ensure_ascii=False, indent=2)
        msg = f"Replaced mapping.json at {mapping_path} from {json_path}"
    except Exception as e:
        msg = f"Failed to write mapping.json at {mapping_path}: {e}"
    print(msg)
    return msg


def replace_mappings(population_or_dir: Any, mapping: Dict[str, Any]) -> str:
    """
    Replace the entire mapping.json with the provided mapping dict.
    """
    base_dir = _resolve_population_dir(population_or_dir)
    mapping_path = os.path.join(base_dir, 'mapping.json')

    if not isinstance(mapping, dict):
        msg = "Provided mapping must be a dict"
        print(msg)
        return msg
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        msg = f"Replaced mapping.json at {mapping_path}"
    except Exception as e:
        msg = f"Failed to write mapping.json at {mapping_path}: {e}"
    print(msg)
    return msg


def _infer_population_size(base_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Infer expected population size by inspecting existing pickle files."""
    preferred = [
        'age.pickle', 'age.pkl',
        'region.pickle', 'region.pkl',
        'household.pickle', 'household.pkl',
    ]
    for fname in preferred:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'rb') as f:
                    obj = pickle.load(f)
                if hasattr(obj, '__len__'):
                    return len(obj), fname
            except Exception:
                continue
    # Fallback: any pickle
    for fname in os.listdir(base_dir):
        if not (fname.lower().endswith('.pkl') or fname.lower().endswith('.pickle')):
            continue
        fpath = os.path.join(base_dir, fname)
        try:
            with open(fpath, 'rb') as f:
                obj = pickle.load(f)
            if hasattr(obj, '__len__'):
                return len(obj), fname
        except Exception:
            continue
    return None, None


def add_data(
    population_or_dir: Any,
    src_path: str,
    dest_name: Optional[str] = None,
    overwrite: bool = False,
    require_match: bool = True,
) -> str:
    """
    Add a .pkl/.pickle file into the population directory.

    Args:
        population_or_dir: population module, loader with population_folder_path, or directory path
        src_path: path to source .pkl/.pickle file
        dest_name: optional destination filename (defaults to basename of src)
        overwrite: overwrite if file exists

    Returns:
        Status message string
    """
    base_dir = _resolve_population_dir(population_or_dir)
    if not os.path.isfile(src_path):
        msg = f"Source file not found: {src_path}"
        print(msg)
        return msg
    lower = src_path.lower()
    if not (lower.endswith('.pkl') or lower.endswith('.pickle')):
        msg = "Only .pkl/.pickle files are supported"
        print(msg)
        return msg

    filename = dest_name if dest_name else os.path.basename(src_path)
    if not (filename.lower().endswith('.pkl') or filename.lower().endswith('.pickle')):
        # enforce extension if custom name lacks it
        filename = f"{filename}.pkl"

    dest_path = os.path.join(base_dir, filename)
    if os.path.exists(dest_path) and not overwrite:
        msg = f"File already exists at destination: {dest_path} (use overwrite=True)"
        print(msg)
        return msg

    # Verify length match if requested
    if require_match:
        expected_len, ref_file = _infer_population_size(base_dir)
        if expected_len is not None:
            try:
                with open(src_path, 'rb') as sf:
                    src_obj = pickle.load(sf)
                if hasattr(src_obj, '__len__'):
                    actual_len = len(src_obj)
                    if actual_len != expected_len:
                        msg = (
                            f"Length mismatch: expected {expected_len} from {ref_file}, "
                            f"got {actual_len} in {src_path}. Aborting."
                        )
                        print(msg)
                        return msg
            except Exception as e:
                msg = f"Failed to validate source file length: {e}"
                print(msg)
                return msg

    try:
        shutil.copy2(src_path, dest_path)
        msg = f"Copied {src_path} -> {dest_path}"
    except Exception as e:
        msg = f"Failed to copy file: {e}"
    print(msg)
    return msg


__all__ = [
    "view_mappings",
    "add_mapping_field",
    "set_mappings_from_json",
    "replace_mappings",
    "add_data",
]

# ---------------------------------------------------------------------------
# Auto-attach helpers to each population submodule for ergonomic usage:
#   import agent_torch.populations.astoria as astoria
#   astoria.view_mappings(...), astoria.add_mapping_field(...), etc.
# ---------------------------------------------------------------------------

def _attach_helpers_to_module(module):
    try:
        # Avoid double-attachment
        if getattr(module, "__agent_torch_helpers_attached__", False):
            return

        def _vm(*, fields: Optional[List[str]] = None, max_values: int = 10, markdown: bool = False, _mod=module):
            return view_mappings(_mod, fields=fields, max_values=max_values, markdown=markdown)

        def _amf(key: str, values: List[str], _mod=module):
            return add_mapping_field(_mod, key, values)

        def _smfj(json_path: str, _mod=module):
            return set_mappings_from_json(_mod, json_path)

        def _sm(mapping: Dict[str, Any], _mod=module):
            return replace_mappings(_mod, mapping)
        
        def _ad(src_path: str, dest_name: Optional[str] = None, overwrite: bool = False, _mod=module):
            return add_data(_mod, src_path, dest_name=dest_name, overwrite=overwrite)


        setattr(module, "view_mappings", _vm)
        setattr(module, "add_mapping_field", _amf)
        setattr(module, "set_mappings_from_json", _smfj)
        setattr(module, "replace_mappings", _sm)
        setattr(module, "add_data", _ad)
        setattr(module, "__agent_torch_helpers_attached__", True)

        
    except Exception:
        # Best-effort; don't fail package import if a submodule is odd
        pass


def _eager_attach_to_subpackages():
    try:
        for finder, name, ispkg in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
            if not ispkg:
                continue
            full_name = f"{__name__}.{name}"
            try:
                module = importlib.import_module(full_name)
                _attach_helpers_to_module(module)
            except Exception:
                # Skip modules that fail to import; helpers will be missing only for those
                continue
    except Exception:
        # Namespace/path issues shouldn't block import
        pass


_eager_attach_to_subpackages()


