from typing import Any, Dict, List, Optional

try:
	import dspy  # type: ignore
except Exception:
	# DSPy is optional at import-time; we only use type names here
	dspy = None  # type: ignore

from agent_torch.core.llm.template import Template
from agent_torch.core.llm.Variable import Variable


def _clean_skill_name(name: str) -> str:
	return (
		name.lower()
		.replace(" ", "_")
		.replace("/", "_")
		.replace("-", "_")
		.replace("(", "_")
		.replace(")", "_")
		.replace("&", "and")
	)


def project_to_template(module: Any, skill_names: List[str], title: Optional[str] = None) -> Template:
	"""
	Create a Template instance from a DSPy module "shape" and a list of skill names.
	- Registers one Variable per skill (binary include/omit)
	- Provides a scaffold header referencing the module and docstring (if present)
	- Renders a content block that respects sparse rows and P3O choices via Template.render_job_info
	"""
	class ProjectedTemplate(Template):
		def __init__(self, module_ref: Any, skill_names_ref: List[str], header_title: Optional[str] = None) -> None:
			super().__init__()
			self._dspy_module = module_ref
			self.skill_names_used = list(skill_names_ref)
			self._header_title = header_title or module_ref.__class__.__name__

			# create Variables for all skills and register dynamically
			skill_variables: Dict[str, Variable] = {}
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				skill_variables[attr_name] = Variable(
					desc=f"Include {skill_name}",
					learnable=True,
					presentations=["", "- {value}"]
				)
			self.register_variables(skill_variables)

		def __prompt__(self) -> str:
			module_name = getattr(self._dspy_module, "__class__", type(self._dspy_module)).__name__
			doc = (getattr(self._dspy_module, "__doc__", None) or "").strip()
			header_lines: List[str] = [
				f"DSPy-Projected Scaffold: {self._header_title}",
				f"Module: {module_name}",
			]
			if doc:
				header_lines.append(doc)
			header = "\n".join(header_lines)

			# Content block: one placeholder per skill variable
			skill_lines: List[str] = []
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				skill_lines.append(f"{{{attr_name}}}")
			skills_block = "\n".join(skill_lines)

			return (
				f"{header}\n\n"
				f"Relevant skills (conditionally included):\n"
				f"{skills_block}\n"
			)

		def configure(self, external_df) -> "ProjectedTemplate":
			self._external_df = external_df
			return self

	return ProjectedTemplate(module, skill_names, title)


def from_dspy(
    module: Any,
    slots: List[str],
    title: Optional[str] = None,
    *,
    marker: Optional[str] = None,
    include_attributes_block: bool = True,
    block_header: Optional[str] = None,
) -> Template:
	"""
	Convert a DSPy module into a Template with the given slots, embedding scaffold.
	- Extracts scaffold (instruction + demos) from module.predictor when available
	- Registers one learnable Variable per slot
	- Renders a content block that respects sparse rows and P3O choices via Template.render_job_info
	"""
	owner = getattr(module, "predictor", module)
	module_name = getattr(module, "__class__", type(module)).__name__
	doc = (getattr(module, "__doc__", None) or "").strip()
	header_title = title or module_name

	# Extract scaffold
	instruction = getattr(owner, "instruction", None)
	if isinstance(instruction, (list, tuple)):
		instruction_text = "\n".join(str(x) for x in instruction)
	else:
		instruction_text = str(instruction) if instruction is not None else ""

	demos = getattr(owner, "demos", None)
	if demos is None:
		demos = getattr(owner, "fewshot", None)
	if isinstance(demos, (list, tuple)):
		demo_texts = [str(d) for d in demos]
	else:
		demo_texts = []

	# Gather signature IO names if available
	input_names: List[str] = []
	output_names: List[str] = []
	sig = getattr(owner, "signature", None)
	if sig is not None:
		inputs_dict = getattr(sig, "input_fields", {})
		outputs_dict = getattr(sig, "output_fields", {})
		input_names = list(inputs_dict.keys())
		output_names = list(outputs_dict.keys())

	class DspyScaffoldTemplate(Template):
		def __init__(self, module_ref: Any, slot_names: List[str], header: str, instr: str, demo_list: List[str]) -> None:
			super().__init__()
			self._dspy_module = module_ref
			self.skill_names_used = list(slot_names)
			self._header_title = header
			self._instruction = instr
			self._demos = list(demo_list)
			self._input_names = input_names
			self._output_names = output_names
			self._marker = marker
			self._include_block = bool(include_attributes_block)
			self._block_header = block_header or "Relevant attributes (conditionally included):"

			# Variables per slot
			skill_variables: Dict[str, Variable] = {}
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				skill_variables[attr_name] = Variable(
					desc=f"Include {skill_name}",
					learnable=True,
					presentations=["", "- {value}"]
				)
			self.register_variables(skill_variables)

		def _skills_block(self) -> str:
            # Slot placeholders assembled as lines (resolved by Template engine)
			lines: List[str] = []
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				lines.append(f"{{{attr_name}}}")
			return "\n".join(lines)

		def __system_prompt__(self) -> str:
			lines: List[str] = [
				f"DSPy Scaffold: {self._header_title}",
				f"Module: {module_name}",
			]
			if doc:
				lines.append(doc)
			if self._instruction:
				lines.append("")
				lines.append(self._instruction)
			if self._input_names:
				lines.append("")
				lines.append(f"Inputs: {', '.join(self._input_names)}")
			sys_text = "\n".join(lines)
			# Optional in-body insertion in system text
			if self._marker and self._marker in sys_text:
				block_parts = [self._block_header, self._skills_block()]
				block_text = "\n".join(p for p in block_parts if p)
				try:
					return sys_text.replace(self._marker, block_text)
				except Exception:
					return sys_text
			return sys_text

		def __prompt__(self) -> str:
			# Optional few-shot section
			fewshot = ""
			if self._demos:
				fewshot = "Few-shot examples:\n" + "\n".join(self._demos) + "\n\n"

			# Build content body according to marker/include flag
			skills_block = self._skills_block()
			body = ""
			if self._include_block:
				body = f"{self._block_header}\n{skills_block}\n"

			# If marker is intended for system, __prompt__ just carries few-shot + optional body
			# (When marker is used in system, we still allow appended body only if include flag is True.)
			return f"{fewshot}{body}"

		def __output__(self) -> str:
			if self._output_names:
				return f"Provide outputs for: {', '.join(self._output_names)}. Output only those."
			return "Provide the required outputs."

	return DspyScaffoldTemplate(module, slots, header_title, instruction_text, demo_texts)

def from_predict(
    module: Any,
    slots: List[str],
    title: Optional[str] = None,
    categories: Optional[List[str]] = None,
    input_field: Optional[str] = None,
    output_field: Optional[str] = None,
    *,
    marker: Optional[str] = None,
    include_attributes_block: bool = True,
    block_header: Optional[str] = None,
) -> Template:
	"""
	Create a Template from a DSPy Predict-like module and explicit IO field names.
	- input_field: the input name in the DSPy signature (e.g., "job_info")
	- output_field: the output name (e.g., "job_metrics")
	- categories: optional keys for JSON output in __output__ (if dict-like)
	"""
	assert hasattr(module, "__call__") or hasattr(module, "forward"), "module must be a callable DSPy module"

	# Infer IO field names and title from module signature if not provided
	owner = getattr(module, "predictor", module)
	signature = getattr(owner, "signature")
	# Expect DSPy Signature to expose input_fields/output_fields dicts
	if input_field is None or output_field is None:
		inputs_dict = getattr(signature, "input_fields")
		outputs_dict = getattr(signature, "output_fields")
		in_names = list(inputs_dict.keys())
		out_names = list(outputs_dict.keys())
		assert len(in_names) >= 1 and len(out_names) >= 1, "Signature must define at least one input and one output field"
		if input_field is None:
			input_field = in_names[0]
		if output_field is None:
			output_field = out_names[0]
	if title is None:
		title = getattr(module, "__class__", type(module)).__name__

	# Extract scaffold if present
	instruction = getattr(owner, "instruction", None)
	if isinstance(instruction, (list, tuple)):
		instruction_text = "\n".join(str(x) for x in instruction)
	else:
		instruction_text = str(instruction) if instruction is not None else ""
	demos = getattr(owner, "demos", None)
	if demos is None:
		demos = getattr(owner, "fewshot", None)
	demo_texts = [str(d) for d in demos] if isinstance(demos, (list, tuple)) else []

	class PredictProjectedTemplate(Template):
		def __init__(self, module_ref: Any, slot_names: List[str], header_title: Optional[str]) -> None:
			super().__init__()
			self._dspy_module = module_ref
			self.skill_names_used = list(slot_names)
			self._header_title = header_title or module_ref.__class__.__name__
			self._instruction = instruction_text
			self._demos = demo_texts
			self._marker = marker
			self._include_block = bool(include_attributes_block)
			self._block_header = block_header or "Relevant attributes (conditionally included):"

			# Create Variables (binary include/omit) and register dynamically
			skill_variables: Dict[str, Variable] = {}
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				skill_variables[attr_name] = Variable(
					desc=f"Include {skill_name}",
					learnable=True,
					presentations=["", "- {value}"]
				)
			self.register_variables(skill_variables)

		def _skills_block(self) -> str:
			lines: List[str] = []
			for skill_name in self.skill_names_used:
				attr_name = _clean_skill_name(skill_name)
				lines.append(f"{{{attr_name}}}")
			return "\n".join(lines)

		def __system_prompt__(self) -> str:
			module_name = getattr(self._dspy_module, "__class__", type(self._dspy_module)).__name__
			doc = (getattr(self._dspy_module, "__doc__", None) or "").strip()
			lines: List[str] = [
				f"DSPy Scaffold: {self._header_title}",
				f"Module: {module_name}",
			]
			if doc:
				lines.append(doc)
			if self._instruction:
				lines.append("")
				lines.append(self._instruction)
			# Optional: list inputs for context
			lines.append("")
			lines.append(f"Inputs: {', '.join(in_names)}")
			sys_text = "\n".join(lines)
			if self._marker and self._marker in sys_text:
				block_parts = [self._block_header, self._skills_block()]
				block_text = "\n".join(p for p in block_parts if p)
				try:
					return sys_text.replace(self._marker, block_text)
				except Exception:
					return sys_text
			return sys_text

		def __prompt__(self) -> str:
			# Optional few-shot section
			fewshot = ""
			if self._demos:
				fewshot = "Few-shot examples:\n" + "\n".join(self._demos) + "\n\n"

			# Content block placeholders; actual values come from external_df via render_job_info
			skills_block = self._skills_block()
			body = ""
			if self._include_block:
				body = f"{self._block_header}\n{skills_block}\n"
			return f"{fewshot}{body}"

		def __output__(self) -> str:
			# If categories provided, emit a strict JSON scaffold; else a concise generic instruction from signature
			if categories:
				body: List[str] = []
				for i, cat in enumerate(categories):
					comma = "," if i < len(categories) - 1 else ""
					body.append(f'  "{cat}": <YOUR_NUMERIC_PREDICTION>{comma}')
				joined = "\n".join(body)
				return (
					"ONLY OUTPUT THIS JSON. DO NOT OUTPUT ANYTHING ELSE!!!!\n\n"
					"{\n" + joined + "\n}"
				)
			return f"Provide outputs for: {', '.join(out_names)}. Output only those."

		def configure(self, external_df) -> "PredictProjectedTemplate":
			self._external_df = external_df
			return self

	return PredictProjectedTemplate(module, slots, title)


# --- Round-robin utilities (general-purpose) ---

def build_selection_block(slot_universe: List[str], selections: Dict[str, int], header: str = "Attributes:") -> str:
	"""
	Render a human-readable attributes block using the raw slot labels gated by binary selections
	keyed by cleaned variable names (same cleaning as used for Template variables).
	"""
	cleaned_to_raw: Dict[str, str] = {_clean_skill_name(s): s for s in slot_universe}
	lines: List[str] = [header]
	for cleaned_name, raw in cleaned_to_raw.items():
		if selections.get(cleaned_name, 0) == 1:
			lines.append(f"- {raw}")
	return "\n".join(lines)


def inject_block_into_examples(
	examples: List[Any],
	input_field: str,
	block_text: str,
	marker: Optional[str] = None,
) -> List[Any]:
	"""
	Return new DSPy examples with the block injected into the given input field.
	If marker is provided and present, replace it; otherwise append the block.
	"""
	import dspy  # type: ignore

	injected: List[Any] = []
	for ex in examples:
		base_text = getattr(ex, input_field)
		if marker and marker in base_text:
			new_text = base_text.replace(marker, block_text)
		else:
			sep = "\n\n" if base_text and not str(base_text).endswith("\n") else ""
			new_text = f"{base_text}{sep}{block_text}"
		# Preserve outputs if present
		fields: Dict[str, Any] = {input_field: new_text}
		for k, v in ex.__dict__.items():
			if k != input_field:
				fields[k] = v
		injected.append(dspy.Example(**fields).with_inputs(input_field))
	return injected


def get_input_field_from_module(module: Any) -> str:
	owner = getattr(module, "predictor", module)
	sig = getattr(owner, "signature")
	in_names = list(getattr(sig, "input_fields").keys())
	assert len(in_names) >= 1, "Signature must define at least one input field"
	return in_names[0]
