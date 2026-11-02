"""Pydantic v2 strict output contract; never coerces or repairs labels locally."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationInfo, model_validator


class BFEOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    labels: list[Annotated[StrictInt, Field(ge=0, le=1)]]
    reasons: list[Annotated[StrictStr, Field(min_length=1)]]

    @model_validator(mode='after')
    def aligned(self, info: ValidationInfo):
        expected = (info.context or {}).get('expected_steps')
        if len(self.labels) != len(self.reasons):
            raise ValueError(f'labels/reasons lengths differ: {len(self.labels)} vs {len(self.reasons)}')
        if expected is not None and len(self.labels) != expected:
            raise ValueError(f'Expected exactly {expected} labels and reasons; received {len(self.labels)}')
        if not self.labels or any(not s.strip() for s in self.reasons):
            raise ValueError('Labels must be nonempty and every reason must contain non-whitespace text')
        return self


def validate_output(body, expected_steps):
    return BFEOutput.model_validate(body, strict=True, context={'expected_steps': expected_steps})


def schema_for_steps(n):
    schema = BFEOutput.model_json_schema()
    for key in ('labels', 'reasons'):
        schema['properties'][key]['minItems'] = n
        schema['properties'][key]['maxItems'] = n
    return schema
