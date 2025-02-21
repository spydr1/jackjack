"""Backbones configurations."""
import dataclasses
from typing import Optional

from official.modeling import hyperparams
from jackjack.super_resolution.legacy.drct.config import DRCT


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    drct: drct backbone config.

  """
  type: Optional[str] = None
  drct: DRCT = dataclasses.field(default_factory=DRCT)
